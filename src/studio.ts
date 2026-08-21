/**
 * Studio: record/upload your own samples, label them, train, then run live
 * inference where every prediction fires the configured action.
 *
 * Two modalities share one pipeline — only the capture widget and the extractor
 * differ:
 *   image: camera frame or file  -> embedPhoto (116-d)
 *   audio: mic recording or file -> embedAudio (200-d)
 */
import { Pipeline, confusionMatrix, type Mat, type Vec } from './linalg.js';
import { separationRatio, runChecks } from './verify.js';
import { store, hasIDB, type StoredModel } from './store.js';
import { samples, trainability, makeThumb, findDuplicates, dedupe,
         type Sample, type Modality } from './samples.js';
import { toRecord, fromRecord, toJSON, classify } from './serialize.js';
import { dispatch, defaultConfig, type ActionConfig, type Sink } from './actions.js';
import { embedPhoto, PHOTO_DIM, fileToSurface, videoToSurface } from './photo.js';
import { embedAudio, AUDIO_DIM, decodeAudioFile, rmsDb, resample, toMono } from './audio.js';
import type { Surface } from './dataset.js';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

let logEl: HTMLElement;
const log = (m: string, cls = '') => {
  const t = new Date().toLocaleTimeString();
  logEl.innerHTML += `\n<span class="dim">${t}</span> ${cls ? `<span class="${cls}">${m}</span>` : m}`;
  logEl.scrollTop = logEl.scrollHeight;
};

const EXTRACTOR: Record<Modality, { name: string; dim: number }> = {
  image: { name: 'photo', dim: PHOTO_DIM },
  audio: { name: 'mel', dim: AUDIO_DIM },
};

let modality: Modality = 'image';
let project = 'default';
let active: { rec: StoredModel; pipe: Pipeline } | null = null;
let config: ActionConfig = defaultConfig([]);
let stream: MediaStream | null = null;
let recorder: MediaRecorder | null = null;
let meterTimer: number | null = null;

// ---------------------------------------------------------------- features
async function featurise(s: Sample): Promise<Vec> {
  const ex = EXTRACTOR[s.modality].name;
  if (s.vecs?.[ex]) return Float64Array.from(s.vecs[ex]);
  let v: Vec;
  if (s.modality === 'image') {
    v = embedPhoto(await fileToSurface(s.blob));
  } else {
    const { samples: pcm, sampleRate } = await decodeAudioFile(s.blob);
    v = embedAudio(pcm, sampleRate);
  }
  s.vecs = { ...(s.vecs ?? {}), [ex]: Array.from(v).map(x => +x.toFixed(6)) };
  if (s.id !== undefined) await samples.update(s);   // cache so retrain is fast
  return v;
}

// ---------------------------------------------------------------- capture
async function startCamera() {
  await stopCapture();
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'environment', width: { ideal: 640 } }, audio: false,
  });
  const v = $<HTMLVideoElement>('cam');
  v.srcObject = stream;
  await v.play();
  $('camWrap').style.display = '';
  log('camera on', 'g');
}

async function startMic() {
  await stopCapture();
  stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
  $('micWrap').style.display = '';
  // live level meter, so the user can see the mic is actually working
  const AC: any = (window as any).AudioContext || (window as any).webkitAudioContext;
  const ctx = new AC();
  const src = ctx.createMediaStreamSource(stream);
  const an = ctx.createAnalyser();
  an.fftSize = 1024;
  src.connect(an);
  const buf = new Float32Array(an.fftSize);
  meterTimer = window.setInterval(() => {
    an.getFloatTimeDomainData(buf);
    const db = rmsDb(buf);
    const pctv = Math.max(0, Math.min(100, (db + 60) / 60 * 100));
    $('meterBar').style.width = pctv + '%';
    $('meterVal').textContent = `${db.toFixed(0)} dB`;
  }, 100);
  log('microphone on', 'g');
}

async function stopCapture() {
  if (meterTimer) { clearInterval(meterTimer); meterTimer = null; }
  if (recorder && recorder.state !== 'inactive') recorder.stop();
  recorder = null;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  $('camWrap').style.display = 'none';
  $('micWrap').style.display = 'none';
}

/** Record a fixed-length clip from the live mic stream. */
function recordClip(ms: number): Promise<Blob> {
  return new Promise((res, rej) => {
    if (!stream) return rej(new Error('microphone not started'));
    const chunks: Blob[] = [];
    let mr: MediaRecorder;
    try { mr = new MediaRecorder(stream); } catch (e) { return rej(e as Error); }
    recorder = mr;
    mr.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
    mr.onstop = () => res(new Blob(chunks, { type: mr.mimeType || 'audio/webm' }));
    mr.onerror = (e: any) => rej(e?.error ?? new Error('recorder error'));
    mr.start();
    setTimeout(() => { if (mr.state !== 'inactive') mr.stop(); }, ms);
  });
}

function canvasToBlob(cv: HTMLCanvasElement): Promise<Blob> {
  return new Promise(res => cv.toBlob(b => res(b!), 'image/jpeg', 0.85));
}

// ---------------------------------------------------------------- add samples
async function addSample(blob: Blob, source: string, label: string, durationMs?: number) {
  if (!label) { log('pick or type a label first', 'r'); return; }
  const s: Sample = {
    project, modality, label, createdAt: Date.now(), source, blob, durationMs,
  };
  if (modality === 'image') {
    try { s.thumb = await makeThumb(blob); } catch { /* preview is optional */ }
  }
  await samples.add(s);
  log(`+1 "${label}" (${source})`, 'g');
  await refreshSamples();
}

// ---------------------------------------------------------------- sample list
async function refreshSamples() {
  const counts = await samples.labels(project, modality);
  const t = trainability(counts);
  const rows = await samples.list(project, modality);

  $('labelChips').innerHTML = Object.keys(counts).sort().map(l =>
    `<span class="chip" data-label="${l}">${l} <b>${counts[l]}</b></span>`).join('')
    || '<span class="dim">no samples yet</span>';

  const dup = findDuplicates(rows, EXTRACTOR[modality].name);
  const dupNote = dup.duplicateCount
    ? ` · <span class="r">${dup.duplicateCount} duplicate sample(s)</span>` +
      ` <button id="dedupe" class="sec" style="padding:3px 9px;font-size:11px">remove</button>`
    : '';
  $('trainability').innerHTML = (t.ok
    ? `<span class="g">ready</span> <span class="dim">${t.reason}</span>`
    : `<span class="o">not ready</span> <span class="dim">${t.reason}</span>`) + dupNote;
  const ddBtn = document.getElementById('dedupe');
  if (ddBtn) ddBtn.addEventListener('click', async () => {
    const n = await dedupe(rows, EXTRACTOR[modality].name);
    log(`removed ${n} duplicate sample(s)`, 'o');
    await refreshSamples();
  });
  $<HTMLButtonElement>('trainBtn').disabled = !t.ok;

  $('grid').innerHTML = rows.slice(-40).map(r => {
    const inner = r.modality === 'image' && r.thumb
      ? `<img src="${r.thumb}" alt="">`
      : `<div class="aud">♪ ${r.durationMs ? (r.durationMs / 1000).toFixed(1) + 's' : 'clip'}</div>`;
    return `<div class="cell" data-id="${r.id}">${inner}<span>${r.label}</span>
      <button class="rm" data-id="${r.id}" title="delete">×</button></div>`;
  }).join('');

  $('grid').querySelectorAll<HTMLButtonElement>('.rm').forEach(b =>
    b.addEventListener('click', async e => {
      e.stopPropagation();
      await samples.remove(+b.dataset.id!);
      log('sample deleted', 'o');
      await refreshSamples();
    }));

  // clicking a chip fills the label box — faster than retyping
  $('labelChips').querySelectorAll<HTMLElement>('.chip').forEach(c =>
    c.addEventListener('click', () => {
      $<HTMLInputElement>('label').value = c.dataset.label!;
    }));
}

// ---------------------------------------------------------------- train
async function train() {
  const rows = await samples.list(project, modality);
  const counts = await samples.labels(project, modality);
  const t = trainability(counts);
  if (!t.ok) { log(t.reason, 'r'); return; }
  if (!t.reason.startsWith(`${t.labels.length}`)) log(t.reason, 'o');

  log(`featurising ${rows.length} samples…`);
  const labels = t.labels;
  const X: Mat = [], y: number[] = [], keys: string[] = [];
  for (const r of rows) {
    X.push(await featurise(r));
    y.push(labels.indexOf(r.label));
    keys.push(String(r.id));
    await new Promise(res => setTimeout(res, 0));      // keep the page responsive
  }

  // Hold out ~25% per label, stratified, so the score means something.
  const byLabel = new Map<number, number[]>();
  y.forEach((c, i) => { if (!byLabel.has(c)) byLabel.set(c, []); byLabel.get(c)!.push(i); });
  const teIdx = new Set<number>();
  for (const idxs of byLabel.values()) {
    const nTest = Math.max(1, Math.floor(idxs.length * 0.25));
    // take every k-th so the split is deterministic, not random per run
    const step = Math.max(1, Math.floor(idxs.length / nTest));
    for (let i = 0, taken = 0; i < idxs.length && taken < nTest; i += step, taken++) {
      teIdx.add(idxs[i]);
    }
  }
  const tr = [...X.keys()].filter(i => !teIdx.has(i));
  const te = [...teIdx];

  const Xtr = tr.map(i => X[i]), ytr = tr.map(i => y[i]);
  const Xte = te.map(i => X[i]), yte = te.map(i => y[i]);

  const pipe = new Pipeline({ epochs: 400 }).fit(Xtr, ytr, labels.length);
  const trainAcc = pipe.score(Xtr, ytr);
  const testAcc = Xte.length ? pipe.score(Xte, yte) : NaN;
  const ratio = Xte.length > 2 ? separationRatio(Xte, yte) : NaN;
  const cm = Xte.length
    ? confusionMatrix(yte, Xte.map(x => pipe.predict(x)), labels.length)
    : [];
  const checks = Xte.length >= labels.length * 2
    ? runChecks(Xtr, ytr, Xte, yte, tr.map(i => keys[i]), te.map(i => keys[i]), labels.length)
    : [];

  const rec = toRecord(pipe, {
    name: `${project}/${modality} · ${labels.join('|')}`,
    extractor: EXTRACTOR[modality].name,
    classes: labels,
    metrics: {
      train: trainAcc, test: Number.isNaN(testAcc) ? trainAcc : testAcc,
      ratio: Number.isNaN(ratio) ? 0 : ratio,
      nTrain: Xtr.length, nTest: Xte.length,
    },
  });
  if (hasIDB) await store.saveModel(rec);
  active = { rec, pipe };
  config = { ...defaultConfig(labels), ...config, targets: config.targets.filter(x => labels.includes(x)) };
  if (!config.targets.length) config.targets = [labels[0]];

  log(`trained ${labels.length} classes · train ${pct(trainAcc)} · ` +
      `test ${Number.isNaN(testAcc) ? 'n/a' : pct(testAcc)}`, 'g');
  renderMetrics(rec, cm, checks, labels);
  renderTargets(labels);
  await refreshModels();
  $<HTMLButtonElement>('runBtn').disabled = false;
}

function renderMetrics(rec: StoredModel, cm: number[][],
                       checks: { name: string; pass: boolean; detail: string }[],
                       labels: string[]) {
  const cmHtml = cm.length
    ? `<table class="cm"><tr><th>true ↓ pred →</th>${labels.map(l => `<th>${l}</th>`).join('')}</tr>` +
      cm.map((row, i) => `<tr><td>${labels[i]}</td>${row.map((v, j) =>
        `<td class="${v && i === j ? 'g' : v ? 'r' : 'dim'}">${v}</td>`).join('')}</tr>`).join('') +
      '</table>'
    : '<div class="dim" style="font-size:11.5px">too few samples for a held-out split — score is on training data</div>';
  $('metrics').innerHTML = `
    <div class="kpis">
      <div class="kpi"><div class="n ${rec.metrics.test > 0.8 ? 'g' : 'o'}">${pct(rec.metrics.test)}</div><div class="l">TEST</div></div>
      <div class="kpi"><div class="n dim">${pct(rec.metrics.train)}</div><div class="l">TRAIN</div></div>
      <div class="kpi"><div class="n b">${rec.dim}</div><div class="l">DIMS</div></div>
      <div class="kpi"><div class="n o">${rec.metrics.nTrain}/${rec.metrics.nTest}</div><div class="l">TRAIN/TEST</div></div>
    </div>${cmHtml}
    ${checks.length ? `<div class="checks">${checks.map(c =>
      `<div><span class="${c.pass ? 'g' : 'r'}">[${c.pass ? 'PASS' : 'FAIL'}]</span> ${c.name}` +
      `<span class="dim"> ${c.detail}</span></div>`).join('')}</div>` : ''}`;
}

function renderTargets(labels: string[]) {
  $('targets').innerHTML = labels.map(l =>
    `<label><input type="checkbox" class="tgt" value="${l}"` +
    `${config.targets.includes(l) ? ' checked' : ''}> ${l}</label>`).join('');
  $('targets').querySelectorAll('input').forEach(i =>
    i.addEventListener('change', syncConfig));
}

// ---------------------------------------------------------------- config
function syncConfig() {
  config = {
    targets: [...document.querySelectorAll<HTMLInputElement>('.tgt:checked')].map(i => i.value),
    threshold: +$<HTMLInputElement>('thr').value / 100,
    onMatched: $<HTMLSelectElement>('sinkM').value as Sink,
    onNotMatched: $<HTMLSelectElement>('sinkN').value as Sink,
    webhookUrl: $<HTMLInputElement>('hookUrl').value.trim(),
    webhookMethod: $<HTMLSelectElement>('hookMethod').value as 'POST' | 'GET',
    cooldownMs: +$<HTMLInputElement>('cool').value,
  };
  $('thrLbl').textContent = `${$<HTMLInputElement>('thr').value}%`;
  localStorage.setItem('aidejavu.studio.cfg', JSON.stringify(config));
}

function restoreConfig() {
  try {
    const c = JSON.parse(localStorage.getItem('aidejavu.studio.cfg') || 'null');
    if (!c) return;
    $<HTMLInputElement>('thr').value = String(Math.round((c.threshold ?? 0.7) * 100));
    $<HTMLSelectElement>('sinkM').value = c.onMatched ?? 'console';
    $<HTMLSelectElement>('sinkN').value = c.onNotMatched ?? 'off';
    $<HTMLInputElement>('hookUrl').value = c.webhookUrl ?? '';
    $<HTMLSelectElement>('hookMethod').value = c.webhookMethod ?? 'POST';
    $<HTMLInputElement>('cool').value = String(c.cooldownMs ?? 0);
    config = { ...config, ...c };
  } catch { /* ignore */ }
}

// ---------------------------------------------------------------- inference
async function runInference(v: Vec, source: string) {
  if (!active) { log('train or load a model first', 'r'); return; }
  const { rec, pipe } = active;
  const res = classify(pipe, rec.classes, v);
  const out = await dispatch({
    predicted: res.predicted, confidence: res.confidence, probs: res.probs,
    classes: rec.classes, modelId: rec.id, modelName: rec.name,
  }, config, log);
  if (hasIDB) {
    await store.addEvent({
      modelId: rec.id, ts: Date.now(), predicted: res.predicted,
      confidence: res.confidence, matched: out.matched, probs: res.probs,
      delivered: out.delivered,
    });
  }
  $('verdict').innerHTML =
    `<span class="${out.matched ? 'g' : 'o'}" style="font-size:18px;font-weight:700">` +
    `${out.matched ? '● MATCH' : '○ no match'}</span> <b>${res.predicted}</b> ` +
    `<span class="dim">${pct(res.confidence)} · ${source} · ${out.delivered}` +
    `${out.skipped ? ' (cooldown)' : ''}</span>` +
    `<div class="bars">${rec.classes.map((c, i) =>
      `<div><span>${c}</span><i style="width:${(res.probs[i] * 100).toFixed(0)}%"></i>` +
      `<em>${(res.probs[i] * 100).toFixed(0)}%</em></div>`).join('')}</div>`;
  $('payload').textContent = JSON.stringify(out.payload, null, 2);
  await renderEvents();
}

async function renderEvents() {
  if (!hasIDB) return;
  const evs = await store.listEvents(25);
  $('events').innerHTML = evs.length
    ? evs.map(e => `<tr><td class="dim">${new Date(e.ts).toLocaleTimeString()}</td>` +
        `<td class="${e.matched ? 'g' : 'dim'}">${e.matched ? 'MATCH' : '—'}</td>` +
        `<td>${e.predicted}</td><td>${pct(e.confidence)}</td>` +
        `<td class="dim">${e.delivered}</td></tr>`).join('')
    : '<tr><td colspan="5" class="dim">no events yet</td></tr>';
}

async function refreshModels() {
  if (!hasIDB) return;
  const ms = await store.listModels();
  $<HTMLSelectElement>('modelSel').innerHTML = ms.map(m =>
    `<option value="${m.id}"${active?.rec.id === m.id ? ' selected' : ''}>` +
    `${m.name} · ${pct(m.metrics.test)}</option>`).join('');
  const u = await store.usage();
  $('usage').textContent = `${u.models} model(s), ${u.events} event(s), ~${(u.bytes / 1024).toFixed(1)} KB`;
}

// ---------------------------------------------------------------- boot
export async function boot() {
  logEl = $('log');
  if (!hasIDB) log('IndexedDB unavailable — nothing will persist', 'r');
  restoreConfig();
  syncConfig();

  $('modality').addEventListener('change', async () => {
    modality = $<HTMLSelectElement>('modality').value as Modality;
    await stopCapture();
    $('imgTools').style.display = modality === 'image' ? '' : 'none';
    $('audTools').style.display = modality === 'audio' ? '' : 'none';
    log(`modality: ${modality} (extractor ${EXTRACTOR[modality].name}, ${EXTRACTOR[modality].dim}-d)`);
    await refreshSamples();
  });

  $('project').addEventListener('change', async () => {
    project = $<HTMLInputElement>('project').value.trim() || 'default';
    await refreshSamples();
  });

  ['thr', 'sinkM', 'sinkN', 'hookUrl', 'hookMethod', 'cool'].forEach(id =>
    $(id).addEventListener('change', syncConfig));
  $('thr').addEventListener('input', syncConfig);

  // ---- image capture
  $('camOn').addEventListener('click', () => startCamera().catch(e =>
    log('camera failed: ' + (e?.message ?? e), 'r')));
  $('camOff').addEventListener('click', () => { stopCapture(); log('capture stopped'); });
  $('shoot').addEventListener('click', async () => {
    const v = $<HTMLVideoElement>('cam');
    if (!v.videoWidth) { log('camera not ready', 'r'); return; }
    const s = videoToSurface(v);
    const blob = await canvasToBlob(s as unknown as HTMLCanvasElement);
    await addSample(blob, 'camera', $<HTMLInputElement>('label').value.trim());
  });
  $('imgFile').addEventListener('change', async e => {
    const files = (e.target as HTMLInputElement).files;
    if (!files) return;
    const label = $<HTMLInputElement>('label').value.trim();
    for (const f of Array.from(files)) await addSample(f, 'upload', label);
    (e.target as HTMLInputElement).value = '';
  });

  // ---- audio capture
  $('micOn').addEventListener('click', () => startMic().catch(e =>
    log('microphone failed: ' + (e?.message ?? e), 'r')));
  $('micOff').addEventListener('click', () => { stopCapture(); log('capture stopped'); });
  $('rec').addEventListener('click', async () => {
    const ms = +$<HTMLInputElement>('clipMs').value;
    const label = $<HTMLInputElement>('label').value.trim();
    if (!label) { log('pick or type a label first', 'r'); return; }
    log(`recording ${ms} ms…`);
    try {
      const blob = await recordClip(ms);
      await addSample(blob, 'mic', label, ms);
    } catch (e: any) { log('record failed: ' + (e?.message ?? e), 'r'); }
  });
  $('audFile').addEventListener('change', async e => {
    const files = (e.target as HTMLInputElement).files;
    if (!files) return;
    const label = $<HTMLInputElement>('label').value.trim();
    for (const f of Array.from(files)) await addSample(f, 'upload', label);
    (e.target as HTMLInputElement).value = '';
  });

  // ---- train / models
  $('trainBtn').addEventListener('click', async () => {
    const b = $<HTMLButtonElement>('trainBtn');
    b.disabled = true;
    try { await train(); } catch (e: any) { log('training failed: ' + (e?.message ?? e), 'r'); }
    b.disabled = false;
  });
  $('loadBtn').addEventListener('click', async () => {
    const id = $<HTMLSelectElement>('modelSel').value;
    const rec = id ? await store.getModel(id) : undefined;
    if (!rec) { log('no model selected', 'r'); return; }
    active = { rec, pipe: fromRecord(rec) };
    config.targets = config.targets.filter(t => rec.classes.includes(t));
    if (!config.targets.length) config.targets = [rec.classes[0]];
    renderTargets(rec.classes);
    log(`loaded "${rec.name}" (${rec.extractor}, ${rec.dim}-d)`, 'g');
    $<HTMLButtonElement>('runBtn').disabled = false;
  });
  $('exportBtn').addEventListener('click', async () => {
    const id = $<HTMLSelectElement>('modelSel').value;
    const rec = id ? await store.getModel(id) : active?.rec;
    if (!rec) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([toJSON(rec)], { type: 'application/json' }));
    a.download = `aidejavu-${rec.id}.json`;
    a.click();
  });
  $('clearSamples').addEventListener('click', async () => {
    const n = await samples.clearProject(project);
    log(`cleared ${n} sample(s) from "${project}"`, 'o');
    await refreshSamples();
  });
  $('clearEv').addEventListener('click', async () => {
    await store.clearEvents(); await renderEvents(); await refreshModels();
  });

  // ---- run inference on a NEW input
  $('runBtn').addEventListener('click', async () => {
    if (!active) return;
    try {
      if (active.rec.extractor === 'photo') {
        const v = $<HTMLVideoElement>('cam');
        if (stream && v.videoWidth) {
          await runInference(embedPhoto(videoToSurface(v)), 'camera');
        } else { log('start the camera, or use "test a file" below', 'o'); }
      } else {
        if (!stream) { log('start the microphone first', 'o'); return; }
        const ms = +$<HTMLInputElement>('clipMs').value;
        log(`listening ${ms} ms…`);
        const blob = await recordClip(ms);
        const { samples: pcm, sampleRate } = await decodeAudioFile(blob);
        await runInference(embedAudio(pcm, sampleRate), 'mic');
      }
    } catch (e: any) { log('inference failed: ' + (e?.message ?? e), 'r'); }
  });

  $('testFile').addEventListener('change', async e => {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (!f || !active) return;
    try {
      if (active.rec.extractor === 'photo') {
        await runInference(embedPhoto(await fileToSurface(f)), 'file');
      } else {
        const { samples: pcm, sampleRate } = await decodeAudioFile(f);
        await runInference(embedAudio(pcm, sampleRate), 'file');
      }
    } catch (err: any) { log('could not read file: ' + (err?.message ?? err), 'r'); }
    (e.target as HTMLInputElement).value = '';
  });

  // ---- continuous mode: classify every N ms and let the actions fire
  let loopTimer: number | null = null;
  $('loop').addEventListener('click', async () => {
    const btn = $<HTMLButtonElement>('loop');
    if (loopTimer) {
      clearInterval(loopTimer); loopTimer = null;
      btn.textContent = 'Start continuous'; btn.classList.remove('on');
      log('continuous mode off');
      return;
    }
    if (!active) { log('train or load a model first', 'r'); return; }
    if (active.rec.extractor !== 'photo') {
      log('continuous mode is image-only for now (audio needs fixed clips)', 'o');
      return;
    }
    if (!stream) { log('start the camera first', 'o'); return; }
    btn.textContent = 'Stop continuous'; btn.classList.add('on');
    log('continuous mode on — actions will fire on every frame that matches', 'g');
    loopTimer = window.setInterval(async () => {
      const v = $<HTMLVideoElement>('cam');
      if (v.videoWidth) await runInference(embedPhoto(videoToSurface(v)), 'live');
    }, Math.max(300, +$<HTMLInputElement>('loopMs').value));
  });

  await refreshSamples();
  await refreshModels();
  await renderEvents();
  log('studio ready — pick a modality, add samples, train', 'g');
}
