#!/usr/bin/env node
/**
 * Tests for the audio and photo extractors.
 *
 * These run headless: audio is synthesised PCM (tones, noise, chirps) and photos
 * are drawn on the canvas shim. The point is to prove each extractor actually
 * SEPARATES the thing it claims to, using the same separation-ratio metric that
 * diagnosed the v1 image failure — not just that it returns numbers.
 */
import { embedAudio, AUDIO_DIM, resample, rmsDb, N_MELS } from './audio.js';
import { embedPhoto, PHOTO_DIM } from './photo.js';
import { embedV2 } from './features.js';
import { createCanvas } from './canvas.js';
import { Pipeline, mulberry32, norm2, mean, type Mat } from './linalg.js';
import type { Surface } from './dataset.js';

let failed = 0;
const check = (name: string, ok: boolean, detail = '') => {
  console.log(`[${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? '  -- ' + detail : ''}`);
  if (!ok) failed++;
};

const SR = 16000;

// ---------------------------------------------------------------- audio gen
function tone(freq: number, sec = 1, amp = 0.3, seed = 1): Float32Array {
  const rng = mulberry32(seed);
  const n = Math.floor(SR * sec);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = amp * Math.sin((2 * Math.PI * freq * i) / SR) + (rng() - 0.5) * 0.01;
  }
  return x;
}
function noise(sec = 1, amp = 0.3, seed = 2): Float32Array {
  const rng = mulberry32(seed);
  const n = Math.floor(SR * sec);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = (rng() * 2 - 1) * amp;
  return x;
}
function chirp(f0: number, f1: number, sec = 1, amp = 0.3): Float32Array {
  const n = Math.floor(SR * sec);
  const x = new Float32Array(n);
  let ph = 0;
  for (let i = 0; i < n; i++) {
    const f = f0 + ((f1 - f0) * i) / n;
    ph += (2 * Math.PI * f) / SR;
    x[i] = amp * Math.sin(ph);
  }
  return x;
}

const sep = (X: Mat, y: number[]) => {
  const w: number[] = [], b: number[] = [];
  for (let i = 0; i < X.length; i++)
    for (let j = i + 1; j < X.length; j++)
      (y[i] === y[j] ? w : b).push(norm2(X[i], X[j]));
  return mean(b) / mean(w);
};

// ---------------------------------------------------------------- audio tests
{
  const a = embedAudio(tone(440), SR);
  const b = embedAudio(tone(440), SR);
  check('audio embedding deterministic', a.every((v, i) => v === b[i]));
  check('audio fixed width', a.length === AUDIO_DIM, `dim=${AUDIO_DIM}`);
  check('audio features finite', Array.from(a).every(Number.isFinite));

  // a clip shorter than one FFT frame must not crash or emit NaN
  const tiny = embedAudio(new Float32Array(100), SR);
  check('short clip degrades safely',
        tiny.length === AUDIO_DIM && Array.from(tiny).every(Number.isFinite));

  // DIFFERENT PITCHES must separate
  const X: Mat = [], y: number[] = [];
  for (let i = 0; i < 6; i++) { X.push(embedAudio(tone(300, 1, 0.3, 10 + i), SR)); y.push(0); }
  for (let i = 0; i < 6; i++) { X.push(embedAudio(tone(1200, 1, 0.3, 20 + i), SR)); y.push(1); }
  const rPitch = sep(X, y);
  check('separates 300 Hz from 1200 Hz', rPitch > 1.5, `ratio ${rPitch.toFixed(2)}`);

  // TONE vs NOISE must separate
  const X2: Mat = [], y2: number[] = [];
  for (let i = 0; i < 6; i++) { X2.push(embedAudio(tone(600, 1, 0.3, 30 + i), SR)); y2.push(0); }
  for (let i = 0; i < 6; i++) { X2.push(embedAudio(noise(1, 0.3, 40 + i), SR)); y2.push(1); }
  const rTN = sep(X2, y2);
  check('separates tone from noise', rTN > 1.5, `ratio ${rTN.toFixed(2)}`);

  // LOUDNESS INVARIANCE: the same tone at 0.05 vs 0.5 amplitude must stay closer
  // to itself than a different pitch is. This is what log + band-mean removal buys.
  const quiet = embedAudio(tone(600, 1, 0.05, 7), SR);
  const loud = embedAudio(tone(600, 1, 0.5, 7), SR);
  const other = embedAudio(tone(1500, 1, 0.3, 7), SR);
  const dLoud = norm2(quiet, loud), dOther = norm2(quiet, other);
  check('loudness-invariant (10x amplitude)', dLoud < dOther,
        `same-pitch/10x-gain ${dLoud.toFixed(2)} < diff-pitch ${dOther.toFixed(2)}`);

  // RISING vs FALLING chirp: only the time-segment features can tell these apart,
  // since their overall spectra are identical. Guards that half of the vector.
  const X3: Mat = [], y3: number[] = [];
  for (let i = 0; i < 5; i++) { X3.push(embedAudio(chirp(300, 2000, 1, 0.3), SR)); y3.push(0); }
  for (let i = 0; i < 5; i++) { X3.push(embedAudio(chirp(2000, 300, 1, 0.3), SR)); y3.push(1); }
  const up = embedAudio(chirp(300, 2000), SR), down = embedAudio(chirp(2000, 300), SR);
  check('time profile distinguishes rising from falling chirp',
        norm2(up, down) > 1.0, `distance ${norm2(up, down).toFixed(2)}`);

  // end-to-end: can the head learn it?
  const pipe = new Pipeline({ epochs: 300 }).fit(X, y, 2);
  check('head learns pitch classes', pipe.score(X, y) === 1,
        `${(pipe.score(X, y) * 100).toFixed(0)}%`);

  // resample keeps duration
  const rs = resample(tone(440, 1), SR, 8000);
  check('resample halves sample count', Math.abs(rs.samples.length - SR / 2) < 10,
        `${rs.samples.length}`);
  check('rmsDb sane', rmsDb(tone(440, 0.1, 0.5)) > -20 && rmsDb(new Float32Array(100)) < -100,
        `tone ${rmsDb(tone(440, 0.1, 0.5)).toFixed(1)} dB, silence ${rmsDb(new Float32Array(100)).toFixed(0)} dB`);
}

// ---------------------------------------------------------------- photo tests
const mk = (w: number, h: number) => createCanvas(w, h) as unknown as Surface;

/** A crude 'photo': coloured region + texture, not a single blob on black. */
function fakePhoto(kind: 'red-left' | 'blue-right' | 'stripes', seed: number): Surface {
  const S = 128;
  const cv = mk(S, S);
  const cx = cv.getContext('2d');
  const rng = mulberry32(seed);
  const id = cx.createImageData(S, S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const i = (y * S + x) * 4;
      let r = 110, g = 110, b = 110;
      if (kind === 'red-left') {
        if (x < S / 2) { r = 200; g = 60; b = 50; }
      } else if (kind === 'blue-right') {
        if (x >= S / 2) { r = 50; g = 70; b = 210; }
      } else {
        const on = Math.floor(y / 6) % 2 === 0;
        r = g = b = on ? 210 : 40;
      }
      const n = (rng() - 0.5) * 26;
      id.data[i] = Math.max(0, Math.min(255, r + n));
      id.data[i + 1] = Math.max(0, Math.min(255, g + n));
      id.data[i + 2] = Math.max(0, Math.min(255, b + n));
      id.data[i + 3] = 255;
    }
  }
  cx.putImageData(id, 0, 0);
  return cv;
}

{
  const p1 = embedPhoto(fakePhoto('red-left', 1));
  const p2 = embedPhoto(fakePhoto('red-left', 1));
  check('photo embedding deterministic', p1.every((v, i) => v === p2[i]));
  check('photo fixed width', p1.length === PHOTO_DIM, `dim=${PHOTO_DIM}`);
  check('photo features finite', Array.from(p1).every(Number.isFinite));

  const kinds = ['red-left', 'blue-right', 'stripes'] as const;
  const X: Mat = [], y: number[] = [], Xv2: Mat = [];
  kinds.forEach((k, li) => {
    for (let i = 0; i < 8; i++) {
      const s = fakePhoto(k, 100 + li * 50 + i);
      X.push(embedPhoto(s)); Xv2.push(embedV2(s)); y.push(li);
    }
  });
  const rPhoto = sep(X, y), rV2 = sep(Xv2, y);
  check('photo extractor separates the three scenes', rPhoto > 1.5,
        `ratio ${rPhoto.toFixed(2)}`);
  // The justification for a separate extractor: v2 is built for one bright blob
  // on a dark field and should do measurably worse on photo-like input.
  check('photo extractor beats v2 on photo-like input', rPhoto > rV2,
        `photo ${rPhoto.toFixed(2)} vs v2 ${rV2.toFixed(2)}`);

  const pipe = new Pipeline({ epochs: 300 }).fit(X, y, 3);
  check('head learns photo classes', pipe.score(X, y) === 1,
        `${(pipe.score(X, y) * 100).toFixed(0)}%`);

  // colour position matters: red-left vs a mirrored version must differ
  const rl = embedPhoto(fakePhoto('red-left', 9));
  const br = embedPhoto(fakePhoto('blue-right', 9));
  check('spatial grid captures position', norm2(rl, br) > 0.5,
        `distance ${norm2(rl, br).toFixed(2)}`);
}


// ---------------------------------------------------------------- schema guard
{
  // store.ts and samples.ts open the SAME IndexedDB. If their VERSION constants
  // diverge, whichever opens with the lower number throws
  // "the requested version (1) is less than the existing version (2)" and the
  // page fails to boot. This actually happened once, so pin it.
  const fs = await import('node:fs');
  const grab = (f: string) => {
    const src = fs.readFileSync(new URL(f, import.meta.url), 'utf8');
    const m = src.match(/const VERSION = (\d+)/);
    return m ? +m[1] : -1;
  };
  const a = grab('../src/store.ts'), b = grab('../src/samples.ts');
  check('store.ts and samples.ts share one DB version', a === b && a > 0,
        `store=${a} samples=${b}`);

  // and both upgrade handlers must create every store, since either can run first
  const srcA = fs.readFileSync(new URL('../src/store.ts', import.meta.url), 'utf8');
  const srcB = fs.readFileSync(new URL('../src/samples.ts', import.meta.url), 'utf8');
  const stores = ['models', 'events', 'samples'];
  check('store.ts upgrade creates all stores',
        stores.every(n => srcA.includes(`'${n}'`)), stores.join(','));
  check('samples.ts upgrade creates all stores',
        stores.every(n => srcB.includes(`'${n}'`)), stores.join(','));
}

console.log(failed ? `\n${failed} FAILURE(S)` : '\nALL PASS');
process.exit(failed ? 1 : 0);
