#!/usr/bin/env node
/**
 * User-flow regression tests.
 *
 * These are static/source checks, because the DOM behaviour is verified live in
 * the browser. What they pin down are the flow bugs actually found by clicking
 * through the Studio:
 *
 *  1. "Capture frame", "Record sample" and continuous mode were ENABLED with no
 *     camera / no mic / no model, so the UI invited an action then reported an
 *     error instead of preventing it.
 *  2. Loading a shapes-demo model (v1/v2) into the Studio "succeeded", enabled
 *     Classify, then failed with a misleading message — and embedFor() would
 *     have silently returned photo features for a 26-d head.
 *  3. Record reported "pick a label first" when the real blocker was no mic.
 *  4. Every id the script touches must exist in the page it belongs to,
 *     otherwise boot() dies on a null dereference.
 */
import { readFileSync } from 'node:fs';

let failed = 0;
const check = (name: string, ok: boolean, detail = '') => {
  console.log(`[${ok ? 'PASS' : 'FAIL'}] ${name}${detail ? '  -- ' + detail : ''}`);
  if (!ok) failed++;
};
const read = (rel: string) => readFileSync(new URL(rel, import.meta.url), 'utf8');
const studioTs = read('../src/studio.ts');
const studioHtml = read('../site/studio.html');
const mvpTs = read('../src/mvp.ts');
const mvpHtml = read('../site/index.html');

// ---------------------------------------------------------------- 4. no null $()
function idsUsed(src: string): string[] {
  const out = new Set<string>();
  for (const m of src.matchAll(/\$(?:<[^>]+>)?\('([A-Za-z0-9_]+)'\)/g)) out.add(m[1]);
  for (const m of src.matchAll(/getElementById\('([A-Za-z0-9_]+)'\)/g)) out.add(m[1]);
  return [...out];
}
function idsDefined(html: string): Set<string> {
  const out = new Set<string>();
  for (const m of html.matchAll(/id="([A-Za-z0-9_]+)"/g)) out.add(m[1]);
  return out;
}
{
  // Ids the script INJECTS at runtime are legitimate even though the static HTML
  // has no such element (e.g. the dedupe button inside the trainability line).
  const injected = idsDefined(studioTs);
  const have = new Set([...idsDefined(studioHtml), ...injected]);
  const missing = idsUsed(studioTs).filter(i => !have.has(i));
  check('studio.ts touches only ids present in studio.html or injected by it',
        missing.length === 0, missing.length ? `missing: ${missing.join(', ')}` : 'all resolve');

  const haveMvp = new Set([...idsDefined(mvpHtml), ...idsDefined(mvpTs)]);
  const missingMvp = idsUsed(mvpTs).filter(i => !haveMvp.has(i));
  check('mvp.ts touches only ids present in index.html',
        missingMvp.length === 0, missingMvp.length ? `missing: ${missingMvp.join(', ')}` : 'all resolve');
}

// ---------------------------------------------------------------- 1. affordance
{
  check('a central syncControls() governs button state',
        studioTs.includes('function syncControls()'));
  for (const id of ['shoot', 'rec', 'runBtn', 'loop']) {
    check(`${id} is disabled until usable`,
          new RegExp(`\\$<HTMLButtonElement>\\('${id}'\\)\\.disabled\\s*=`).test(studioTs),
          'was enabled with no device/model');
  }
  check('syncControls runs on label input',
        studioTs.includes("$('label').addEventListener('input', syncControls)"),
        'typing a label must enable capture without a reload');
  check('syncControls runs after capture start/stop',
        (studioTs.match(/syncControls\(\);/g) ?? []).length >= 5);
  check('a hint explains why capture is blocked',
        studioTs.includes("$('captureHint')") && studioHtml.includes('id="captureHint"'));
  check('hint lives outside the modality panels',
        studioHtml.indexOf('id="captureHint"') > studioHtml.indexOf('id="audTools"'),
        'otherwise it disappears when audio is selected');
}

// ---------------------------------------------------------------- 2. compatibility
{
  check('a compatibility() gate exists', studioTs.includes('function compatibility('));
  check('embedFor refuses unknown extractors',
        /cannot compute .*features/.test(studioTs) && studioTs.includes('throw new Error'),
        'must not fall through to embedPhoto for a v1/v2 head');
  check('load refuses an incompatible model',
        studioTs.includes('cannot load:'),
        'shapes-demo models share the same IndexedDB');
  check('incompatible models are flagged in the dropdown',
        studioTs.includes('not usable here'));
  check('loading aligns modality with the model',
        studioTs.includes('modality !== compat.reason'),
        'so Classify uses the right capture widget');
}

// ---------------------------------------------------------------- 3. error order
{
  const recBlock = studioTs.slice(studioTs.indexOf("$('rec').addEventListener"));
  const micIdx = recBlock.indexOf('microphone on first');
  const labelIdx = recBlock.indexOf('pick or type a label first');
  check('record reports the missing device before the missing label',
        micIdx > -1 && micIdx < labelIdx,
        'the real blocker must be named first');

  const shootBlock = studioTs.slice(studioTs.indexOf("$('shoot').addEventListener"));
  check('capture reports no-camera before no-label',
        shootBlock.indexOf('camera on first') > -1 &&
        shootBlock.indexOf('camera on first') < shootBlock.indexOf('label first'));
  check('capture distinguishes "not started" from "still starting"',
        shootBlock.includes('still starting'),
        'a warming-up camera is not the same error as no camera');
}

// ---------------------------------------------------------------- mobile
{
  // Camera and mic are commonly DENIED on mobile / in embedded WebViews
  // (measured NotAllowedError for both), so upload is the essential path and
  // the failure must be actionable rather than a raw DOMException message.
  check('device errors are translated for humans',
        studioTs.includes('function deviceError(') &&
        studioTs.includes('NotAllowedError') &&
        studioTs.includes('use the upload button'),
        'raw "Permission denied" gives the user nothing to do');
  check('a failed camera start re-syncs controls',
        /camera unavailable[\s\S]{0,120}syncControls\(\)/.test(studioTs),
        'Capture stayed enabled after the camera failed');
  check('a failed mic start re-syncs controls',
        /microphone unavailable[\s\S]{0,120}syncControls\(\)/.test(studioTs));
  check('adding a sample re-syncs controls',
        /await refreshSamples\(\);\s*\n\s*syncControls\(\);/.test(studioTs),
        'the hint went stale/empty after an upload');
  check('image input offers the phone camera',
        /id="imgFile"[^>]*capture="environment"/.test(studioHtml),
        'the upload fallback should reach the camera when getUserMedia cannot');
  check('capture button sits under the video frame',
        studioHtml.indexOf('id="shoot"') > studioHtml.indexOf('id="cam"') &&
        studioHtml.indexOf('id="shoot"') < studioHtml.indexOf('</div>', studioHtml.indexOf('id="cam"')) + 400,
        'aim then tap, instead of a button above the preview');
  check('real controls meet a mobile tap target',
        /\.row button[^{]*\{min-height:44px\}/.test(studioHtml),
        'all controls measured 27-35px tall');
  check('tap-target rule is scoped, not every button',
        !/(^|\n)button\{min-height:44px\}/.test(studioHtml),
        'min-height always beats height, so a blanket rule inflated the ' +
        '22px thumbnail delete badge into a 44px oval over the preview');
}

// ---------------------------------------------------------------- meters
{
  check('probability meters are rendered by a dedicated function',
        studioTs.includes('function renderVerdict('));
  check('meter nodes are REUSED so widths animate',
        studioTs.includes("bars.dataset.classes !== classes.join('|')") &&
        studioTs.includes("fill.style.width"),
        'rebuilding innerHTML restarts the element and kills the transition');
  check('meters have a CSS width transition',
        /\.bfill\{[^}]*transition:width/.test(studioHtml));
  check('winner and match are visually distinct',
        studioHtml.includes('.bfill.win') && studioHtml.includes('.bfill.hit') &&
        studioTs.includes("' hit'") && studioTs.includes("' win'"),
        'colour should show WHY it matched, not just which is largest');
  check('threshold marker drawn on target classes only',
        studioTs.includes("thr.style.left") && studioTs.includes("isTarget ? '' : 'none'"));
  check('percentages shown to one decimal',
        studioTs.includes('(p * 100).toFixed(1)'));
}

console.log(failed ? `\n${failed} FAILURE(S)` : '\nALL PASS');
process.exit(failed ? 1 : 0);
