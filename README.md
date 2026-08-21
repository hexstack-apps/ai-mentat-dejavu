# AI Déjà Vu

Train an image classifier **in the browser**, persist it to **IndexedDB**, then run
live inference where every prediction fires a switchable `onMatched` /
`onNotMatched` action — `console.log`, webhook, both, or off.

No server, no upload, no API key. Training and inference both run locally.

**Live:** https://ai-dejavu.pages.dev/ai-dejavu/

## Architecture

A **frozen feature extractor** turns each image into a vector; a small **softmax
head** is the only thing trained. That's the Teachable Machine architecture, and
it's why a trained model is ~1 KB and fits in a database row.

```
image → extractor (frozen, 0 params) → vector → softmax head (81 params) → class
                                                          ↓
                                            onMatched / onNotMatched
                                            console · webhook · both · off
```

## Quick start

```sh
npm install
npm run build          # tsc -> dist/
npm test               # 24 core + 33 action tests
npm run site           # serve the MVP at :8773
npm run cli            # headless v1 vs v2 comparison
```

## What it measures

| representation | dim | infer | separation | test acc |
|---|---|---|---|---|
| v1 absolute orientation | 224 | ~3 ms | 0.99 | **38.9%** |
| **v2 rotation-invariant** | **26** | ~5 ms | 1.99 | **94.4%** |
| DINOv2-small CLS (pretrained) | 384 | ~2.1 s | 2.26 | **100.0%** |

**Representation beats classifier.** v2 has 8.6× *fewer* dimensions than v1 and
scores ~56 points higher. The classifier is byte-identical.

### v1 is a deliberate broken control

```
        rotation gap   between-class   ratio
v1         0.713          0.414        0.58   rotation dominates
v2         0.196          0.395        2.01   OK
```

Rotating the *same square* moves its v1 embedding **further** than changing the
shape does. Within-class variation exceeds between-class separation, so no amount
of training can succeed. v2 measures gradient orientation *relative to the shape's
centroid*, adds radial ring occupancy and pose-free scalars (compactness, extent).

## Match rule

A prediction is a **match** when the predicted class is in `targets` **and**
confidence ≥ `threshold`. The two branches are configured independently, because
you usually want a webhook on match and console on no-match.

Webhook payload:

```json
{
  "event": "onMatched",
  "matched": true,
  "predicted": "triangle",
  "confidence": 0.9993,
  "threshold": 0.5,
  "targets": ["circle", "square", "triangle"],
  "probabilities": { "circle": 0, "square": 0.0007, "triangle": 0.9993 },
  "model": { "id": "mmt2mj9wbo1r3", "name": "v2 · …" },
  "ts": "2026-08-21T07:25:18.140Z"
}
```

`POST` sends JSON; `GET` puts the payload in a `?payload=` query param.
Cross-origin endpoints need CORS. **Webhook failures are logged, never thrown** —
a dead endpoint must not break the inference loop. An optional per-branch cooldown
suppresses rapid re-fires.

## Storage

IndexedDB (`tmjs` v1), two stores:

- `models` — scaler + head weights + metrics recorded at training time, so a
  loaded model is self-describing
- `events` — the match/no-match audit log with what was delivered

Models survive reloads and export to portable JSON. Action config persists in
`localStorage`.

## Integrity checks

Accuracy alone is not evidence, so every training run asserts:

- train/test keys disjoint; no duplicate vectors across splits
- test images are not near-copies of training images
- balanced test set
- 5-fold stratified CV on train agrees
- **shuffled labels collapse to chance** — the decisive one. With more features
  than samples, a head that memorised noise would still score high on permuted
  labels. Measured: 31.1% vs 33.3% chance.

The property suite also guards the dataset: **mean brightness must not identify
the class.** An earlier generator sized shapes by equal *radius*, but a circle
covers πr² and a triangle only ~1.3r², so brightness leaked the label. Shapes are
now sized for equal filled **area** — the test caught that bug.

## Layout

```
src/
  linalg.ts     StandardScaler, softmax regression (SGD), Pipeline,
                stratified k-fold, cross-val — replaces numpy + sklearn
  dataset.ts    seeded synthetic shapes; deterministic split/class/index keys
  features.ts   v1 (224-d) and v2 (26-d) frozen extractors
  verify.ts     integrity checks + rotation diagnostic
  store.ts      IndexedDB models + event log
  actions.ts    match rule and onMatched/onNotMatched dispatch
  serialize.ts  Pipeline <-> stored record <-> JSON
  canvas.ts     pure-JS raster shim so the dataset works headless in Node
  mvp.ts        the app
  cli.ts        headless pipeline
  server.ts     static files + resumable embedding cache
site/index.html the deployable static page
```

Pure TypeScript, `tsc --strict` clean. The same modules drive the CLI, the server
and the browser — no duplicated logic, so the CLI and the page produce identical
numbers.

## Optional: pretrained backbone

DINOv2-small via `transformers.js` reaches 100%, but costs ~2.1 s/image under WASM
and only runs in the browser here (`onnxruntime-node` is glibc-linked and fails on
musl with `__getauxval`). `web/cache_run.html` embeds resumably — each vector is
POSTed to `/cache/put` immediately, so an interruption costs at most one image —
then `node dist/cli.js --cache emb_cache.jsonl` trains the head in ~350 ms.

Note: most ImageNet *classifier* exports only expose `logits`, which are a poor
representation for out-of-distribution shapes (MobileNetV4 scored 51.7%). Only
models with **no classifier head** export usable features. Check
`session.outputNames` before assuming.
