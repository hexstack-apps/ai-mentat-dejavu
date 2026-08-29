# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

Train an image classifier **in the browser**, persist it to **IndexedDB**, then run

## Repository location

- Local: `/var/minis/repos/ai-mentat-dejavu` — **all repos live under `/var/minis/repos/`**
- Remote: `hexstack-apps/ai-mentat-dejavu` (private)

## Stack

JavaScript, Python, TypeScript

## Commands

```sh
npm run build        # tsc
npm run test         # node dist/test.js && node dist/test_actions.js
npm run cli          # node dist/cli.js
npm run serve        # node dist/server.js 8772 .
npm run site         # tsc && mkdir -p site/dist && cp dist/*.js site/dist/ && node
```

## Tests

```sh
node desktop/test_shim.mjs
```

## Conventions

- One logical change = one commit, with the measurements behind it.
- Add a `Requested: "..."` trailer citing the originating request.
- Push to the private `hexstack-apps` remote — that is the backup.
- Never `mv` a git repo inside `/var/minis` (it corrupts the object
  store on this Android FS); re-clone from GitHub instead.
- Run tests AND build before deploying; smoke-test the bundle.
