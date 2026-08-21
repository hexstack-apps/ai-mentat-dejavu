# Deploy

```sh
npm run build
mkdir -p site/dist && cp dist/*.js site/dist/
# drop server-only modules from the browser bundle
rm -f site/dist/{server,cli,test,test_actions,ui}.js*

python3 deploy/pages_deploy.py ai-dejavu <site-root>
```

Always verify over the wire — never trust the upload response:

```sh
curl -s -o /dev/null -w '%{http_code}\n' https://ai-dejavu.hexstack.app/ai-dejavu/
```

## Live

- https://ai-dejavu.hexstack.app/ai-dejavu/  (custom domain)
- https://ai-dejavu.pages.dev/ai-dejavu/     (Pages subdomain)

## Notes

Cloudflare Pages project `ai-dejavu`, account `6dda101664eca020bf086a4de83118a6`,
zone `hexstack.app` = `2aa5defa47b368ac81d43590dada5004`.

Two tokens, NOT interchangeable: `CLOUDFLARE_API_TOKEN` for Pages,
`CLOUDFLARE_ZONES_TOKEN` for DNS.

The apex `hexstack.app` was NOT repointed. It still carries A/AAAA records for an
unrelated (currently dead) origin — `hexstack.app`, `n8n.` and `mc.` all return
403 from Cloudflare with no working backend. Serving the MVP at the apex path
would mean deleting those records, which is a destructive change to a shared
domain, so a dedicated subdomain was used instead. Attaching the apex to the
Pages project stays `pending` until its A/AAAA records are replaced by a CNAME.
