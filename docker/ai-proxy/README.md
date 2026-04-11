# ai-proxy — unified AI services reverse proxy

One HTTP entry point (`http://localhost:8888`) in front of every qontinui
AI service. Path-based routing via Caddy:

| Path           | Backend          | Protocol                          |
|----------------|------------------|-----------------------------------|
| `/parse*`      | OmniParser       | multipart image upload, non-OpenAI |
| `/v1/score*`   | PRM (reserved)   | returns 503 until wired            |
| `/v1/*`, `*`   | llama-swap       | OpenAI-format LLM inference        |
| `/health`      | proxy itself     | liveness probe                     |
| `/health/omniparser` / `/health/llama-swap` | backend healthchecks |           |

## Why this exists

`llama-swap` is not a generic HTTP proxy — every route handler parses
the JSON body and dispatches by `model` field, so it cannot front
OmniParser (`POST /parse` with multipart image) or PRM (`/v1/score`).
Before this stack, a caller wanting to use all three had to hit three
different `localhost:XXXX` ports. This proxy consolidates them.

## Bring it up

```bash
docker compose -f docker/ai-proxy/docker-compose.yml up -d
curl http://localhost:8888/health
```

Set qontinui env vars to point at the proxy:

```bash
export QONTINUI_OMNIPARSER_ENABLED=true
export QONTINUI_OMNIPARSER_PROVIDER=service
export QONTINUI_OMNIPARSER_SERVICE_URL=http://localhost:8888
export QONTINUI_UITARS_VLLM_SERVER_URL=http://localhost:8888
```

## Relation to the standalone composes

The per-service files at `docker/omniparser/` and `docker/llama-swap/`
remain as alternatives for focused dev and debugging — they expose
their backend ports to the host directly and use separate volume
names. Don't run both the unified stack and a standalone service at
the same time or you'll get port conflicts on the non-proxied service.

## Adding PRM

Once `qontinui-prm` has a stable compose interface, add a `prm`
service to this compose file (joining `ai-net`) and update the
`handle /v1/score*` block in the `Caddyfile` to `reverse_proxy
prm:8400`. No other changes needed.

## Gotchas

- First-time boot pulls the OmniParser + llama-swap builds (~15GB
  total) and downloads models. Subsequent starts are near-instant
  because of the named volumes.
- GPU is shared between both backends via llama-swap's time-sharing;
  OmniParser holds its own allocation. On a 5090 (32GB) you have
  headroom for both resident.
- The proxy adds no auth and terminates no TLS. It's a localhost
  convenience — put a real gateway in front if you expose it.
