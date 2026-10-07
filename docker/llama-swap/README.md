# llama-swap Model Orchestration

Time-shares a single GPU across qontinui's vision / grounding / OCR models (and, since the llama.cpp stage, GGUF LLM candidates) using [llama-swap](https://github.com/mostlygeek/llama-swap). Only the active model occupies VRAM; idle models unload after their TTL.

## Models Served

Routed by the `model` field in the request body (see `config.yaml`):

| Model | Used by | Notes |
|-------|---------|-------|
| `qontinui-grounding-v1` / `-v5` | healing / grounding | LoRA-merged UI-TARS-1.5-7B, local `/models/*` checkpoints |
| `ByteDance-Seed/UI-TARS-{2B,7B,1.5-7B}` | `uitars` provider | GUI grounding; sizes are mutually exclusive |
| `Zery/CUA_World_State_Model` | runner `WorldStateVerifier` | 7B action-verification judge (2 images/call) |
| `paddleocr` | runner `OcrClient` (Vision Pipeline Phase 4) | classical PP-OCR behind an OpenAI-compatible shim |
| `Aria-UI/Aria-UI-base` / `-context-aware` | healing (opt-in) | 25B MoE — **does not fit a 32GB GPU**, see VRAM note |
| `gemma-4-26b-a4b` / `qwen3-coder-30b-a3b` / `gpt-oss-20b` | local-LLM evaluation (see below) | GGUF via llama.cpp `llama-server`; weights bind-mounted under `/models/gguf/` |

## Why

With llama-swap, peak VRAM = the largest single *loaded* model, because only the active model occupies the GPU — instead of running separate aria-ui / ui-tars / grounding containers simultaneously.

## Quick Start

```bash
cd qontinui/docker
docker compose -f llama-swap/docker-compose.yml up --build
```

AriaUI clients (`aria_ui_client.py`) continue to POST to `http://localhost:8100/v1/chat/completions` with no changes needed (default endpoint is already 8100).

For UI-TARS VLLMProvider, set the server URL to point at llama-swap instead of a standalone vLLM instance:
```bash
export QONTINUI_UITARS_VLLM_SERVER_URL=http://localhost:8100
```

llama-swap routes by the `model` field in the request body and auto-loads the correct backend.

## Fixed-Host Redeploy (prebuilt image, absolute paths)

On a dedicated host (e.g. the canonical GPU box serving `:8100`), redeploy the
**already-built** image without rebuilding, and pin config + weights to
**absolute** paths so the deploy can't silently grab the wrong files:

```bash
cd qontinui
# Pin the config to a current source — the footgun below bites stale branches:
git fetch origin && git checkout main && git pull --ff-only

LLAMA_SWAP_CONFIG=$PWD/docker/llama-swap/config.yaml \
LLAMA_SWAP_MODELS=$PWD/../models \
docker compose -p llama-swap -f docker/llama-swap/docker-compose.yml up -d --no-build
```

Why each flag matters:

- **`-p llama-swap`** — a fixed project name, so a redeploy *replaces* the
  running container and reuses the same `llama-swap_llama-swap-hf-cache` volume
  (cached HF weights aren't re-pulled), instead of spawning a parallel stack
  that collides on port 8100.
- **`--no-build`** — reuse the image already built from the Dockerfile rather
  than rebuilding on every redeploy. Build it once with `... up -d --build`.
- **`LLAMA_SWAP_CONFIG` / `LLAMA_SWAP_MODELS`** — absolute paths. The compose
  defaults (`./config.yaml`, `../../../models`) resolve against the **compose
  file's directory**, so deploying from a *worktree* mounts an empty `/models`
  (grounding models silently fail to load) and deploying from a *stale branch*
  mounts a `config.yaml` without the `paddleocr` entry (OCR silently absent).
  Absolute paths + a fresh `git pull` to `main` remove that coupling.

## How It Works

1. Client sends request with `"model": "Aria-UI/Aria-UI-base"` to port 8100
2. llama-swap starts `serve.py` (AriaUI backend) on a dynamic port
3. Request is proxied to the backend, response returned to client
4. After 300s idle (TTL), AriaUI is unloaded to free VRAM
5. Next request for `"model": "ByteDance-Seed/UI-TARS-2B-SFT"` triggers AriaUI unload, UI-TARS load

## Configuration

Edit `config.yaml` (hot-reloaded via volume mount, no rebuild needed).

Key settings:
- `ttl`: Idle timeout in seconds before unloading (default: 300)
- `healthCheckTimeout`: Max wait for model to load (default: 120s)
- Groups: Uncomment the `groups` section in config.yaml to co-load two small models simultaneously (e.g. `UI-TARS-2B` + a 7B grounding model). Note the committed example names Aria-UI, which won't co-load on a 32GB GPU — see the VRAM Reality note.

## Pre-downloaded Models

The Dockerfile pre-downloads the HF-hosted models into the image's HF cache
(reused at runtime via the `llama-swap-hf-cache` volume). Local fine-tuned
checkpoints (`qontinui-grounding-v1/v5`) are bind-mounted from `/models`, not
baked in. `Aria-UI/Aria-UI-base` (~25GB on disk) is pre-downloadable but will
not load on a 32GB GPU — see VRAM Reality.

For additional models, uncomment lines in the Dockerfile and rebuild.

## Local LLM candidates (llama.cpp)

The image builds llama.cpp's `llama-server` from a pinned release tag
(`LLAMA_CPP_TAG` / `LLAMA_CPP_COMMIT` build args in the Dockerfile; the running
version is in `/etc/llama-cpp-version`). Three GGUF entries in `config.yaml` use
it, for the evaluation in qontinui-dev-notes plan
`2026-10-07-measure-which-ai-work-a-local-model-on-spaceships-5090-can-take-over`.
They swap exclusively with the vision models like every other entry: at most one
model is resident, so a vision request unloads a resident LLM and vice versa.

Bring them up in this order on the GPU box. Run every command from the
`qontinui` checkout root, with the Fixed-Host variables above set
(`LLAMA_SWAP_CONFIG`, `LLAMA_SWAP_MODELS` — absolute paths).

**1. Stop `qontinui-gemma-server`.** It is always-on and holds ~26 GB of VRAM;
with it up, a vLLM vision entry (which reserves ~90% of the card by default) or
an LLM entry will OOM. `docker stop qontinui-gemma-server`, and note that it was
running so you can start it again afterwards.

**2. Put the weights under `$LLAMA_SWAP_MODELS/gguf/`.** Reuse gemma-server's
copy of the Gemma GGUF with a hardlink (same filesystem) — a symlink does not
resolve inside the container, because it points outside the bind mount. The
loop downloads only what is still missing.

| Entry | File (`<models>/gguf/…`) | Source | Licence on the model card (2026-10-08) |
|-------|--------------------------|--------|------------------|
| `gemma-4-26b-a4b` | `gemma-4-26B-A4B-it-UD-Q6_K.gguf` (~23 GB) | `unsloth/gemma-4-26B-A4B-it-GGUF` | apache-2.0 |
| `qwen3-coder-30b-a3b` | `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf` (~18 GB) | `unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF` | apache-2.0 |
| `gpt-oss-20b` | `gpt-oss-20b-MXFP4.gguf` (~12 GB) | `ggml-org/gpt-oss-20b-GGUF` | apache-2.0 |

```bash
M="$LLAMA_SWAP_MODELS/gguf"; mkdir -p "$M"
G=docker/gemma-server/models/gemma-4-26B-A4B-it-UD-Q6_K.gguf
[ -s "$G" ] && [ ! -e "$M/${G##*/}" ] && ln "$G" "$M/"
for f in unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
         unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
         ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-MXFP4.gguf; do
  repo="${f%/*}"; file="${f##*/}"
  [ -s "$M/$file" ] && continue
  curl -fL -o "$M/$file.part" "https://huggingface.co/$repo/resolve/main/$file" && mv "$M/$file.part" "$M/$file"
done
```

**3. Bind loopback only (host-local override).** The tracked compose file
publishes `8100:8100` on every host interface. `ports` lists are *merged*
across compose files, so the override replaces the list with the `!override`
merge tag. Confirm your Compose honours it with the `config` command below —
it must print a single published port with `host_ip: 127.0.0.1`.

```yaml
# docker/llama-swap/docker-compose.override.yml  (host-local; gitignored)
services:
  llama-swap:
    ports: !override
      - "127.0.0.1:8100:8100"
```

Compose loads an override file automatically only when no `-f` is given, and
every command here passes `-f` — so name both files on EVERY command, this
step's and step 4's alike (a redeploy without the second `-f` recreates the
container on `0.0.0.0:8100`):

```bash
F="-f docker/llama-swap/docker-compose.yml -f docker/llama-swap/docker-compose.override.yml"
docker compose -p llama-swap $F config | grep -B3 -A1 'published:'   # one entry, host_ip: 127.0.0.1
```

**4. Rebuild once, then redeploy — a `--no-build` redeploy alone will NOT pick
these entries up.** `config.yaml` is bind-mounted, so without a rebuild the three
entries load into an image that has no `/usr/local/bin/llama-server` and every
request to them fails at exec time (the vision entries are unaffected). Pin the
llama-swap version to the one already serving: stage 1 resolves `latest` when it
is not cached, and its binary is copied in BEFORE the weight layers, so a newer
`latest` would also invalidate the ~29 GB weight layers.

```bash
V="$(docker exec "$(docker compose -p llama-swap ps -q llama-swap)" cat /etc/llama-swap-version)"
docker compose -p llama-swap $F build --build-arg LLAMA_SWAP_VERSION="$V"
docker compose -p llama-swap $F up -d --no-build
docker port "$(docker compose -p llama-swap ps -q llama-swap)" 8100   # must print 127.0.0.1:8100 only
```

The llama-server layers sit after the weight downloads, so with stage 1 pinned
this re-uses the ~29 GB weight layers — **only if this box's build cache still
holds them**. If the cache was pruned, or the image was pulled rather than built
here, the build re-downloads them whatever the layer order: check that the build
log shows `CACHED` on the `snapshot_download` steps (or `docker buildx du`).

**5. Use them.** Each entry answers OpenAI `/v1/chat/completions` on `:8100`
with `"model": "<entry>"`. llama.cpp's native routes are reachable through
llama-swap's upstream passthrough, e.g. the raw completion endpoint the runner's
`gemma_local_warm` emitter uses:

```bash
curl -s http://127.0.0.1:8100/upstream/gemma-4-26b-a4b/completion \
  -d '{"prompt":"<user|>hi<turn|>\n<model|>\n","n_predict":16,"stop":["<turn|>","<user|>"]}'
```

Pointing `scripted_output.gemma_local_endpoint` at
`http://127.0.0.1:8100/upstream/gemma-4-26b-a4b` is what would let
`docker/gemma-server` be retired — but only once that passthrough is verified on
the box and the emitter's 5 s timeout survives a cold swap (the plan's Decision 2
conditions). Until then `docker/gemma-server` stays; start it again when the
evaluation is done.

## Services NOT Managed by llama-swap

These use non-OpenAI APIs and must run as separate containers:
- **OmniParser** (port 8080): `docker compose -f omniparser/docker-compose.yml up`
- **PRM** (port 8400): `cd ../../qontinui-prm && docker compose -f docker/docker-compose.yml up`

## Standalone Mode (Without llama-swap)

To revert to individual containers (original behavior):
```bash
docker compose -f aria-ui/docker-compose.yml --profile standalone up
```

## VRAM Reality (read before enabling Aria-UI)

The 7B-class models — `qontinui-grounding-v1/v5`, `UI-TARS-{2B,7B,1.5-7B}`, and
`CUA_World_State_Model` — load comfortably and time-share fine on a 24–32GB GPU.

`Aria-UI/Aria-UI-base` is a **25B MoE**. With its vision-attention activations it
**OOMs even on a 32GB GPU** in this configuration, so it does not load on the
canonical 32GB host. It is an **opt-in** healing backend only (used solely when
`QONTINUI_ARIA_UI_ENABLED=true` and `llm_mode=ARIA_UI`), not on the default path;
healing's default grounding uses the 7B `qontinui-grounding-*` models instead.
Leave Aria-UI out of `config.yaml` (or expect a load failure) unless you are on a
GPU with materially more than 32GB.

## GPU Tier Guide

| GPU VRAM | Recommended Config |
|----------|-------------------|
| 8GB | `UI-TARS-2B` only |
| 12GB | one 7B model at a time (`grounding`, `UI-TARS-7B`, or `WSM`), time-shared |
| 16–24GB | any single 7B model with headroom; co-load two small ones via `groups` |
| 32GB | full 7B stack time-shared (grounding + UI-TARS + WSM + paddleocr), plus one GGUF LLM at a time (≤ 30B-class at 4–6-bit); **Aria-UI still does not fit** |
| >40GB | required to load `Aria-UI/Aria-UI-base` (25B MoE) |
