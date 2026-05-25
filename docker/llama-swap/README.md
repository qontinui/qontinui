# llama-swap Model Orchestration

Time-shares a single GPU across qontinui's vision / grounding / OCR models using [llama-swap](https://github.com/mostlygeek/llama-swap). Only the active model occupies VRAM; idle models unload after their TTL.

## Models Served

Routed by the `model` field in the request body (see `config.yaml`):

| Model | Used by | Notes |
|-------|---------|-------|
| `qontinui-grounding-v1` / `-v5` | healing / grounding | LoRA-merged UI-TARS-1.5-7B, local `/models/*` checkpoints |
| `ByteDance-Seed/UI-TARS-{2B,7B,1.5-7B}` | `uitars` provider | GUI grounding; sizes are mutually exclusive |
| `Zery/CUA_World_State_Model` | runner `WorldStateVerifier` | 7B action-verification judge (2 images/call) |
| `paddleocr` | runner `OcrClient` (Vision Pipeline Phase 4) | classical PP-OCR behind an OpenAI-compatible shim |
| `Aria-UI/Aria-UI-base` / `-context-aware` | healing (opt-in) | 25B MoE — **does not fit a 32GB GPU**, see VRAM note |

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
| 32GB | full 7B stack time-shared (grounding + UI-TARS + WSM + paddleocr); **Aria-UI still does not fit** |
| >40GB | required to load `Aria-UI/Aria-UI-base` (25B MoE) |
