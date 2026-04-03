# llama-swap Model Orchestration

Time-shares a single GPU across AriaUI and UI-TARS models using [llama-swap](https://github.com/mostlygeek/llama-swap).

## Why

Running AriaUI (~13GB) + UI-TARS 7B (~12GB) + OmniParser (~6GB) simultaneously requires 31GB+ VRAM. With llama-swap, peak VRAM = the largest single model (~13GB), because only the active model is loaded.

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
- Groups: Uncomment the `groups` section in config.yaml to co-load AriaUI + UI-TARS-2B simultaneously (~17GB)

## Pre-downloaded Models

The Dockerfile pre-downloads:
- `Aria-UI/Aria-UI-base` (~25GB)
- `ByteDance-Seed/UI-TARS-2B-SFT` (~4GB)

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

## GPU Tier Guide

| GPU VRAM | Recommended Config |
|----------|-------------------|
| 8GB | UI-TARS 2B only, remove AriaUI from config |
| 12GB | AriaUI OR UI-TARS 7B (time-shared) |
| 16GB | AriaUI + UI-TARS 2B (co-loaded via groups) |
| 24GB+ | AriaUI + UI-TARS 7B (co-loaded via groups) |
| 32GB | Full stack + OmniParser alongside |
