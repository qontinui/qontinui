# gemma-server — local Gemma 4 for the scripted-output emitter

Standalone llama.cpp server serving Gemma 4 26B-A4B for the runner's
`gemma_local_warm` emitter lane. Zero Anthropic traffic, zero dollars, zero
quota — but you pay in VRAM (~26 GB while loaded) and a one-time 23 GB model
download.

## When to run this

Run this if **any** of the following applies:

- Your Claude Max/Pro quota feels tight and you'd rather burn GPU cycles than
  subscription quota on emitter calls.
- You need the emitter to work offline or behind a network partition.
- Your dev machine has a capable GPU (RTX 4090/5090, A6000, H100) and the
  VRAM to spare.

If you're on a Claude API key and don't mind the billing, the existing
`claude_api_warm` lane is simpler and — with the prompt-fattening commit —
does not require this service.

## Prereqs

- Docker + NVIDIA Container Toolkit (`nvidia-smi` must work in a container).
- ~23 GB free on the volume mounted at `./models`.
- A GPU with ≥ 28 GB VRAM (26 GB model + KV cache headroom).

## Setup

```bash
# From qontinui/docker/gemma-server/
mkdir -p models

curl -L -o models/gemma-4-26B-A4B-it-UD-Q6_K.gguf \
  https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q6_K.gguf

docker compose up -d

# Watch it load (first boot takes ~90 s while the GGUF reads into VRAM):
docker logs -f qontinui-gemma-server
```

Server is ready when `curl http://localhost:8200/health` returns
`{"status":"ok"}`.

## Runner integration

The provider is already wired. By default the runner's `ScriptedOutputSettings`
has:

- `provider: auto` — the emitter tries Claude warm first, falls through to
  Gemma local only if no Claude credential is configured.
- `gemma_local_endpoint: "http://localhost:8200"`
- `gemma_local_model_alias: "gemma-4-26b-a4b"`

To **force** Gemma regardless of Claude credentials, set
`scripted_output.provider = "gemma_local_warm"` in `settings.json`. Useful if
you want to deliberately avoid any Anthropic-side traffic on a dev machine.

No runner restart is needed when the Gemma server comes up or goes down —
the provider probes the endpoint on each emit call. A down server surfaces as
an `EmitError::LlmError` under the forced mode or falls through to `Disabled`
under `auto` when it's the last lane.

## Different model or quant?

Swap the `-m` argument in `docker-compose.yml`. Supported shapes llama.cpp
reads out of the box: GGUF files of any Gemma 4 variant, any other local
model that produces JSON-ish output. If you change the base model (not just
the quant), update `scripted_output.gemma_local_model_alias` for telemetry
clarity.

## Troubleshooting

- **Container exits immediately on startup** — check that
  `nvidia-smi` works inside a container: `docker run --rm --gpus all
  nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi`. If that fails, install
  the NVIDIA Container Toolkit.
- **Empty responses on `/v1/chat/completions`** — the emitter deliberately
  uses `/completion`, not `/v1/chat/completions`. llama.cpp's built-in Gemma
  4 chat template strips the model's `<channel|>` reasoning wrapper too
  aggressively. Don't hit the chat endpoint from external tools expecting
  chat output.
- **Slow first call** — the first emit after server start pays cold-prompt
  evaluation (~1 s on the 5090 for our 3361-token system prefix). Subsequent
  calls reuse the KV cache and drop to ~600 ms prompt eval.

## Teardown

```bash
docker compose down
# To also free the 23 GB GGUF:
rm -rf models/
```
