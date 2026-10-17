# speech — Kokoro TTS + Whisper ASR for the studio's narration

Two HTTP services that qontinui-studio's narration step (`studio-scenes
narrate`, and the weekly `studio pulse run`) calls. Plan
`2026-10-09-qontinui-marketing-studio-animated-process-explainers`, Phase 5.

| Service | Image (pinned by digest) | Host port | Studio variable |
|---------|--------------------------|-----------|-----------------|
| `tts` — Kokoro-82M via [remsky/Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) (Apache-2.0 wrapper) | `ghcr.io/remsky/kokoro-fastapi-cpu:v0.9.0` | `127.0.0.1:8880` | `STUDIO_TTS_URL=http://127.0.0.1:8880` |
| `asr` — Whisper via [speaches](https://github.com/speaches-ai/speaches) (faster-whisper, MIT) | `ghcr.io/speaches-ai/speaches:0.8.3-cpu` | `127.0.0.1:8890` | `STUDIO_ASR_URL=http://127.0.0.1:8890` |

The studio's default `STUDIO_ASR_MODEL` is `whisper-1`; `model_aliases.json`
maps it to `Systran/faster-whisper-small.en` (upstream maps it to large-v3,
which is slow on CPU). The `asr-model` one-shot downloads that model on every
`up` (a no-op once cached). Until it has exited 0, transcription answers 404
("not installed locally").

## Why it looks like this

- **Licence boundary.** Kokoro's Python stack depends on `phonemizer-fork`
  (GPLv3+) and bundles espeak-ng (GPL-3.0). It therefore runs as its own
  container, and only its audio output enters a render. Nothing in qontinui or
  qontinui-studio links it. Running the image is fine. Do not push a derived
  image anywhere, because that would be a GPL distribution.
- **CPU, not GPU.** Both models are small. On the GPU box, the card is
  time-shared exclusively by `docker/llama-swap`, and a resident speech model
  would hold VRAM that llama-swap cannot see. Moving either service to a GPU
  image is a separate decision.
- **Loopback by default.** The host ports are published on `SPEECH_BIND`,
  which defaults to `127.0.0.1`. Binding any other interface (the tailnet or
  `0.0.0.0`) is a port exposure. That makes it a security-surface change under
  the served `security-and-autonomy` policy, so it needs its own reviewed
  change, not a casual override. Run these services on the same host as the
  studio that calls them. Neither service authenticates, and on the default
  bridge network other containers on the host can reach their container
  ports.
- **Thread caps.** torch sizes its pool from the host's core count, not from
  the container's `cpus` limit. On a 48-core box capped at 8 CPUs, a 3-second
  clip took about 130 s. With `OMP_NUM_THREADS=8` the same clip takes about
  1 s. If you change `cpus`, change the thread variables to match.
- **No project network.** Each service uses the default bridge network or
  shares `asr`'s network namespace. As a result, `up` still works on a host
  whose Docker address pools are exhausted by ephemeral stacks.

## Bring it up

Deploy from a stable copy of `main`, not from an agent worktree. The
`model_aliases.json` bind mount is resolved against the compose file's
directory. If that worktree is retired, Docker creates the missing source as
an empty directory, and `asr` then fails to start.

```bash
# from the qontinui checkout root, with origin fetched
DEPLOY=../.deploy/speech        # any stable, host-local directory
mkdir -p "$DEPLOY"
git archive origin/main docker/speech | tar -x -C "$DEPLOY" --strip-components=2
docker compose -p qontinui-speech -f "$DEPLOY/docker-compose.yml" up -d
docker compose -p qontinui-speech -f "$DEPLOY/docker-compose.yml" wait asr-model  # returns once the download has exited; expect status 0
docker compose -p qontinui-speech -f "$DEPLOY/docker-compose.yml" ps              # tts + asr healthy
```

The Kokoro weights ship inside the image. The first start downloads the
whisper model into the named volume `qontinui-speech_speech-hf-cache`, where
it persists.
`-p qontinui-speech` fixes the project name, so a redeploy replaces the
containers instead of starting a second stack that collides on the ports.

## Smoke test

```bash
curl -sf -X POST http://127.0.0.1:8880/dev/captioned_speech \
  -H 'content-type: application/json' \
  -d '{"model":"kokoro","input":"Forty two pull requests this week.","voice":"af_heart","response_format":"wav","stream":false,"return_timestamps":true}' \
  -o tts.json
python3 -c 'import json,base64; d=json.load(open("tts.json")); open("clip.wav","wb").write(base64.b64decode(d["audio"])); print(d["timestamps"])'

curl -sf http://127.0.0.1:8890/v1/audio/transcriptions -F file=@clip.wav \
  -F model=whisper-1 -F response_format=verbose_json \
  -F 'timestamp_granularities[]=word' -F language=en
```

The first call returns `{audio, audio_format, timestamps: [{word, start_time, end_time}]}`.
The second returns `{text, words: [{word, start, end}]}`. These are the shapes
`qontinui-studio/scenes/src/narration/services.ts` reads.

## Stop

```bash
DEPLOY=../.deploy/speech   # the directory you deployed from
docker compose -p qontinui-speech -f "$DEPLOY/docker-compose.yml" down   # volumes kept
```
