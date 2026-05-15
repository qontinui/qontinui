# PaddleOCR — llama-swap OpenAI-API shim

Wraps PaddleOCR with the `POST /v1/chat/completions` API that the
qontinui-runner's `OcrClient` (Phase 4 of the UI Bridge Vision
Pipeline) routes through. The runner's OCR_SYSTEM_PROMPT asks the
model to emit a JSON array of `{bbox, text, confidence}`; this shim's
`serve.py` runs PaddleOCR and returns exactly that shape as
`choices[0].message.content`.

## Usage

The shim is built into the `llama-swap` container at
`/opt/backends/paddleocr/serve.py` (see `../llama-swap/Dockerfile`)
and selected via the `paddleocr` model alias:

```bash
curl -s -X POST http://localhost:8100/v1/chat/completions \
  -H "content-type: application/json" \
  -d '{
    "model": "paddleocr",
    "messages": [
      { "role": "user", "content": [
          { "type": "text", "text": "Extract all text from this image." },
          { "type": "image_url",
            "image_url": { "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA…" } }
        ]
      }
    ]
  }' | jq -r '.choices[0].message.content' | jq .
```

→
```json
[
  { "bbox": { "x": 120, "y": 44, "w": 88, "h": 32 },
    "text": "Save", "confidence": 0.97 }
]
```

The runner's `OcrClient.extract()` does this call automatically when
the agent fires `POST /ui-bridge/vision/extract`.

## Configuration

Env vars, set on the llama-swap container or via the model's `cmd`:

| Var | Default | Purpose |
|---|---|---|
| `PADDLEOCR_LANG` | `en` | Language hint; comma-list for multi-lang (`en,ch`). |
| `PADDLEOCR_USE_GPU` | `0` | Set `1` to enable GPU inference (requires `paddlepaddle-gpu` wheel). |
| `PADDLEOCR_USE_ANGLE_CLS` | `1` | Enable text-angle classifier. Cheap, leave on. |

## Why an OpenAI-API shim, not a native PaddleOCR API?

The runner is intentionally backend-agnostic — it speaks one protocol
(OpenAI chat completions) to llama-swap and lets the proxy route to
whatever's behind the alias. That way a future swap (PaddleOCR →
Tesseract → a VLM-based OCR) is one alias edit, not a runner code
change. The shim is ~200 lines of FastAPI; the contract savings are
worth it.

## GPU variant

The default Dockerfile installs `paddlepaddle==2.6.1` (CPU). For a
GPU host, rebuild the llama-swap image after swapping the line:

```diff
-    paddlepaddle==2.6.1 \
+    paddlepaddle-gpu==2.6.1.post120 \
```

The CUDA version must match the host driver — `post120` is for CUDA
12.0, `post118` for 11.8, etc. PaddleOCR's GPU wheel matrix is
documented at https://www.paddlepaddle.org.cn/install/quick.
