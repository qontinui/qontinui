"""OpenAI-compatible API server for PaddleOCR.

Wraps PaddleOCR (https://github.com/PaddlePaddle/PaddleOCR) with the
`POST /v1/chat/completions` endpoint llama-swap expects, so the runner's
`OcrClient` (qontinui-runner@30f74d59, src-tauri/src/mcp/ui_bridge/
vision_ai.rs) can route through the same OpenAI-shaped proxy it uses for
VLMs.

The model field in the chat-completions request selects which model
llama-swap dispatches to; PaddleOCR-shaped requests come in as
`{"model": "paddleocr", "messages": [..., {"image_url": ...}, ...]}`.
We extract the image, run PaddleOCR, and emit a single-choice response
whose `message.content` is the JSON array the runner's OCR_SYSTEM_PROMPT
asks the model to produce. The runner's parser is fence-tolerant + handles
bare arrays.

Why match the prompt-shape contract? Because the runner has *no idea*
which OCR engine is behind the alias — it asks the model to emit JSON
in a specific shape, and we (as the "model") oblige. This keeps the
runner backend-agnostic.

Env vars:
  PADDLEOCR_LANG       — language hint, default "en". Comma-separated for
                         multi-language (e.g., "en,ch").
  PADDLEOCR_USE_GPU    — "1" to enable GPU inference. Default off (CPU)
                         since most callers are happy with classical OCR
                         on CPU.
  PADDLEOCR_USE_ANGLE_CLS — "1" to enable text-angle classification.
                            Default on; classifier is cheap.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("paddleocr-serve")
logging.basicConfig(level=logging.INFO)

# Lazy globals so import doesn't trigger PaddleOCR model download.
_ocr = None  # type: ignore[var-annotated]


def load_ocr():
    """Import + initialize PaddleOCR. Called from the FastAPI lifespan."""
    from paddleocr import PaddleOCR  # type: ignore[import-not-found]

    lang = os.environ.get("PADDLEOCR_LANG", "en")
    use_gpu = os.environ.get("PADDLEOCR_USE_GPU", "0") == "1"
    use_angle_cls = os.environ.get("PADDLEOCR_USE_ANGLE_CLS", "1") == "1"

    logger.info(
        "Initializing PaddleOCR (lang=%s, use_gpu=%s, use_angle_cls=%s)",
        lang,
        use_gpu,
        use_angle_cls,
    )
    return PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, use_gpu=use_gpu, show_log=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ocr
    _ocr = load_ocr()
    yield
    _ocr = None


app = FastAPI(title="PaddleOCR OpenAI-compatible shim", lifespan=lifespan)


class ChatMessageContent(BaseModel):
    type: str
    text: str | None = None
    image_url: dict[str, Any] | None = None


class ChatMessage(BaseModel):
    role: str
    content: list[ChatMessageContent] | str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 0.0
    max_tokens: int | None = 4096


def _extract_image_bytes(messages: list[ChatMessage]) -> bytes | None:
    """Pull the first image_url out of the message stream, decode if it's
    a `data:image/...;base64,...` URI."""
    for msg in messages:
        if isinstance(msg.content, str):
            continue
        for part in msg.content:
            if part.type == "image_url" and part.image_url:
                url = part.image_url.get("url", "")
                if url.startswith("data:"):
                    # data:image/png;base64,XXXX
                    head, _, payload = url.partition(",")
                    if "base64" in head and payload:
                        try:
                            return base64.b64decode(payload)
                        except Exception:  # noqa: BLE001
                            logger.exception("base64 decode failed")
                            return None
                else:
                    # Remote URL — out of scope; the runner always sends data: URIs.
                    parsed = urlparse(url)
                    logger.warning("non-data image_url received: %s", parsed.scheme)
                    return None
    return None


def _run_paddleocr(image_bytes: bytes) -> list[dict[str, Any]]:
    """Run PaddleOCR on raw image bytes; return blocks in the shape the
    runner's OcrClient expects."""
    from PIL import Image  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)

    assert _ocr is not None, "PaddleOCR not initialized"
    raw = _ocr.ocr(arr, cls=True)

    blocks: list[dict[str, Any]] = []
    # PaddleOCR result shape: List[List[[bbox_poly, (text, confidence)], ...]]
    # The outer list is per-page (always one image here, so [0]).
    if not raw or not raw[0]:
        return blocks
    for det in raw[0]:
        if not det or len(det) < 2:
            continue
        bbox_poly, text_conf = det[0], det[1]
        if not bbox_poly or not text_conf or len(text_conf) < 2:
            continue
        text, confidence = text_conf[0], float(text_conf[1])
        # Convert polygon to axis-aligned bbox (x, y, w, h).
        xs = [int(p[0]) for p in bbox_poly]
        ys = [int(p[1]) for p in bbox_poly]
        x = max(0, min(xs))
        y = max(0, min(ys))
        w = max(0, max(xs) - x)
        h = max(0, max(ys) - y)
        if not text or w == 0 or h == 0:
            continue
        blocks.append(
            {
                "bbox": {"x": x, "y": y, "w": w, "h": h},
                "text": text,
                "confidence": confidence,
            }
        )
    return blocks


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "ready": _ocr is not None}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest) -> JSONResponse:
    """OpenAI-shaped façade. Pulls the image, runs OCR, returns the JSON
    array as the assistant message's `content`."""
    if _ocr is None:
        return JSONResponse(
            {"error": {"message": "PaddleOCR not initialized", "type": "server_error"}},
            status_code=503,
        )
    img_bytes = _extract_image_bytes(request.messages)
    if img_bytes is None:
        return JSONResponse(
            {
                "error": {
                    "message": (
                        "PaddleOCR shim requires a `data:image/...;base64,...` "
                        "image_url in the messages stream"
                    ),
                    "type": "invalid_request_error",
                }
            },
            status_code=400,
        )

    import json as _json

    t0 = time.perf_counter()
    blocks = _run_paddleocr(img_bytes)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info("paddleocr extracted %d blocks in %d ms", len(blocks), elapsed_ms)
    body = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": _json.dumps(blocks),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,  # PaddleOCR is not token-billed
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    return JSONResponse(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
