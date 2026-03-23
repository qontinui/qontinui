"""OpenAI-compatible API server for Aria-UI using transformers + bitsandbytes.

Loads the model with 4-bit quantization (NF4) so the 25B MoE model fits
in ~13GB VRAM on an RTX 5090 (32GB).

Exposes a POST /v1/chat/completions endpoint that accepts image_url
messages (base64 or URL) and returns grounding coordinates.
"""

import argparse
import base64
import io
import logging
import time
import uuid
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("aria-ui")
logging.basicConfig(level=logging.INFO)

# Global model/processor refs set during lifespan
model = None
processor = None
model_name = None


def load_model(name: str):
    """Load Aria-UI with 4-bit quantization."""
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    logger.info("Loading processor from %s", name)
    proc = AutoProcessor.from_pretrained(name, trust_remote_code=True)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading model from %s with 4-bit quantization", name)
    mdl = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=quant_config,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    mdl.eval()

    # Patch projector's MultiheadAttention to work with quantized weights.
    # F.multi_head_attention_forward calls F.linear directly on out_proj_weight
    # which fails with quantized Byte weights. Replace MHA modules with a
    # wrapper that routes through the bnb Linear4bit forward instead.
    _patch_mha_for_bnb(mdl)

    logger.info("Model loaded successfully")
    return mdl, proc


def _patch_mha_for_bnb(mdl):
    """Replace nn.MultiheadAttention in the projector with a bnb-compatible version."""
    import torch.nn as nn

    for name, parent in mdl.named_modules():
        for attr_name, child in list(parent.named_children()):
            if not isinstance(child, nn.MultiheadAttention):
                continue
            if "projector" not in name and "projector" not in attr_name:
                continue

            # Replace MHA with a manual implementation that uses the
            # existing bnb-quantized in_proj and out_proj Linear4bit layers
            wrapper = _BnbMultiheadAttention(child)
            setattr(parent, attr_name, wrapper)
            logger.info("Patched MHA %s.%s for bnb compatibility", name, attr_name)


class _BnbMultiheadAttention(torch.nn.Module):
    """Drop-in replacement for nn.MultiheadAttention that works with bnb Linear4bit."""

    def __init__(self, orig_mha):
        super().__init__()
        self.num_heads = orig_mha.num_heads
        self.head_dim = orig_mha.head_dim
        self.embed_dim = orig_mha.embed_dim
        # Keep references to the original (possibly quantized) parameters
        self.in_proj_weight = orig_mha.in_proj_weight
        self.in_proj_bias = orig_mha.in_proj_bias
        self.out_proj = orig_mha.out_proj  # This is the bnb Linear4bit

    def forward(self, query, key, value, attn_mask=None, **kwargs):
        # Manual multi-head attention that routes through bnb linear layers
        bsz, tgt_len, embed_dim = query.size()

        # in_projection: project Q, K, V using the packed in_proj_weight
        # Use F.linear which will dequantize if needed via bnb
        w = self.in_proj_weight
        b = self.in_proj_bias

        # Dequantize if it's a bnb parameter
        if hasattr(w, "quant_state"):
            import bitsandbytes.functional as bnb_F

            w = bnb_F.dequantize_4bit(w.data, w.quant_state).to(query.dtype)

        if b is not None:
            proj = torch.nn.functional.linear(query, w, b)
        else:
            proj = torch.nn.functional.linear(query, w)

        q, k, v = proj.chunk(3, dim=-1)

        # Reshape for multi-head
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)

        # out_proj: this is a bnb Linear4bit, calling its forward will dequantize properly
        attn_output = self.out_proj(attn_output)

        return attn_output, None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, model_name
    model, processor = load_model(model_name)
    yield


app = FastAPI(lifespan=lifespan)


# --- Pydantic models for OpenAI-compatible API ---


class ImageUrl(BaseModel):
    url: str


class ContentPart(BaseModel):
    type: str
    text: str | None = None
    image_url: ImageUrl | None = None


class Message(BaseModel):
    role: str
    content: str | list[ContentPart]


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int = 512
    temperature: float = 0.0
    stream: bool = False


def _extract_images_and_text(messages: list[Message]):
    """Parse OpenAI-format messages into text + PIL images."""
    import requests
    from PIL import Image

    images = []
    text_parts = []

    for msg in messages:
        if isinstance(msg.content, str):
            text_parts.append(msg.content)
            continue
        for part in msg.content:
            if part.type == "text" and part.text:
                text_parts.append(part.text)
            elif part.type == "image_url" and part.image_url:
                url = part.image_url.url
                if url.startswith("data:"):
                    # base64 encoded
                    header, data = url.split(",", 1)
                    img_bytes = base64.b64decode(data)
                    images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                else:
                    # URL
                    resp = requests.get(url, timeout=30)
                    images.append(Image.open(io.BytesIO(resp.content)).convert("RGB"))

    # Resize large images to reduce VRAM usage during vision encoding
    MAX_DIM = 1024
    resized = []
    for img in images:
        w, h = img.size
        if max(w, h) > MAX_DIM:
            scale = MAX_DIM / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        resized.append(img)

    return "\n".join(text_parts), resized


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    try:
        text, images = _extract_images_and_text(req.messages)

        # Build chat messages for the processor's apply_chat_template
        chat_messages = [{"role": "user", "content": []}]
        for _img in images:
            chat_messages[0]["content"].append({"type": "image"})
        chat_messages[0]["content"].append({"type": "text", "text": text})

        # Use the processor's chat template to format correctly
        if hasattr(processor, "apply_chat_template"):
            prompt = processor.apply_chat_template(
                chat_messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            # Fallback: manual image token insertion
            image_tokens = "<|img|>" * len(images) if images else ""
            prompt = f"{image_tokens}\n{text}" if image_tokens else text

        # Process inputs
        inputs = processor(
            text=prompt,
            images=images if images else None,
            return_tensors="pt",
        )

        # Move to device and cast float tensors to bfloat16 to match model weights
        def _prepare(k, v):
            if not hasattr(v, "to"):
                return v
            v = v.to(model.device)
            if v.is_floating_point():
                v = v.to(torch.bfloat16)
            return v

        inputs = {k: _prepare(k, v) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=req.temperature > 0,
                temperature=req.temperature if req.temperature > 0 else None,
            )

        # Decode only the new tokens
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[0][input_len:]
        response_text = processor.decode(generated, skip_special_tokens=True)

        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": input_len,
                    "completion_tokens": len(generated),
                    "total_tokens": input_len + len(generated),
                },
            }
        )

    except Exception as e:
        logger.exception("Error processing chat completion")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error"}},
        )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "owned_by": "aria-ui",
            }
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Aria-UI/Aria-UI-base")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    args = parser.parse_args()

    model_name = args.model
    uvicorn.run(app, host=args.host, port=args.port)
