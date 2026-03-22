"""OmniParser HTTP service.

Exposes the OmniParser YOLO + EasyOCR + Florence-2 pipeline as a REST API.
Compatible with qontinui's OmniParserServiceBackend.

Endpoints:
    GET  /health          — liveness probe
    POST /parse           — detect elements in a screenshot image
    GET  /info            — model and device info
"""

from __future__ import annotations

import io
import logging
import os
import time

import easyocr
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omniparser-service")

app = FastAPI(title="OmniParser Service", version="1.0.0")

# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_MODEL_ID = os.environ.get("OMNIPARSER_YOLO_MODEL", "microsoft/OmniParser-v2.0")
CAPTION_MODEL_ID = os.environ.get("OMNIPARSER_CAPTION_MODEL", "microsoft/Florence-2-base")

yolo_model: YOLO | None = None
caption_processor = None
caption_model = None
ocr_reader: easyocr.Reader | None = None


@app.on_event("startup")
def load_models():
    global yolo_model, caption_processor, caption_model, ocr_reader

    logger.info("Loading OmniParser models on device=%s ...", DEVICE)

    # YOLO
    yolo_model = YOLO(YOLO_MODEL_ID)
    if DEVICE == "cuda":
        yolo_model.to("cuda")
    logger.info("YOLO model loaded")

    # EasyOCR
    ocr_reader = easyocr.Reader(["en"], gpu=(DEVICE == "cuda"), verbose=False)
    logger.info("EasyOCR loaded")

    # Florence-2
    caption_processor = AutoProcessor.from_pretrained(CAPTION_MODEL_ID, trust_remote_code=True)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    caption_model = AutoModelForCausalLM.from_pretrained(
        CAPTION_MODEL_ID,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(DEVICE)
    caption_model.eval()
    logger.info("Florence-2 loaded")

    logger.info("All models ready")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": yolo_model is not None}


@app.get("/info")
def info():
    return {
        "device": DEVICE,
        "yolo_model": YOLO_MODEL_ID,
        "caption_model": CAPTION_MODEL_ID,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/parse")
async def parse(
    image: UploadFile = File(...),
    iou_threshold: float = Form(0.3),
    confidence_threshold: float = Form(0.3),
):
    """Detect UI elements in a screenshot.

    Returns a JSON list of detected elements with bounding boxes,
    labels, confidence scores, and element types.
    """
    t0 = time.perf_counter()

    img_bytes = await image.read()
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_w, img_h = pil_img.size

    # Stage 1: YOLO detection
    results = yolo_model(
        pil_img,
        conf=confidence_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = float(box.cls[0].cpu().numpy())
            boxes.append(
                {
                    "xyxy": xyxy.tolist(),
                    "confidence": conf,
                    "class_id": int(cls),
                }
            )

    # Stage 2: OCR on each region
    img_np = np.array(pil_img)
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
        crop = img_np[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        try:
            text_results = ocr_reader.readtext(crop, detail=0)
            text = " ".join(text_results).strip()
            if text and len(text) >= 2:
                b["ocr_text"] = text
        except Exception:
            pass

    # Stage 3: Florence-2 captioning on non-text regions
    for b in boxes:
        if "ocr_text" in b:
            continue
        x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
        crop_img = pil_img.crop((x1, y1, x2, y2))
        if crop_img.size[0] == 0 or crop_img.size[1] == 0:
            continue
        try:
            inputs = caption_processor(
                text="<CAPTION>",
                images=crop_img,
                return_tensors="pt",
            ).to(DEVICE)
            with torch.no_grad():
                gen_ids = caption_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=50,
                    num_beams=3,
                )
            caption = caption_processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
            if caption:
                b["caption"] = caption
        except Exception:
            pass

    # Build response
    elements = []
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
        label = b.get("ocr_text") or b.get("caption")
        elements.append(
            {
                "bbox": [x1, y1, x2, y2],
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "confidence": b["confidence"],
                "label": label,
                "type": _classify(label, x2 - x1, y2 - y1),
                "has_ocr": "ocr_text" in b,
                "has_caption": "caption" in b,
            }
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("Parsed %d elements in %.0fms", len(elements), elapsed_ms)

    return JSONResponse(
        {
            "elements": elements,
            "image_size": {"width": img_w, "height": img_h},
            "elapsed_ms": round(elapsed_ms),
            "device": DEVICE,
        }
    )


def _classify(label: str | None, w: int, h: int) -> str:
    """Simple element type classification from label and size."""
    if label:
        ll = label.lower()
        if any(kw in ll for kw in ("button", "btn", "submit", "cancel", "ok")):
            return "button"
        if any(kw in ll for kw in ("input", "search", "enter", "type")):
            return "text_field"
        if any(kw in ll for kw in ("checkbox", "check box")):
            return "checkbox"
        if any(kw in ll for kw in ("dropdown", "select", "combo")):
            return "dropdown"
        if any(kw in ll for kw in ("link", "http")):
            return "link"
    if w < 40 and h < 40:
        return "icon"
    aspect = w / h if h > 0 else 1.0
    if aspect > 4 and h < 40:
        return "text_field"
    if 1.5 < aspect < 6 and h < 60:
        return "button"
    return "interactive_element"
