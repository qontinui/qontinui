"""OmniParser-based UI element detector.

Wraps Microsoft's OmniParser (YOLO + Florence-2) for zero-shot GUI element
detection with semantic labeling. Implements BaseAnalyzer so it plugs into
the existing discovery pipeline.

OmniParser pipeline:
1. YOLO detection — finds interactive element bounding boxes
2. OCR (EasyOCR) — extracts text from detected regions
3. Florence-2 captioning — generates semantic descriptions for icon regions

Models are lazy-loaded to avoid GPU memory waste when not in use.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any

import numpy as np
from PIL import Image

from qontinui.find.backends.omniparser_config import OmniParserSettings

from .analysis_base import (
    AnalysisInput,
    AnalysisResult,
    AnalysisType,
    BaseAnalyzer,
    BoundingBox,
    DetectedElement,
)

logger = logging.getLogger(__name__)


class OmniParserDetector(BaseAnalyzer):
    """Zero-shot UI element detector using Microsoft OmniParser.

    Detects interactive UI elements and provides semantic labels without
    requiring pre-recorded reference images. Suitable as a Tier 2 fallback
    between template matching and full VLM grounding.

    Models are lazy-loaded on first call. GPU is preferred but CPU works (slower).
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        settings: OmniParserSettings | None = None,
    ) -> None:
        super().__init__(config)
        self._settings = settings or OmniParserSettings()
        self._yolo_model: Any = None
        self._caption_processor: Any = None
        self._caption_model: Any = None
        self._ocr_reader: Any = None
        self._device: str | None = None
        self._last_used: float = 0.0

    @property
    def analysis_type(self) -> AnalysisType:
        return AnalysisType.SINGLE_SHOT

    @property
    def name(self) -> str:
        return "omniparser_detector"

    @property
    def supports_multi_screenshot(self) -> bool:
        return False

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "iou_threshold": self._settings.iou_threshold,
            "confidence_threshold": self._settings.confidence_threshold,
            "caption_batch_size": self._settings.caption_batch_size,
            "enable_captioning": True,
            "enable_ocr": True,
        }

    async def analyze(self, input_data: AnalysisInput) -> AnalysisResult:
        """Run OmniParser detection on input screenshots.

        Args:
            input_data: AnalysisInput with screenshot_data (list of image bytes).

        Returns:
            AnalysisResult with DetectedElement objects containing semantic labels.
        """
        params = {**self.get_default_parameters(), **input_data.parameters}

        all_elements: list[DetectedElement] = []
        for idx, img_bytes in enumerate(input_data.screenshot_data):
            elements = self._detect_screenshot(img_bytes, idx, params)
            all_elements.extend(elements)

        return AnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            elements=all_elements,
            confidence=self._overall_confidence(all_elements),
            metadata={
                "num_screenshots": len(input_data.screenshot_data),
                "total_elements": len(all_elements),
                "device": self._device or "not_loaded",
            },
        )

    def detect_from_numpy(self, screenshot: np.ndarray) -> list[DetectedElement]:
        """Convenience method: detect elements from a numpy array (BGR or RGB).

        This bypasses the AnalysisInput wrapper for simpler integration
        with the DetectionBackend / cascade pipeline.
        """
        if len(screenshot.shape) == 3 and screenshot.shape[2] == 3:
            pil_img = Image.fromarray(screenshot[..., ::-1])  # BGR -> RGB
        else:
            pil_img = Image.fromarray(screenshot)

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return self._detect_screenshot(buf.getvalue(), 0, self.get_default_parameters())

    # ------------------------------------------------------------------
    # Internal detection pipeline
    # ------------------------------------------------------------------

    def _detect_screenshot(
        self, img_bytes: bytes, screenshot_index: int, params: dict[str, Any]
    ) -> list[DetectedElement]:
        """Run full OmniParser pipeline on one screenshot."""
        self._maybe_unload_idle()
        self._ensure_loaded()

        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_w, img_h = pil_img.size

        t0 = time.perf_counter()

        # Stage 1: YOLO detection
        raw_boxes = self._run_yolo(pil_img, params)

        # Stage 2: OCR on detected text regions
        ocr_map: dict[int, str] = {}
        if params.get("enable_ocr", True) and raw_boxes:
            ocr_map = self._run_ocr(pil_img, raw_boxes)

        # Stage 3: Florence-2 captioning on icon/non-text regions
        caption_map: dict[int, str] = {}
        if params.get("enable_captioning", True) and raw_boxes:
            non_text_indices = [i for i in range(len(raw_boxes)) if i not in ocr_map]
            if non_text_indices:
                caption_map = self._run_captioning(pil_img, raw_boxes, non_text_indices, params)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("OmniParser detected %d elements in %.0fms", len(raw_boxes), elapsed_ms)
        self._last_used = time.perf_counter()

        # Build DetectedElement list
        elements: list[DetectedElement] = []
        for i, box in enumerate(raw_boxes):
            x1, y1, x2, y2, conf, cls_id = box

            # Clamp to image bounds
            bx = max(0, int(x1))
            by = max(0, int(y1))
            bw = min(img_w, int(x2)) - bx
            bh = min(img_h, int(y2)) - by
            if bw <= 0 or bh <= 0:
                continue

            # Determine label and element_type
            label = ocr_map.get(i) or caption_map.get(i)
            element_type = self._classify_element(label, bw, bh, cls_id)

            elements.append(
                DetectedElement(
                    bounding_box=BoundingBox(x=bx, y=by, width=bw, height=bh),
                    confidence=float(conf),
                    label=label,
                    element_type=element_type,
                    screenshot_index=screenshot_index,
                    metadata={
                        "source": "omniparser",
                        "yolo_class": int(cls_id),
                        "has_ocr_text": i in ocr_map,
                        "has_caption": i in caption_map,
                    },
                )
            )

        return elements

    def _run_yolo(
        self, pil_img: Image.Image, params: dict[str, Any]
    ) -> list[tuple[float, float, float, float, float, float]]:
        """Run YOLO detection, return list of (x1, y1, x2, y2, conf, cls)."""
        results = self._yolo_model(
            pil_img,
            conf=params.get("confidence_threshold", 0.3),
            iou=params.get("iou_threshold", 0.3),
            verbose=False,
        )

        boxes: list[tuple[float, float, float, float, float, float]] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = float(box.cls[0].cpu().numpy())
                boxes.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls))

        return boxes

    def _run_ocr(
        self,
        pil_img: Image.Image,
        boxes: list[tuple[float, float, float, float, float, float]],
    ) -> dict[int, str]:
        """Run OCR on regions that likely contain text. Returns {box_index: text}."""
        img_np = np.array(pil_img)
        ocr_map: dict[int, str] = {}

        for i, (x1, y1, x2, y2, _, _) in enumerate(boxes):
            crop = img_np[int(y1) : int(y2), int(x1) : int(x2)]
            if crop.size == 0:
                continue
            try:
                results = self._ocr_reader.readtext(crop, detail=0)
                text = " ".join(results).strip()
                if text and len(text) >= 2:
                    ocr_map[i] = text
            except Exception:
                logger.debug("OCR failed for box %d", i, exc_info=True)

        return ocr_map

    def _run_captioning(
        self,
        pil_img: Image.Image,
        boxes: list[tuple[float, float, float, float, float, float]],
        indices: list[int],
        params: dict[str, Any],
    ) -> dict[int, str]:
        """Run Florence-2 captioning on icon/non-text regions."""
        import torch

        batch_size = params.get("caption_batch_size", 64)
        caption_map: dict[int, str] = {}

        # Crop regions
        crops: list[Image.Image] = []
        valid_indices: list[int] = []
        for idx in indices:
            x1, y1, x2, y2, _, _ = boxes[idx]
            crop = pil_img.crop((int(x1), int(y1), int(x2), int(y2)))
            if crop.size[0] > 0 and crop.size[1] > 0:
                crops.append(crop)
                valid_indices.append(idx)

        if not crops:
            return caption_map

        # Process in batches
        for batch_start in range(0, len(crops), batch_size):
            batch_crops = crops[batch_start : batch_start + batch_size]
            batch_indices = valid_indices[batch_start : batch_start + batch_size]

            try:
                for crop_img, box_idx in zip(batch_crops, batch_indices, strict=True):
                    inputs = self._caption_processor(
                        text="<CAPTION>",
                        images=crop_img,
                        return_tensors="pt",
                    ).to(self._device)

                    with torch.no_grad():
                        generated_ids = self._caption_model.generate(
                            input_ids=inputs["input_ids"],
                            pixel_values=inputs["pixel_values"],
                            max_new_tokens=50,
                            num_beams=3,
                        )

                    caption = self._caption_processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0].strip()

                    if caption:
                        caption_map[box_idx] = caption
            except Exception:
                logger.warning(
                    "Florence-2 captioning failed for batch starting at %d",
                    batch_start,
                    exc_info=True,
                )

        return caption_map

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _maybe_unload_idle(self) -> None:
        """Unload models if idle longer than unload_after_seconds."""
        timeout = self._settings.unload_after_seconds
        if timeout <= 0 or not self.is_loaded:
            return
        if self._last_used > 0 and (time.perf_counter() - self._last_used) > timeout:
            logger.info(
                "OmniParser idle for %.0fs (threshold %.0fs), unloading",
                time.perf_counter() - self._last_used,
                timeout,
            )
            self.unload()

    def _ensure_loaded(self) -> None:
        """Lazy-load all models on first use."""
        if self._yolo_model is not None:
            return

        self._device = self._settings.resolve_device()
        logger.info(
            "Loading OmniParser models on device=%s (this may take a moment)",
            self._device,
        )

        # YOLO model
        from ultralytics import YOLO

        yolo_path = self._settings.model_path or self._settings.yolo_model
        self._yolo_model = YOLO(yolo_path)
        if self._device == "cuda":
            self._yolo_model.to("cuda")

        # EasyOCR reader (already a qontinui dependency)
        import easyocr

        self._ocr_reader = easyocr.Reader(["en"], gpu=(self._device == "cuda"), verbose=False)

        # Florence-2 captioning model
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._caption_processor = AutoProcessor.from_pretrained(
            self._settings.caption_model, trust_remote_code=True
        )
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._caption_model = AutoModelForCausalLM.from_pretrained(
            self._settings.caption_model,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self._device)
        self._caption_model.eval()

        self._last_used = time.perf_counter()
        logger.info("OmniParser models loaded successfully")

    def unload(self) -> None:
        """Explicitly unload models to free GPU memory."""
        if self._yolo_model is None:
            return

        import gc

        self._yolo_model = None
        self._caption_model = None
        self._caption_processor = None
        self._ocr_reader = None
        self._device = None

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        gc.collect()
        logger.info("OmniParser models unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._yolo_model is not None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_element(label: str | None, width: int, height: int, cls_id: float) -> str:
        """Infer element_type from label text, size, and YOLO class."""
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
            if any(kw in ll for kw in ("toggle", "switch")):
                return "toggle"
            if any(kw in ll for kw in ("link", "click here", "http")):
                return "link"
            if any(kw in ll for kw in ("menu", "nav")):
                return "menu_item"
            if any(kw in ll for kw in ("tab",)):
                return "tab"

        # Size-based heuristics
        aspect = width / height if height > 0 else 1.0
        if width < 40 and height < 40:
            return "icon"
        if aspect > 4 and height < 40:
            return "text_field"
        if 1.5 < aspect < 6 and height < 60:
            return "button"

        return "interactive_element"

    @staticmethod
    def _overall_confidence(elements: list[DetectedElement]) -> float:
        """Compute overall confidence from individual element confidences."""
        if not elements:
            return 0.0
        return sum(e.confidence for e in elements) / len(elements)
