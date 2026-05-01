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
import threading
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


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection-over-union for (x, y, width, height) bboxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


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
        # Guards the lazy-load paths against concurrent callers —
        # InteractabilityFilter + UI-TARS executor can both hit the same
        # detector from different threads and both trigger a YOLO load
        # without this.
        self._load_lock = threading.Lock()

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
    # Interactability classifier API (pre-filter mode)
    # ------------------------------------------------------------------

    def get_interactive_regions(
        self, haystack: np.ndarray
    ) -> list[tuple[int, int, int, int, float]]:
        """Run only the YOLO interactability head on a screenshot.

        Returns a list of (x, y, width, height, confidence) tuples for every
        region the OmniParser YOLO head classifies as interactive. Skips the
        OCR and Florence-2 stages — ~10x faster than ``detect_from_numpy``
        when the caller only needs interactability gating.
        """
        self._maybe_unload_idle()
        self._ensure_yolo_loaded()

        if len(haystack.shape) == 3 and haystack.shape[2] == 3:
            pil_img = Image.fromarray(haystack[..., ::-1])  # BGR -> RGB
        else:
            pil_img = Image.fromarray(haystack)

        raw_boxes = self._run_yolo(pil_img, self.get_default_parameters())
        self._last_used = time.perf_counter()

        img_h, img_w = haystack.shape[:2]
        regions: list[tuple[int, int, int, int, float]] = []
        for x1, y1, x2, y2, conf, _cls in raw_boxes:
            bx = max(0, int(x1))
            by = max(0, int(y1))
            bw = min(img_w, int(x2)) - bx
            bh = min(img_h, int(y2)) - by
            if bw > 0 and bh > 0:
                regions.append((bx, by, bw, bh, float(conf)))
        return regions

    def classify_region(
        self,
        haystack: np.ndarray,
        bbox: tuple[int, int, int, int],
        iou_threshold: float = 0.3,
        interactive_regions: list[tuple[int, int, int, int, float]] | None = None,
    ) -> tuple[bool, float]:
        """Classify a single bbox as interactive or not.

        Args:
            haystack: Full screenshot the bbox refers to.
            bbox: Candidate region as (x, y, width, height).
            iou_threshold: Minimum IoU against any YOLO-detected interactive
                region to count as interactive.
            interactive_regions: Optional pre-computed list from
                ``get_interactive_regions`` — pass this when classifying many
                candidates against the same haystack to avoid re-running YOLO.

        Returns:
            (is_interactive, confidence). ``confidence`` is the YOLO
            confidence of the best-overlapping interactive region, or the
            highest IoU if nothing overlaps (for diagnostics). Callers should
            use ``is_interactive`` for the gate decision.
        """
        if interactive_regions is None:
            interactive_regions = self.get_interactive_regions(haystack)

        if not interactive_regions:
            return (False, 0.0)

        best_iou = 0.0
        best_conf = 0.0
        for ix, iy, iw, ih, iconf in interactive_regions:
            iou = _bbox_iou(bbox, (ix, iy, iw, ih))
            if iou > best_iou:
                best_iou = iou
                best_conf = iconf

        is_interactive = best_iou >= iou_threshold
        return (is_interactive, best_conf if is_interactive else best_iou)

    def classify_point(
        self,
        haystack: np.ndarray,
        x: int,
        y: int,
        interactive_regions: list[tuple[int, int, int, int, float]] | None = None,
    ) -> tuple[bool, float]:
        """Classify a single (x, y) pixel as landing on an interactive element.

        Used by the UI-TARS executor hook to gate LLM-produced click
        coordinates before execution.
        """
        if interactive_regions is None:
            interactive_regions = self.get_interactive_regions(haystack)

        for ix, iy, iw, ih, iconf in interactive_regions:
            if ix <= x < ix + iw and iy <= y < iy + ih:
                return (True, iconf)
        return (False, 0.0)

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
                caption_map = self._run_captioning(
                    pil_img, raw_boxes, non_text_indices, params
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug(
            "OmniParser detected %d elements in %.0fms", len(raw_boxes), elapsed_ms
        )
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

    def _ensure_yolo_loaded(self) -> None:
        """Load only the YOLO interactability head.

        Used by the fast interactability-classifier API
        (``get_interactive_regions`` / ``classify_region`` / ``classify_point``)
        which doesn't need OCR or Florence-2 captioning. Keeping this a
        distinct step avoids dragging in heavy transformers dependencies
        just to gate a click.
        """
        if self._yolo_model is not None:
            return

        with self._load_lock:
            if self._yolo_model is not None:
                return  # another thread won the race while we waited

            if self._device is None:
                self._device = self._settings.resolve_device()
                logger.info(
                    "Loading OmniParser YOLO head on device=%s",
                    self._device,
                )

            # ultralytics' YOLO() requires a local .pt path and does not
            # resolve HuggingFace repo IDs. If the configured value points
            # at a non-existent path that looks like a HF repo id, download
            # icon_detect/model.pt from that repo and load from the local
            # cache path instead.
            from pathlib import Path as _Path

            from ultralytics import YOLO

            yolo_path = self._settings.model_path or self._settings.yolo_model
            if (
                not _Path(yolo_path).exists()
                and "/" in yolo_path
                and not yolo_path.endswith(".pt")
            ):
                from huggingface_hub import hf_hub_download

                logger.info(
                    "Resolving OmniParser YOLO weights from HF repo %s",
                    yolo_path,
                )
                yolo_path = hf_hub_download(yolo_path, "icon_detect/model.pt")
            self._yolo_model = YOLO(yolo_path)
            if self._device == "cuda":
                self._yolo_model.to("cuda")
            self._last_used = time.perf_counter()

    def _ensure_loaded(self) -> None:
        """Lazy-load all models on first use (full detection pipeline)."""
        if self._yolo_model is not None and self._caption_model is not None:
            return

        self._ensure_yolo_loaded()

        with self._load_lock:
            if self._caption_model is not None:
                return  # another thread loaded while we waited

            logger.info(
                "Loading OmniParser OCR + Florence-2 on device=%s (this may take a moment)",
                self._device,
            )

            # EasyOCR reader (already a qontinui dependency)
            if self._ocr_reader is None:
                import easyocr

                self._ocr_reader = easyocr.Reader(
                    ["en"], gpu=(self._device == "cuda"), verbose=False
                )

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
            ).to(
                self._device  # type: ignore[arg-type]
            )
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
    def _classify_element(
        label: str | None, width: int, height: int, cls_id: float
    ) -> str:
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
