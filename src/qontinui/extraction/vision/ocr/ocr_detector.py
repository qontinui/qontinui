"""
OCR Detection for StateImage Candidate Discovery.

Uses OCR (EasyOCR or Tesseract) to detect text elements in screenshots.
Text regions often correspond to labels, buttons, or other interactive elements.
"""

import base64
import logging
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from ..models import (
    BoundingBox,
    ExtractedStateImageCandidate,
    OCRConfig,
    OCRResult,
)

logger = logging.getLogger(__name__)


class OCRDetector:
    """
    OCR-based text detection for UI element discovery.

    Uses EasyOCR (default) or Tesseract to detect text regions,
    which often correspond to labels, buttons, and other UI elements.
    """

    def __init__(self, config: OCRConfig | None = None) -> None:
        """
        Initialize the OCR detector.

        Args:
            config: OCR configuration. Uses defaults if not provided.
        """
        self.config = config or OCRConfig()
        self._ocr_engine = None
        self._ocr_available = False
        self._result_counter = 0

        if self.config.enabled:
            self._try_load_ocr()

    def _try_load_ocr(self) -> None:
        """Try to load OCR engine."""
        if self.config.engine == "easyocr":
            self._try_load_easyocr()
        elif self.config.engine == "tesseract":
            self._try_load_tesseract()
        else:
            logger.warning(f"Unknown OCR engine: {self.config.engine}")

    def _try_load_easyocr(self) -> None:
        """Try to load EasyOCR."""
        try:
            import easyocr

            self._ocr_engine = easyocr.Reader(
                self.config.languages,
                gpu=True,  # Will fall back to CPU if GPU not available
            )
            self._ocr_available = True
            logger.info(f"EasyOCR loaded with languages: {self.config.languages}")

        except ImportError:
            logger.warning("EasyOCR not installed. Install with: pip install easyocr")
        except Exception as e:
            logger.warning(f"Failed to load EasyOCR: {e}")

    def _try_load_tesseract(self) -> None:
        """Try to load Tesseract OCR."""
        try:
            import pytesseract

            # Test that tesseract is available
            pytesseract.get_tesseract_version()

            self._ocr_engine = pytesseract
            self._ocr_available = True
            logger.info("Tesseract OCR loaded")

        except ImportError:
            logger.warning("pytesseract not installed. Install with: pip install pytesseract")
        except Exception as e:
            logger.warning(f"Failed to load Tesseract: {e}")

    @property
    def is_available(self) -> bool:
        """Check if OCR is available."""
        return self._ocr_available

    def detect(
        self,
        screenshot: np.ndarray,
        screenshot_id: str,
    ) -> tuple[list[OCRResult], np.ndarray | None]:
        """
        Detect text regions using OCR.

        Args:
            screenshot: BGR image as numpy array.
            screenshot_id: ID of the screenshot for reference.

        Returns:
            Tuple of:
                - List of OCRResult
                - OCR overlay image (debug visualization)
        """
        if not self.config.enabled:
            return [], None

        if not self._ocr_available:
            logger.warning("OCR not available, skipping text detection")
            return [], None

        logger.info("Running OCR detection...")

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)

        try:
            if self.config.engine == "easyocr":
                results = self._detect_with_easyocr(image_rgb)
            else:
                results = self._detect_with_tesseract(image_rgb)

            # Filter by confidence
            filtered_results = [
                r for r in results if r.confidence >= self.config.confidence_threshold
            ]

            # Filter by text height
            filtered_results = [
                r
                for r in filtered_results
                if self.config.min_text_height <= r.bbox.height <= self.config.max_text_height
            ]

            # Create debug overlay
            overlay = self._create_ocr_overlay(screenshot, filtered_results)

            logger.info(
                f"OCR found {len(results)} text regions, " f"{len(filtered_results)} passed filters"
            )

            return filtered_results, overlay

        except Exception as e:
            logger.error(f"OCR detection failed: {e}")
            return [], None

    def _detect_with_easyocr(self, image_rgb: np.ndarray) -> list[OCRResult]:
        """Detect text using EasyOCR."""
        results: list[OCRResult] = []

        # Run OCR
        assert self._ocr_engine is not None, "EasyOCR engine not loaded"
        ocr_results = self._ocr_engine.readtext(image_rgb)

        for bbox_points, text, confidence in ocr_results:
            # Convert bbox format from [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] to (x,y,w,h)
            x_coords = [int(p[0]) for p in bbox_points]
            y_coords = [int(p[1]) for p in bbox_points]
            x = min(x_coords)
            y = min(y_coords)
            w = max(x_coords) - x
            h = max(y_coords) - y

            self._result_counter += 1
            result_id = f"ocr_{self._result_counter:06d}"

            result = OCRResult(
                id=result_id,
                bbox=BoundingBox(x=x, y=y, width=w, height=h),
                text=text,
                confidence=float(confidence),
                language=self.config.languages[0] if self.config.languages else "en",
                word_boxes=[
                    {
                        "points": [[int(p[0]), int(p[1])] for p in bbox_points],
                        "text": text,
                        "confidence": float(confidence),
                    }
                ],
            )
            results.append(result)

        return results

    def _detect_with_tesseract(self, image_rgb: np.ndarray) -> list[OCRResult]:
        """Detect text using Tesseract."""
        results: list[OCRResult] = []

        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)

        # Get detailed OCR data
        assert self._ocr_engine is not None, "Tesseract engine not loaded"
        data = self._ocr_engine.image_to_data(
            pil_image,
            output_type=self._ocr_engine.Output.DICT,
            lang="+".join(self.config.languages) if self.config.languages else "eng",
        )

        # Group by block/line to get text regions
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = data["text"][i].strip()
            if not text:
                continue

            conf = float(data["conf"][i])
            if conf < 0:  # Tesseract returns -1 for invalid
                continue

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            self._result_counter += 1
            result_id = f"ocr_{self._result_counter:06d}"

            result = OCRResult(
                id=result_id,
                bbox=BoundingBox(x=x, y=y, width=w, height=h),
                text=text,
                confidence=conf / 100.0,  # Tesseract returns 0-100
                language=self.config.languages[0] if self.config.languages else "en",
            )
            results.append(result)

        return results

    def _create_ocr_overlay(
        self,
        screenshot: np.ndarray,
        results: list[OCRResult],
    ) -> np.ndarray:
        """
        Create debug visualization with OCR boxes and text overlaid.

        Args:
            screenshot: Original screenshot.
            results: OCR detection results.

        Returns:
            BGR image with OCR overlays.
        """
        overlay = screenshot.copy()

        for result in results:
            bbox = result.bbox

            # Color based on confidence (green = high, red = low)
            conf = result.confidence
            green = int(conf * 255)
            red = int((1 - conf) * 255)
            color = (0, green, red)

            # Draw bounding box
            cv2.rectangle(
                overlay,
                (bbox.x, bbox.y),
                (bbox.x2, bbox.y2),
                color,
                2,
            )

            # Draw text label above box
            label = f"{result.text[:20]}..." if len(result.text) > 20 else result.text
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

            # Background for text
            cv2.rectangle(
                overlay,
                (bbox.x, bbox.y - label_size[1] - 4),
                (bbox.x + label_size[0] + 4, bbox.y),
                color,
                -1,
            )

            # Text
            cv2.putText(
                overlay,
                label,
                (bbox.x + 2, bbox.y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        return overlay

    def results_to_candidates(
        self,
        ocr_results: list[OCRResult],
        screenshot_id: str,
    ) -> list[ExtractedStateImageCandidate]:
        """
        Convert OCR results to StateImage candidates.

        Args:
            ocr_results: Results from OCR detection.
            screenshot_id: ID of the source screenshot.

        Returns:
            List of ExtractedStateImageCandidate.
        """
        candidates = []

        for result in ocr_results:
            # Classify based on text content and size
            category = self._classify_text(result)

            candidate = ExtractedStateImageCandidate(
                id=f"ocr_{result.id}",
                bbox=result.bbox,
                confidence=result.confidence,
                screenshot_id=screenshot_id,
                category=category,
                text=result.text,
                detection_technique="ocr",
                is_clickable=(category in ("button", "link")),
                metadata={
                    "language": result.language,
                    "text_length": len(result.text),
                },
            )
            candidates.append(candidate)

        return candidates

    def _classify_text(self, result: OCRResult) -> str:
        """
        Classify text region as element category.

        This is for description only - categories don't have
        functional significance in the state machine.
        """
        text = result.text.lower().strip()
        bbox = result.bbox

        # Check for common button text patterns
        button_keywords = [
            "submit",
            "cancel",
            "ok",
            "yes",
            "no",
            "save",
            "delete",
            "add",
            "remove",
            "edit",
            "update",
            "create",
            "close",
            "next",
            "back",
            "previous",
            "continue",
            "done",
            "finish",
            "login",
            "logout",
            "sign in",
            "sign out",
            "sign up",
            "search",
            "filter",
            "sort",
            "reset",
            "clear",
        ]

        if any(keyword in text for keyword in button_keywords):
            return "button"

        # Check for link-like text (URLs, "click here", etc.)
        if text.startswith("http") or "click" in text or "learn more" in text:
            return "link"

        # Short text with specific aspect ratio: likely button
        if len(text) < 15 and 1.5 < bbox.width / max(bbox.height, 1) < 6:
            return "button"

        # Longer text: label or paragraph
        if len(text) > 50:
            return "paragraph"

        # Default to label
        return "label"

    def get_overlay_base64(self, overlay: np.ndarray) -> str:
        """
        Convert overlay image to base64 for API transport.

        Args:
            overlay: BGR numpy array.

        Returns:
            Base64-encoded PNG string.
        """
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")

        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def reset_counter(self) -> None:
        """Reset the result counter (call between screenshots)."""
        self._result_counter = 0
