"""OCR-based detection backend.

Wraps the HAL-level IOCREngine as a DetectionBackend for text-based
element detection. Moderate cost (~300ms).
"""

import logging
from typing import Any

from .base import DetectionBackend, DetectionResult

logger = logging.getLogger(__name__)


class OCRBackend(DetectionBackend):
    """Detection backend using OCR text recognition.

    Wraps ``IOCREngine`` from ``hal.interfaces.ocr_engine``.
    Finds elements by searching for text content in the screenshot.

    Args:
        ocr_engine: An existing ``IOCREngine`` implementation.
                    Must be provided — no default is created since
                    OCR engines require platform-specific setup.
    """

    def __init__(self, ocr_engine: Any) -> None:
        self._engine = ocr_engine

    def find(
        self, needle: Any, haystack: Any, config: dict[str, Any]
    ) -> list[DetectionResult]:
        """Find text needle in haystack screenshot using OCR.

        Args:
            needle: Text string to search for.
            haystack: PIL Image of the screenshot.
            config: Keys used: ``min_confidence``, ``find_all``,
                    ``case_sensitive``.

        Returns:
            List of DetectionResult for each text match found.
        """
        if not isinstance(needle, str):
            return []

        # Convert haystack to PIL if needed
        haystack_image = self._to_pil(haystack)
        if haystack_image is None:
            return []

        min_confidence = config.get("min_confidence", 0.8)
        find_all = config.get("find_all", False)
        case_sensitive = config.get("case_sensitive", False)

        try:
            if find_all:
                text_matches = self._engine.find_all_text(
                    haystack_image,
                    needle,
                    case_sensitive=case_sensitive,
                    confidence=min_confidence,
                )
            else:
                single = self._engine.find_text(
                    haystack_image,
                    needle,
                    case_sensitive=case_sensitive,
                    confidence=min_confidence,
                )
                text_matches = [single] if single else []

            results: list[DetectionResult] = []
            for tm in text_matches:
                region = tm.region
                results.append(
                    DetectionResult(
                        x=region.x,
                        y=region.y,
                        width=region.width,
                        height=region.height,
                        confidence=tm.similarity,
                        backend_name=self.name,
                        label=tm.text,
                        metadata={"ocr_text": tm.text},
                    )
                )

            results.sort(key=lambda r: r.confidence, reverse=True)
            return results

        except Exception:
            logger.exception("OCRBackend: OCR detection failed")
            return []

    def _to_pil(self, image: Any) -> Any:
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            return image

        import numpy as np

        if isinstance(image, np.ndarray):
            import cv2

            if len(image.shape) == 3 and image.shape[2] >= 3:
                rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
                return PILImage.fromarray(rgb)
            return PILImage.fromarray(image)

        return None

    def supports(self, needle_type: str) -> bool:
        return needle_type == "text"

    def estimated_cost_ms(self) -> float:
        return 300.0

    @property
    def name(self) -> str:
        return "ocr"
