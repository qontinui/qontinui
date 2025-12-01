"""PaddleOCR engine implementation."""

import logging
from typing import Any

import numpy as np

from ......model.element.region import Region
from .base_ocr_engine import BaseOCREngine, OCRResult

logger = logging.getLogger(__name__)


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR engine implementation.

    High-performance OCR based on PaddlePaddle framework.
    """

    def __init__(self, use_gpu: bool = False) -> None:
        """Initialize PaddleOCR engine.

        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu
        self._paddle: Any | None = None

    def is_available(self) -> bool:
        """Check if PaddleOCR is available.

        Returns:
            True if paddleocr is installed
        """
        try:
            from paddleocr import PaddleOCR  # noqa: F401

            return True
        except ImportError:
            logger.debug("PaddleOCR not available")
            return False

    def extract_text(
        self,
        image: np.ndarray,
        region: Region,
        language: str,
        confidence_threshold: float,
    ) -> list[OCRResult]:
        """Extract text using PaddleOCR.

        Args:
            image: Preprocessed image
            region: Source region coordinates
            language: Language code (e.g., 'en', 'ch')
            confidence_threshold: Minimum confidence (0.0-1.0)

        Returns:
            List of OCR results
        """
        if not self.is_available():
            logger.debug("PaddleOCR not available")
            return []

        try:
            from paddleocr import PaddleOCR

            # Initialize PaddleOCR lazily (expensive operation)
            if self._paddle is None:
                self._paddle = PaddleOCR(lang=language, use_gpu=self.use_gpu)

            # Perform OCR
            result = self._paddle.ocr(image)

            # Parse results
            ocr_results = []
            if result is None:
                return []

            for line in result:
                if line is None:
                    continue

                for word_info in line:
                    bbox = word_info[0]
                    text = word_info[1][0]
                    confidence = word_info[1][1]

                    if confidence < confidence_threshold:
                        continue

                    # Convert bbox coordinates to region
                    # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    x = min(p[0] for p in bbox) + region.x
                    y = min(p[1] for p in bbox) + region.y
                    w = max(p[0] for p in bbox) - min(p[0] for p in bbox)
                    h = max(p[1] for p in bbox) - min(p[1] for p in bbox)

                    text_region = Region(int(x), int(y), int(w), int(h))

                    result = OCRResult(
                        text=text, confidence=confidence, region=text_region
                    )
                    ocr_results.append(result)

            return ocr_results

        except Exception as e:
            self._log_ocr_error("PaddleOCR", e)
            return []
