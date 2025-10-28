"""EasyOCR engine implementation."""

import logging
from typing import Any

import numpy as np

from ......model.element.region import Region
from .base_ocr_engine import BaseOCREngine, OCRResult

logger = logging.getLogger(__name__)


class EasyOCREngine(BaseOCREngine):
    """EasyOCR engine implementation.

    Modern neural network-based OCR with support for 80+ languages.
    """

    def __init__(self) -> None:
        """Initialize EasyOCR engine."""
        self._reader: Any | None = None

    def is_available(self) -> bool:
        """Check if EasyOCR is available.

        Returns:
            True if easyocr is installed
        """
        try:
            import easyocr  # noqa: F401

            return True
        except ImportError:
            logger.debug("EasyOCR not available")
            return False

    def extract_text(
        self, image: np.ndarray, region: Region, language: str, confidence_threshold: float
    ) -> list[OCRResult]:
        """Extract text using EasyOCR.

        Args:
            image: Preprocessed image
            region: Source region coordinates
            language: Language code (e.g., 'en', 'fr')
            confidence_threshold: Minimum confidence (0.0-1.0)

        Returns:
            List of OCR results
        """
        if not self.is_available():
            logger.debug("EasyOCR not available")
            return []

        try:
            import easyocr

            # Initialize reader lazily (expensive operation)
            if self._reader is None:
                self._reader = easyocr.Reader([language])

            # Perform OCR
            results = self._reader.readtext(image)

            # Parse results
            ocr_results = []
            for bbox, text, confidence in results:
                if confidence < confidence_threshold:
                    continue

                # Convert bbox coordinates to region
                # bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                x = min(bbox[0][0], bbox[3][0]) + region.x
                y = min(bbox[0][1], bbox[1][1]) + region.y
                w = max(bbox[1][0], bbox[2][0]) - min(bbox[0][0], bbox[3][0])
                h = max(bbox[2][1], bbox[3][1]) - min(bbox[0][1], bbox[1][1])

                text_region = Region(int(x), int(y), int(w), int(h))

                result = OCRResult(text=text, confidence=confidence, region=text_region)
                ocr_results.append(result)

            return ocr_results

        except Exception as e:
            self._log_ocr_error("EasyOCR", e)
            return []
