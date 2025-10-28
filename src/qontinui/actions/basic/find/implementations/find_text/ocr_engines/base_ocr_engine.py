"""Base OCR engine interface.

Abstract base class for OCR engine implementations.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ......model.element.region import Region

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result from OCR operation.

    Contains recognized text and metadata.
    """

    text: str
    confidence: float
    region: Region
    word_boxes: list[tuple[str, Region]] | None = None


class BaseOCREngine(ABC):
    """Abstract base class for OCR engines.

    Implements the Strategy pattern for different OCR backends.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this OCR engine is available.

        Returns:
            True if engine dependencies are installed
        """
        pass

    @abstractmethod
    def extract_text(
        self, image: np.ndarray, region: Region, language: str, confidence_threshold: float
    ) -> list[OCRResult]:
        """Extract text from an image.

        Args:
            image: Preprocessed image to perform OCR on
            region: Source region coordinates
            language: Language code for OCR
            confidence_threshold: Minimum confidence for results

        Returns:
            List of OCR results with text, confidence, and regions
        """
        pass

    def _log_ocr_error(self, engine_name: str, error: Exception) -> None:
        """Log OCR error in a consistent format.

        Args:
            engine_name: Name of the OCR engine
            error: Exception that occurred
        """
        logger.error(f"{engine_name} OCR failed: {error}")
