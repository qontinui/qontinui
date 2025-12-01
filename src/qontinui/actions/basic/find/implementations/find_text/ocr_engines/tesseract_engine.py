"""Tesseract OCR engine implementation."""

import logging

import numpy as np

from ......model.element.region import Region
from .base_ocr_engine import BaseOCREngine, OCRResult

logger = logging.getLogger(__name__)


class TesseractEngine(BaseOCREngine):
    """Tesseract OCR engine implementation.

    Traditional open-source OCR engine with broad language support.
    """

    def __init__(
        self,
        psm_mode: int = 3,
        oem_mode: int = 3,
        whitelist_chars: str = "",
        blacklist_chars: str = "",
        return_word_boxes: bool = False,
    ) -> None:
        """Initialize Tesseract engine.

        Args:
            psm_mode: Page Segmentation Mode (default: 3 - fully automatic)
            oem_mode: OCR Engine Mode (default: 3 - default)
            whitelist_chars: Characters to recognize (empty = all)
            blacklist_chars: Characters to ignore
            return_word_boxes: Return individual word bounding boxes
        """
        self.psm_mode = psm_mode
        self.oem_mode = oem_mode
        self.whitelist_chars = whitelist_chars
        self.blacklist_chars = blacklist_chars
        self.return_word_boxes = return_word_boxes
        self._tesseract = None

    def is_available(self) -> bool:
        """Check if Tesseract is available.

        Returns:
            True if pytesseract is installed
        """
        try:
            import pytesseract  # noqa: F401

            return True
        except ImportError:
            logger.warning("Tesseract OCR not available")
            return False

    def extract_text(
        self,
        image: np.ndarray,
        region: Region,
        language: str,
        confidence_threshold: float,
    ) -> list[OCRResult]:
        """Extract text using Tesseract.

        Args:
            image: Preprocessed image
            region: Source region coordinates
            language: Language code (e.g., 'eng', 'fra')
            confidence_threshold: Minimum confidence (0.0-1.0)

        Returns:
            List of OCR results
        """
        if not self.is_available():
            logger.error("Tesseract not available")
            return []

        try:
            import pytesseract

            # Build configuration
            config = f"--psm {self.psm_mode} --oem {self.oem_mode}"
            if self.whitelist_chars:
                config += f" -c tessedit_char_whitelist={self.whitelist_chars}"
            if self.blacklist_chars:
                config += f" -c tessedit_char_blacklist={self.blacklist_chars}"

            # Perform OCR with detailed output
            data = pytesseract.image_to_data(
                image, lang=language, config=config, output_type=pytesseract.Output.DICT
            )

            # Parse results
            results = []
            n_boxes = len(data["text"])

            for i in range(n_boxes):
                text = data["text"][i].strip()
                if not text:
                    continue

                # Convert confidence to 0-1 range
                confidence = float(data["conf"][i]) / 100.0
                if confidence < confidence_threshold:
                    continue

                # Calculate absolute region coordinates
                x = region.x + data["left"][i]
                y = region.y + data["top"][i]
                w = data["width"][i]
                h = data["height"][i]
                text_region = Region(x, y, w, h)

                result = OCRResult(text=text, confidence=confidence, region=text_region)
                results.append(result)

            # Merge word-level results if not returning word boxes
            if not self.return_word_boxes and results:
                merged_text = " ".join(r.text for r in results)
                avg_confidence = sum(r.confidence for r in results) / len(results)
                merged_result = OCRResult(
                    text=merged_text,
                    confidence=avg_confidence,
                    region=region,
                    word_boxes=[(r.text, r.region) for r in results],
                )
                return [merged_result]

            return results

        except Exception as e:
            self._log_ocr_error("Tesseract", e)
            return []
