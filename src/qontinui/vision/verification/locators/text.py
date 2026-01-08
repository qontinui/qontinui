"""Text-based locator using OCR.

Provides element detection using OCR (EasyOCR or Tesseract)
with support for exact match, contains, and regex patterns.
"""

import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.assertions import BoundingBox, LocatorType

from qontinui.vision.verification.locators.base import BaseLocator, LocatorMatch

if TYPE_CHECKING:
    from qontinui.vision.verification.config import VisionConfig

logger = logging.getLogger(__name__)


class TextLocator(BaseLocator):
    """Locator using OCR text detection.

    Finds elements by detecting text on screen and matching against
    the specified text pattern.

    Usage:
        # Exact match
        locator = TextLocator("Submit")
        matches = await locator.find_all(screenshot)

        # Contains match
        locator = TextLocator("Submit", exact=False)

        # Regex match
        locator = TextLocator(r"Order #\\d+", regex=True)
    """

    # Lazy-loaded OCR readers
    _easyocr_reader: Any = None
    _pytesseract_imported: bool = False

    def __init__(
        self,
        text: str,
        config: "VisionConfig | None" = None,
        exact: bool = True,
        case_sensitive: bool = True,
        regex: bool = False,
        language: str | None = None,
        ocr_engine: str | None = None,
        confidence_threshold: float | None = None,
        **options: Any,
    ) -> None:
        """Initialize text locator.

        Args:
            text: Text to search for (or regex pattern).
            config: Vision configuration.
            exact: Require exact text match.
            case_sensitive: Case-sensitive matching.
            regex: Treat text as regex pattern.
            language: OCR language code.
            ocr_engine: OCR engine ('easyocr', 'tesseract', 'auto').
            confidence_threshold: Minimum OCR confidence.
            **options: Additional options.
        """
        super().__init__(text, config, **options)

        self._text = text
        self._exact = exact
        self._case_sensitive = case_sensitive
        self._regex = regex
        self._language = language
        self._ocr_engine = ocr_engine
        self._confidence_threshold = confidence_threshold

        # Compiled regex pattern
        self._pattern: re.Pattern[str] | None = None
        if regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            self._pattern = re.compile(text, flags)

    @property
    def locator_type(self) -> LocatorType:
        """Get the locator type."""
        return LocatorType.TEXT

    def _get_language(self) -> str:
        """Get OCR language code.

        Returns:
            Language code.
        """
        if self._language is not None:
            return self._language
        if self._config is not None:
            return self._config.detection.ocr_language
        return "en"

    def _get_ocr_engine(self) -> str:
        """Get OCR engine to use.

        Returns:
            OCR engine name.
        """
        if self._ocr_engine is not None:
            return self._ocr_engine
        if self._config is not None:
            return self._config.detection.ocr_engine
        return "easyocr"

    def _get_confidence_threshold(self) -> float:
        """Get OCR confidence threshold.

        Returns:
            Confidence threshold.
        """
        if self._confidence_threshold is not None:
            return self._confidence_threshold
        if self._config is not None:
            return self._config.detection.ocr_confidence_threshold
        return 0.7

    def _get_easyocr_reader(self) -> Any:
        """Get or create EasyOCR reader.

        Returns:
            EasyOCR Reader instance.
        """
        if TextLocator._easyocr_reader is None:
            try:
                import easyocr

                use_gpu = False
                if self._config is not None:
                    use_gpu = self._config.detection.ocr_gpu

                lang = self._get_language()
                # EasyOCR uses language list
                langs = [lang] if lang != "en" else ["en"]

                TextLocator._easyocr_reader = easyocr.Reader(langs, gpu=use_gpu)
                logger.info(f"Initialized EasyOCR with languages: {langs}")
            except ImportError:
                logger.warning("EasyOCR not available")
                raise

        return TextLocator._easyocr_reader

    async def _find_matches(
        self,
        screenshot: NDArray[np.uint8],
        region: BoundingBox | None = None,
    ) -> list[LocatorMatch]:
        """Find all text matches.

        Args:
            screenshot: Screenshot to search.
            region: Optional region constraint.

        Returns:
            List of matches.
        """
        # Crop to region if specified
        search_area = screenshot
        offset_x, offset_y = 0, 0
        if region is not None:
            search_area = self._crop_to_region(screenshot, region)
            offset_x, offset_y = region.x, region.y

        # Run OCR
        engine = self._get_ocr_engine()
        if engine == "easyocr":
            ocr_results = self._run_easyocr(search_area)
        elif engine == "tesseract":
            ocr_results = self._run_tesseract(search_area)
        else:
            # Auto: try easyocr first, fall back to tesseract
            try:
                ocr_results = self._run_easyocr(search_area)
            except ImportError:
                ocr_results = self._run_tesseract(search_area)

        # Filter by text match
        threshold = self._get_confidence_threshold()
        matches = []

        for result in ocr_results:
            text, confidence, bounds = result

            # Check confidence
            if confidence < threshold:
                continue

            # Check text match
            if self._matches_text(text):
                match = LocatorMatch(
                    bounds=BoundingBox(
                        x=bounds[0] + offset_x,
                        y=bounds[1] + offset_y,
                        width=bounds[2],
                        height=bounds[3],
                    ),
                    confidence=confidence,
                    text=text,
                )
                matches.append(match)

        return matches

    def _matches_text(self, detected_text: str) -> bool:
        """Check if detected text matches the target.

        Args:
            detected_text: Text detected by OCR.

        Returns:
            True if text matches.
        """
        if self._regex and self._pattern is not None:
            return self._pattern.search(detected_text) is not None

        target = self._text
        candidate = detected_text

        if not self._case_sensitive:
            target = target.lower()
            candidate = candidate.lower()

        if self._exact:
            return candidate.strip() == target.strip()
        else:
            return target in candidate

    def _run_easyocr(
        self,
        image: NDArray[np.uint8],
    ) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        """Run EasyOCR on image.

        Args:
            image: Image to process.

        Returns:
            List of (text, confidence, (x, y, w, h)) tuples.
        """
        reader = self._get_easyocr_reader()

        # EasyOCR expects RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = image[:, :, ::-1]  # BGR to RGB
        else:
            rgb_image = image

        results = reader.readtext(rgb_image)

        ocr_results = []
        for bbox, text, confidence in results:
            # EasyOCR returns [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            x1 = int(min(p[0] for p in bbox))
            y1 = int(min(p[1] for p in bbox))
            x2 = int(max(p[0] for p in bbox))
            y2 = int(max(p[1] for p in bbox))

            ocr_results.append(
                (
                    text,
                    float(confidence),
                    (x1, y1, x2 - x1, y2 - y1),
                )
            )

        return ocr_results

    def _run_tesseract(
        self,
        image: NDArray[np.uint8],
    ) -> list[tuple[str, float, tuple[int, int, int, int]]]:
        """Run Tesseract OCR on image.

        Args:
            image: Image to process.

        Returns:
            List of (text, confidence, (x, y, w, h)) tuples.
        """
        try:
            import pytesseract
        except ImportError:
            logger.error("pytesseract not available")
            raise

        # Get bounding box data
        lang = self._get_language()
        data = pytesseract.image_to_data(
            image,
            lang=lang,
            output_type=pytesseract.Output.DICT,
        )

        ocr_results = []
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = data["text"][i].strip()
            if not text:
                continue

            confidence = float(data["conf"][i]) / 100.0
            if confidence < 0:
                continue

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            ocr_results.append((text, confidence, (x, y, w, h)))

        return ocr_results


__all__ = ["TextLocator"]
