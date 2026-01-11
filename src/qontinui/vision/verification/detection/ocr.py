"""OCR detection engine with environment integration.

Provides text detection using EasyOCR or Tesseract with
typography hints from the discovered GUI environment for
improved accuracy and performance.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray
from qontinui_schemas.testing.assertions import BoundingBox
from qontinui_schemas.testing.environment import GUIEnvironment, Typography


class BoundsLike(Protocol):
    """Protocol for objects with bounding box properties."""

    x: int
    y: int
    width: int
    height: int

if TYPE_CHECKING:
    from qontinui.vision.verification.config import VisionConfig

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result from OCR text detection."""

    text: str
    confidence: float
    bounds: BoundingBox
    line_number: int = 0
    word_index: int = 0
    font_size_estimate: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        """Get center point of text region."""
        return (
            self.bounds.x + self.bounds.width // 2,
            self.bounds.y + self.bounds.height // 2,
        )


class OCREngine:
    """OCR engine with environment-aware text detection.

    Integrates with the discovered GUI environment to:
    - Use typography hints for better recognition
    - Filter results by expected text sizes
    - Optimize preprocessing based on theme
    - Cache OCR readers for performance

    Usage:
        engine = OCREngine(config, environment)
        results = await engine.detect_text(screenshot)
        results = await engine.find_text("Submit", screenshot)
    """

    # Class-level reader cache
    _easyocr_readers: dict[str, Any] = {}
    _tesseract_available: bool | None = None

    def __init__(
        self,
        config: "VisionConfig | None" = None,
        environment: GUIEnvironment | None = None,
    ) -> None:
        """Initialize OCR engine.

        Args:
            config: Vision configuration.
            environment: GUI environment with typography hints.
        """
        self._config = config
        self._environment = environment
        self._typography: Typography | None = None

        if environment is not None:
            self._typography = environment.typography

    def set_environment(self, environment: GUIEnvironment) -> None:
        """Update the environment.

        Args:
            environment: New GUI environment.
        """
        self._environment = environment
        self._typography = environment.typography

    def _get_engine_name(self) -> str:
        """Get OCR engine name from config.

        Returns:
            Engine name: 'easyocr', 'tesseract', or 'auto'.
        """
        if self._config is not None:
            return self._config.detection.ocr_engine
        return "easyocr"

    def _get_language(self) -> str:
        """Get OCR language from config or environment.

        Returns:
            Language code.
        """
        # Check environment first
        if self._typography is not None and self._typography.languages_detected:
            return str(self._typography.languages_detected[0])

        if self._config is not None:
            return self._config.detection.ocr_language

        return "en"

    def _get_confidence_threshold(self) -> float:
        """Get confidence threshold from config.

        Returns:
            Confidence threshold (0.0-1.0).
        """
        if self._config is not None:
            return self._config.detection.ocr_confidence_threshold
        return 0.7

    def _get_easyocr_reader(self, language: str) -> Any:
        """Get or create EasyOCR reader for language.

        Args:
            language: Language code.

        Returns:
            EasyOCR Reader instance.
        """
        if language not in OCREngine._easyocr_readers:
            try:
                import easyocr

                use_gpu = False
                if self._config is not None:
                    use_gpu = self._config.detection.ocr_gpu

                # Map common language codes
                lang_map = {"en": ["en"], "eng": ["en"], "ja": ["ja"], "ko": ["ko"]}
                langs = lang_map.get(language, [language])

                reader = easyocr.Reader(langs, gpu=use_gpu)
                OCREngine._easyocr_readers[language] = reader
                logger.info(f"Initialized EasyOCR for language: {language}")

            except ImportError:
                logger.error("EasyOCR not installed")
                raise

        return OCREngine._easyocr_readers[language]

    def _check_tesseract(self) -> bool:
        """Check if Tesseract is available.

        Returns:
            True if Tesseract is available.
        """
        if OCREngine._tesseract_available is None:
            try:
                import pytesseract

                pytesseract.get_tesseract_version()
                OCREngine._tesseract_available = True
            except Exception:
                OCREngine._tesseract_available = False

        return OCREngine._tesseract_available

    def _preprocess_image(
        self,
        image: NDArray[np.uint8],
        region: BoundingBox | None = None,
    ) -> NDArray[np.uint8]:
        """Preprocess image for OCR.

        Applies environment-aware preprocessing:
        - Crops to region if specified
        - Adjusts for dark/light theme
        - Applies sharpening for small text

        Args:
            image: Input image (BGR).
            region: Optional region to crop to.

        Returns:
            Preprocessed image.
        """
        # Crop to region
        if region is not None:
            image = image[
                region.y : region.y + region.height,
                region.x : region.x + region.width,
            ].copy()

        # Check if we have environment info
        if self._environment is not None:
            theme = self._environment.colors.theme_type.value

            # For dark themes, invert for better OCR
            if theme == "dark":
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Invert
                gray = cv2.bitwise_not(gray)
                # Convert back to BGR for consistency
                image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.uint8)

        # Apply slight sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        image = cv2.filter2D(image, -1, kernel).astype(np.uint8)

        return image

    async def detect_text(
        self,
        image: NDArray[np.uint8],
        region: BoundingBox | None = None,
        min_confidence: float | None = None,
    ) -> list[OCRResult]:
        """Detect all text in image.

        Args:
            image: Image to process (BGR).
            region: Optional region to limit search.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of OCR results.
        """
        if min_confidence is None:
            min_confidence = self._get_confidence_threshold()

        # Preprocess
        processed = self._preprocess_image(image, region)

        # Get offset for coordinate adjustment
        offset_x = region.x if region else 0
        offset_y = region.y if region else 0

        # Run OCR
        engine = self._get_engine_name()

        if engine == "easyocr" or (engine == "auto" and True):
            results = self._run_easyocr(processed, min_confidence)
        elif engine == "tesseract" or (engine == "auto" and self._check_tesseract()):
            results = self._run_tesseract(processed, min_confidence)
        else:
            logger.warning("No OCR engine available")
            return []

        # Adjust coordinates for region offset
        for result in results:
            result.bounds = BoundingBox(
                x=result.bounds.x + offset_x,
                y=result.bounds.y + offset_y,
                width=result.bounds.width,
                height=result.bounds.height,
            )

        # Apply typography-based filtering if available
        if self._typography is not None:
            results = self._filter_by_typography(results)

        return results

    def _run_easyocr(
        self,
        image: NDArray[np.uint8],
        min_confidence: float,
    ) -> list[OCRResult]:
        """Run EasyOCR on image.

        Args:
            image: Preprocessed image.
            min_confidence: Minimum confidence.

        Returns:
            List of OCR results.
        """
        language = self._get_language()
        reader = self._get_easyocr_reader(language)

        # EasyOCR expects RGB
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image

        raw_results = reader.readtext(rgb_image)

        results = []
        for i, (bbox, text, confidence) in enumerate(raw_results):
            if confidence < min_confidence:
                continue

            # EasyOCR bbox format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            x1 = int(min(p[0] for p in bbox))
            y1 = int(min(p[1] for p in bbox))
            x2 = int(max(p[0] for p in bbox))
            y2 = int(max(p[1] for p in bbox))

            # Estimate font size from height
            height = y2 - y1
            font_size = self._estimate_font_size(height)

            results.append(
                OCRResult(
                    text=text,
                    confidence=float(confidence),
                    bounds=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
                    word_index=i,
                    font_size_estimate=font_size,
                    metadata={"engine": "easyocr"},
                )
            )

        return results

    def _run_tesseract(
        self,
        image: NDArray[np.uint8],
        min_confidence: float,
    ) -> list[OCRResult]:
        """Run Tesseract on image.

        Args:
            image: Preprocessed image.
            min_confidence: Minimum confidence.

        Returns:
            List of OCR results.
        """
        try:
            import pytesseract
        except ImportError:
            logger.error("pytesseract not installed")
            return []

        language = self._get_language()

        # Map language codes for Tesseract
        lang_map = {"en": "eng", "ja": "jpn", "ko": "kor", "zh": "chi_sim"}
        tess_lang = lang_map.get(language, language)

        data = pytesseract.image_to_data(
            image,
            lang=tess_lang,
            output_type=pytesseract.Output.DICT,
        )

        results = []
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = data["text"][i].strip()
            if not text:
                continue

            confidence = float(data["conf"][i]) / 100.0
            if confidence < min_confidence:
                continue

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            font_size = self._estimate_font_size(h)

            results.append(
                OCRResult(
                    text=text,
                    confidence=confidence,
                    bounds=BoundingBox(x=x, y=y, width=w, height=h),
                    line_number=int(data["line_num"][i]),
                    word_index=int(data["word_num"][i]),
                    font_size_estimate=font_size,
                    metadata={"engine": "tesseract", "block_num": data["block_num"][i]},
                )
            )

        return results

    def _estimate_font_size(self, height_px: int) -> int:
        """Estimate font size from pixel height.

        Uses typography hints if available for better accuracy.

        Args:
            height_px: Text height in pixels.

        Returns:
            Estimated font size in pixels.
        """
        # Rough conversion: font size ~= height * 0.75
        base_estimate = int(height_px * 0.75)

        if self._typography is not None:
            # Find closest known text size
            sizes = self._typography.text_sizes
            known_sizes = [
                s
                for s in [
                    sizes.heading_large,
                    sizes.heading,
                    sizes.heading_small,
                    sizes.body,
                    sizes.small,
                    sizes.tiny,
                ]
                if s is not None
            ]

            if known_sizes:
                # Find closest match
                closest = min(known_sizes, key=lambda s: abs(s - base_estimate))
                # If close enough, use the known size
                if abs(closest - base_estimate) < 4:
                    return int(closest)

        return base_estimate

    def _filter_by_typography(self, results: list[OCRResult]) -> list[OCRResult]:
        """Filter results using typography hints.

        Removes results that don't match expected text characteristics.

        Args:
            results: Raw OCR results.

        Returns:
            Filtered results.
        """
        if self._typography is None:
            return results

        # Get known text regions
        text_regions = self._typography.common_text_regions

        filtered = []
        for result in results:
            # Check if result is in a known text region
            in_known_region = False
            for region in text_regions:
                if self._bounds_overlap(result.bounds, region.bounds):
                    in_known_region = True
                    # Boost confidence for text in known regions
                    result.confidence = min(1.0, result.confidence * 1.1)
                    break

            # Keep results in known regions or with high confidence
            if in_known_region or result.confidence >= 0.8:
                filtered.append(result)

        return filtered

    def _bounds_overlap(self, a: BoundsLike, b: BoundsLike) -> bool:
        """Check if two bounding boxes overlap.

        Args:
            a: First bounding box.
            b: Second bounding box.

        Returns:
            True if boxes overlap.
        """
        return not (
            a.x + a.width < b.x
            or b.x + b.width < a.x
            or a.y + a.height < b.y
            or b.y + b.height < a.y
        )

    async def find_text(
        self,
        target_text: str,
        image: NDArray[np.uint8],
        region: BoundingBox | None = None,
        exact: bool = True,
        case_sensitive: bool = True,
        regex: bool = False,
    ) -> list[OCRResult]:
        """Find specific text in image.

        Args:
            target_text: Text to search for.
            image: Image to search.
            region: Optional region to limit search.
            exact: Require exact match vs contains.
            case_sensitive: Case-sensitive matching.
            regex: Treat target_text as regex pattern.

        Returns:
            List of matching OCR results.
        """
        # Get all text
        all_results = await self.detect_text(image, region)

        # Filter for matches
        matches = []
        pattern = None

        if regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(target_text, flags)

        for result in all_results:
            detected = result.text
            target = target_text

            if not case_sensitive:
                detected = detected.lower()
                target = target.lower()

            is_match = False
            if regex and pattern is not None:
                is_match = pattern.search(result.text) is not None
            elif exact:
                is_match = detected.strip() == target.strip()
            else:
                is_match = target in detected

            if is_match:
                matches.append(result)

        return matches

    async def extract_text_from_region(
        self,
        image: NDArray[np.uint8],
        region: BoundingBox,
        join_lines: bool = True,
    ) -> str:
        """Extract all text from a region.

        Args:
            image: Image to process.
            region: Region to extract text from.
            join_lines: Join text lines with spaces.

        Returns:
            Extracted text.
        """
        results = await self.detect_text(image, region)

        if not results:
            return ""

        # Sort by position (top to bottom, left to right)
        results.sort(key=lambda r: (r.bounds.y, r.bounds.x))

        if join_lines:
            return " ".join(r.text for r in results)
        else:
            # Group by line
            lines: list[list[OCRResult]] = []
            current_line: list[OCRResult] = []
            last_y = -100

            for result in results:
                if abs(result.bounds.y - last_y) > 10:
                    if current_line:
                        lines.append(current_line)
                    current_line = [result]
                else:
                    current_line.append(result)
                last_y = result.bounds.y

            if current_line:
                lines.append(current_line)

            return "\n".join(" ".join(r.text for r in line) for line in lines)


# Global engine instance
_ocr_engine: OCREngine | None = None


def get_ocr_engine(
    config: "VisionConfig | None" = None,
    environment: GUIEnvironment | None = None,
) -> OCREngine:
    """Get the global OCR engine instance.

    Args:
        config: Optional vision configuration.
        environment: Optional GUI environment.

    Returns:
        OCREngine instance.
    """
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = OCREngine(config=config, environment=environment)
    elif environment is not None:
        _ocr_engine.set_environment(environment)
    return _ocr_engine


__all__ = [
    "OCREngine",
    "OCRResult",
    "get_ocr_engine",
]
