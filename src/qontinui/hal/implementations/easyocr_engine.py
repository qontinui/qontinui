"""EasyOCR-based OCR engine implementation."""

import difflib
from typing import Any

import easyocr
import numpy as np
from PIL import Image

from ...logging import get_logger
from ..config import HALConfig
from ..interfaces.ocr_engine import IOCREngine, TextMatch, TextRegion

logger = get_logger(__name__)


class EasyOCREngine(IOCREngine):
    """OCR engine implementation using EasyOCR.

    EasyOCR provides:
    - Support for 80+ languages
    - GPU acceleration when available
    - Better accuracy than traditional OCR
    - Built-in text detection and recognition
    """

    def __init__(self, config: HALConfig | None = None) -> None:
        """Initialize EasyOCR engine.

        Args:
            config: HAL configuration
        """
        self.config = config or HALConfig()
        self._readers: dict[str, Any] = {}  # Cache readers by language
        self._default_languages = ["en"]

        # Check GPU availability
        self.use_gpu = self.config.ocr_gpu_enabled
        try:
            import torch

            if not torch.cuda.is_available():
                self.use_gpu = False
                logger.info("CUDA not available, using CPU for OCR")
        except ImportError:
            self.use_gpu = False
            logger.info("PyTorch not installed, using CPU for OCR")

        logger.info(
            "easyocr_engine_initialized",
            gpu_enabled=self.use_gpu,
            default_languages=self._default_languages,
        )

    def _get_reader(self, languages: list[str]) -> easyocr.Reader:
        """Get or create EasyOCR reader for languages.

        Args:
            languages: List of language codes

        Returns:
            EasyOCR Reader instance
        """
        # Use default if no languages specified
        if not languages:
            languages = self._default_languages

        # Create cache key
        cache_key = ",".join(sorted(languages))

        # Return cached reader if exists
        if cache_key in self._readers:
            return self._readers[cache_key]

        # Create new reader
        try:
            reader = easyocr.Reader(languages, gpu=self.use_gpu, verbose=False)
            self._readers[cache_key] = reader

            logger.debug(
                "easyocr_reader_created", languages=languages, gpu=self.use_gpu
            )

            return reader

        except Exception as e:
            logger.error(f"Failed to create EasyOCR reader: {e}")
            # Fallback to English
            if languages != ["en"]:
                return self._get_reader(["en"])
            raise

    def _pil_to_numpy(self, image: Image.Image) -> np.ndarray[Any, Any]:
        """Convert PIL Image to numpy array.

        Args:
            image: PIL Image

        Returns:
            Numpy array
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.array(image)

    def extract_text(
        self, image: Image.Image, languages: list[str] | None = None
    ) -> str:
        """Extract all text from image.

        Args:
            image: Image to extract text from
            languages: List of language codes (e.g., ['en', 'es'])

        Returns:
            Extracted text as string
        """
        try:
            reader = self._get_reader(languages or self._default_languages)
            np_image = self._pil_to_numpy(image)

            # Perform OCR
            results = reader.readtext(np_image, detail=0)

            # Join all text
            text = " ".join(results) if results else ""

            logger.debug("text_extracted", char_count=len(text), languages=languages)

            return text

        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""

    def get_text_regions(
        self,
        image: Image.Image,
        languages: list[str] | None = None,
        min_confidence: float = 0.5,
    ) -> list[TextRegion]:
        """Get all text regions with bounding boxes.

        Args:
            image: Image to analyze
            languages: List of language codes
            min_confidence: Minimum confidence threshold

        Returns:
            List of TextRegion objects
        """
        try:
            reader = self._get_reader(languages or self._default_languages)
            np_image = self._pil_to_numpy(image)

            # Perform OCR with details
            results = reader.readtext(np_image, detail=1)

            regions = []
            for bbox, text, confidence in results:
                if confidence >= min_confidence:
                    # Convert bbox to x, y, width, height
                    # EasyOCR returns [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x = int(min(x_coords))
                    y = int(min(y_coords))
                    width = int(max(x_coords) - x)
                    height = int(max(y_coords) - y)

                    region = TextRegion(
                        text=text,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        confidence=float(confidence),
                        language=languages[0] if languages else "en",
                    )
                    regions.append(region)

            logger.debug(
                "text_regions_found", count=len(regions), min_confidence=min_confidence
            )

            return regions

        except Exception as e:
            logger.error(f"Text region detection failed: {e}")
            return []

    def find_text(
        self,
        image: Image.Image,
        text: str,
        case_sensitive: bool = False,
        confidence: float = 0.8,
    ) -> TextMatch | None:
        """Find specific text in image.

        Args:
            image: Image to search
            text: Text to find
            case_sensitive: Whether search is case-sensitive
            confidence: Minimum confidence threshold

        Returns:
            TextMatch if found, None otherwise
        """
        try:
            regions = self.get_text_regions(image, min_confidence=confidence)

            target = text if case_sensitive else text.lower()

            for region in regions:
                region_text = region.text if case_sensitive else region.text.lower()

                # Check for exact match
                if target == region_text:
                    return TextMatch(text=region.text, region=region, similarity=1.0)

                # Check for partial match with similarity
                similarity = difflib.SequenceMatcher(None, target, region_text).ratio()
                if similarity >= confidence:
                    return TextMatch(
                        text=region.text, region=region, similarity=similarity
                    )

            logger.debug("text_not_found", target=text, regions_checked=len(regions))

            return None

        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return None

    def find_all_text(
        self,
        image: Image.Image,
        text: str,
        case_sensitive: bool = False,
        confidence: float = 0.8,
    ) -> list[TextMatch]:
        """Find all occurrences of text in image.

        Args:
            image: Image to search
            text: Text to find
            case_sensitive: Whether search is case-sensitive
            confidence: Minimum confidence threshold

        Returns:
            List of TextMatch objects
        """
        try:
            regions = self.get_text_regions(image, min_confidence=confidence)

            target = text if case_sensitive else text.lower()
            matches = []

            for region in regions:
                region_text = region.text if case_sensitive else region.text.lower()

                # Check for exact match
                if target == region_text:
                    matches.append(
                        TextMatch(text=region.text, region=region, similarity=1.0)
                    )
                else:
                    # Check for partial match with similarity
                    similarity = difflib.SequenceMatcher(
                        None, target, region_text
                    ).ratio()
                    if similarity >= confidence:
                        matches.append(
                            TextMatch(
                                text=region.text, region=region, similarity=similarity
                            )
                        )

            # Sort by similarity
            matches.sort(key=lambda m: m.similarity, reverse=True)

            logger.debug("text_matches_found", target=text, count=len(matches))

            return matches

        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

    def extract_text_from_region(
        self,
        image: Image.Image,
        region: tuple[int, int, int, int],
        languages: list[str] | None = None,
    ) -> str:
        """Extract text from specific region.

        Args:
            image: Image containing text
            region: Region bounds (x, y, width, height)
            languages: List of language codes

        Returns:
            Extracted text from region
        """
        try:
            x, y, width, height = region

            # Crop image to region
            cropped = image.crop((x, y, x + width, y + height))

            # Extract text from cropped image
            return self.extract_text(cropped, languages)

        except Exception as e:
            logger.error(f"Region text extraction failed: {e}")
            return ""

    def get_supported_languages(self) -> list[str]:
        """Get list of supported language codes.

        Returns:
            List of language codes
        """
        # EasyOCR supports 80+ languages
        # Common ones listed here
        return [
            "en",  # English
            "es",  # Spanish
            "fr",  # French
            "de",  # German
            "it",  # Italian
            "pt",  # Portuguese
            "ru",  # Russian
            "ja",  # Japanese
            "ko",  # Korean
            "zh",  # Chinese (Simplified)
            "ar",  # Arabic
            "hi",  # Hindi
            "tr",  # Turkish
            "pl",  # Polish
            "nl",  # Dutch
            "sv",  # Swedish
            "da",  # Danish
            "no",  # Norwegian
            "fi",  # Finnish
            "cs",  # Czech
            "hu",  # Hungarian
            "ro",  # Romanian
            "bg",  # Bulgarian
            "uk",  # Ukrainian
            "he",  # Hebrew
            "th",  # Thai
            "vi",  # Vietnamese
            "id",  # Indonesian
            "ms",  # Malay
            "fa",  # Persian/Farsi
        ]

    def preprocess_image(
        self,
        image: Image.Image,
        grayscale: bool = True,
        denoise: bool = True,
        threshold: bool = False,
    ) -> Image.Image:
        """Preprocess image for better OCR results.

        Args:
            image: Input image
            grayscale: Convert to grayscale
            denoise: Apply denoising
            threshold: Apply thresholding

        Returns:
            Preprocessed image
        """
        try:
            import cv2

            # Convert to numpy array
            np_image = self._pil_to_numpy(image)

            # Convert to grayscale if requested
            if grayscale and len(np_image.shape) == 3:
                np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)

            # Apply denoising
            if denoise:
                if len(np_image.shape) == 2:
                    np_image = cv2.fastNlMeansDenoising(np_image, h=10)
                else:
                    np_image = cv2.fastNlMeansDenoisingColored(
                        np_image, h=10, hColor=10
                    )

            # Apply thresholding
            if threshold and len(np_image.shape) == 2:
                _, np_image = cv2.threshold(
                    np_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

            # Convert back to PIL
            if len(np_image.shape) == 2:
                return Image.fromarray(np_image, mode="L")
            else:
                return Image.fromarray(np_image, mode="RGB")

        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image

    def detect_text_orientation(self, image: Image.Image) -> dict[str, Any]:
        """Detect text orientation in image.

        Args:
            image: Image to analyze

        Returns:
            Dictionary with orientation info
        """
        try:
            # EasyOCR doesn't provide direct orientation detection
            # We can infer it by trying different rotations

            np_image = self._pil_to_numpy(image)
            reader = self._get_reader(["en"])

            best_angle = 0
            best_confidence = 0.0

            # Try different angles
            for angle in [0, 90, 180, 270]:
                if angle > 0:
                    # Rotate image
                    rotated = image.rotate(angle, expand=True)
                    test_image = self._pil_to_numpy(rotated)
                else:
                    test_image = np_image

                # Get OCR results
                results = reader.readtext(test_image, detail=1)

                # Calculate average confidence
                if results:
                    avg_confidence = sum(r[2] for r in results) / len(results)
                    if avg_confidence > best_confidence:
                        best_confidence = avg_confidence
                        best_angle = angle

            return {
                "angle": best_angle,
                "confidence": best_confidence,
                "script": "latin",  # Default, could be enhanced
            }

        except Exception as e:
            logger.error(f"Orientation detection failed: {e}")
            return {"angle": 0, "confidence": 0.0, "script": "unknown"}
