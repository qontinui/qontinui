"""Find text implementation - ported from Qontinui framework.

OCR-based text finding using various engines.
"""

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import numpy as np

from .....model.element.location import Location
from .....model.element.region import Region
from .....model.match.match import Match
from ....object_collection import ObjectCollection
from ..options.text_find_options import OCREngine, TextFindOptions, TextMatchType, TextPreprocessing

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


@dataclass
class FindText:
    """OCR-based text finding implementation.

    Port of FindText from Qontinui framework class.

    Implements text detection and recognition using various OCR engines
    with support for preprocessing, multiple match types, and fuzzy matching.
    """

    # OCR engine availability
    _tesseract_available: bool = field(default=False, init=False)
    _easyocr_available: bool = field(default=False, init=False)
    _paddleocr_available: bool = field(default=False, init=False)

    # OCR engine instances (lazy loaded)
    _tesseract: Any | None = field(default=None, init=False)
    _easyocr_reader: Any | None = field(default=None, init=False)
    _paddleocr: Any | None = field(default=None, init=False)

    # Cache for OCR results
    _ocr_cache: dict[str, list[OCRResult]] = field(default_factory=dict)

    def __post_init__(self):
        """Check for available OCR engines."""
        self._check_ocr_availability()

    def _check_ocr_availability(self):
        """Check which OCR engines are available."""
        # Check Tesseract
        try:
            import pytesseract  # noqa: F401

            self._tesseract_available = True
            logger.debug("Tesseract OCR available")
        except ImportError:
            logger.warning("Tesseract OCR not available")

        # Check EasyOCR
        try:
            import easyocr  # noqa: F401

            self._easyocr_available = True
            logger.debug("EasyOCR available")
        except ImportError:
            logger.debug("EasyOCR not available")

        # Check PaddleOCR
        try:
            from paddleocr import PaddleOCR  # noqa: F401

            self._paddleocr_available = True
            logger.debug("PaddleOCR available")
        except ImportError:
            logger.debug("PaddleOCR not available")

    def find(self, object_collection: ObjectCollection, options: TextFindOptions) -> list[Match]:
        """Find text using OCR.

        Args:
            object_collection: Objects containing text to find
            options: Text find configuration

        Returns:
            List of matches found
        """
        matches = []

        # Get search texts
        search_texts = self._get_search_texts(object_collection, options)
        if not search_texts:
            logger.warning("No text to search for")
            return []

        # Get regions to search
        search_regions = self._get_search_regions(options)

        # Perform OCR on regions
        ocr_results = []
        for region in search_regions:
            # Check cache
            cache_key = self._get_cache_key(region, options)
            if options.use_cache and cache_key in self._ocr_cache:
                ocr_results.extend(self._ocr_cache[cache_key])
            else:
                # Perform OCR
                results = self._perform_ocr(region, options)
                ocr_results.extend(results)

                # Update cache
                if options.use_cache:
                    self._ocr_cache[cache_key] = results

        # Match text against OCR results
        for search_text in search_texts:
            text_matches = self._match_text(search_text, ocr_results, options)
            matches.extend(text_matches)

            # Early termination for FIRST search
            if options.search_type.name == "FIRST" and matches:
                break

        return matches

    def _get_search_texts(
        self, object_collection: ObjectCollection, options: TextFindOptions
    ) -> list[str]:
        """Extract search texts from collection and options.

        Args:
            object_collection: Object collection
            options: Text options

        Returns:
            List of texts to search for
        """
        texts = []

        # From options
        if options.search_text:
            texts.append(options.search_text)
        texts.extend(options.search_texts)

        # From object collection
        for state_string in object_collection.state_strings:
            if hasattr(state_string, "text"):
                texts.append(state_string.text)

        return texts

    def _get_search_regions(self, options: TextFindOptions) -> list[Region]:
        """Get regions to perform OCR on.

        Args:
            options: Text options

        Returns:
            List of regions to search
        """
        regions = []

        # Use specified text regions
        if options.text_regions:
            regions.extend(options.text_regions)
        # Use general search regions
        elif options.search_regions:
            regions.extend(options.search_regions)
        else:
            # Use full screen
            # This would get actual screen dimensions
            regions.append(Region(0, 0, 1920, 1080))  # Placeholder

        return regions

    def _perform_ocr(self, region: Region, options: TextFindOptions) -> list[OCRResult]:
        """Perform OCR on a region.

        Args:
            region: Region to perform OCR on
            options: Text options

        Returns:
            List of OCR results
        """
        # Capture region image
        image = self._capture_region(region)
        if image is None:
            logger.warning(f"Could not capture region: {region}")
            return []

        # Preprocess image
        processed_image = self._preprocess_image(image, options)

        # Perform OCR based on engine
        if options.ocr_engine == OCREngine.TESSERACT:
            return self._ocr_tesseract(processed_image, region, options)
        elif options.ocr_engine == OCREngine.EASYOCR:
            return self._ocr_easyocr(processed_image, region, options)
        elif options.ocr_engine == OCREngine.PADDLEOCR:
            return self._ocr_paddleocr(processed_image, region, options)
        else:
            logger.warning(f"OCR engine not supported: {options.ocr_engine}")
            return []

    def _ocr_tesseract(
        self, image: np.ndarray[Any, Any], region: Region, options: TextFindOptions
    ) -> list[OCRResult]:
        """Perform OCR using Tesseract.

        Args:
            image: Preprocessed image
            region: Source region
            options: Text options

        Returns:
            List of OCR results
        """
        if not self._tesseract_available:
            logger.error("Tesseract not available")
            return []

        try:
            import pytesseract

            # Configure Tesseract
            config = f"--psm {options.psm_mode} --oem {options.oem_mode}"
            if options.whitelist_chars:
                config += f" -c tessedit_char_whitelist={options.whitelist_chars}"
            if options.blacklist_chars:
                config += f" -c tessedit_char_blacklist={options.blacklist_chars}"

            # Perform OCR with confidence scores
            data = pytesseract.image_to_data(
                image, lang=options.language, config=config, output_type=pytesseract.Output.DICT
            )

            # Parse results
            results = []
            n_boxes = len(data["text"])

            for i in range(n_boxes):
                text = data["text"][i].strip()
                if not text:
                    continue

                confidence = float(data["conf"][i]) / 100.0
                if confidence < options.confidence_threshold:
                    continue

                # Create region for this text
                x = region.x + data["left"][i]
                y = region.y + data["top"][i]
                w = data["width"][i]
                h = data["height"][i]
                text_region = Region(x, y, w, h)

                result = OCRResult(text=text, confidence=confidence, region=text_region)
                results.append(result)

            # Merge word-level results if not returning word boxes
            if not options.return_word_boxes and results:
                merged_text = " ".join(r.text for r in results)
                avg_confidence = sum(r.confidence for r in results) / len(results)
                merged_result = OCRResult(
                    text=merged_text,
                    confidence=avg_confidence,
                    region=region,
                    word_boxes=[(r.text, r.region) for r in results] if results else None,
                )
                return [merged_result]

            return results

        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return []

    def _ocr_easyocr(
        self, image: np.ndarray[Any, Any], region: Region, options: TextFindOptions
    ) -> list[OCRResult]:
        """Perform OCR using EasyOCR.

        Args:
            image: Preprocessed image
            region: Source region
            options: Text options

        Returns:
            List of OCR results
        """
        if not self._easyocr_available:
            logger.debug("EasyOCR not available, falling back to Tesseract")
            return self._ocr_tesseract(image, region, options)

        try:
            import easyocr

            # Initialize reader if needed
            if self._easyocr_reader is None:
                self._easyocr_reader = easyocr.Reader([options.language])

            # Perform OCR
            results = self._easyocr_reader.readtext(image)

            # Parse results
            ocr_results = []
            for bbox, text, confidence in results:
                if confidence < options.confidence_threshold:
                    continue

                # Convert bbox to region
                x = min(bbox[0][0], bbox[3][0]) + region.x
                y = min(bbox[0][1], bbox[1][1]) + region.y
                w = max(bbox[1][0], bbox[2][0]) - min(bbox[0][0], bbox[3][0])
                h = max(bbox[2][1], bbox[3][1]) - min(bbox[0][1], bbox[1][1])

                text_region = Region(int(x), int(y), int(w), int(h))

                result = OCRResult(text=text, confidence=confidence, region=text_region)
                ocr_results.append(result)

            return ocr_results

        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return []

    def _ocr_paddleocr(
        self, image: np.ndarray[Any, Any], region: Region, options: TextFindOptions
    ) -> list[OCRResult]:
        """Perform OCR using PaddleOCR.

        Args:
            image: Preprocessed image
            region: Source region
            options: Text options

        Returns:
            List of OCR results
        """
        if not self._paddleocr_available:
            logger.debug("PaddleOCR not available, falling back to Tesseract")
            return self._ocr_tesseract(image, region, options)

        try:
            from paddleocr import PaddleOCR

            # Initialize PaddleOCR if needed
            if self._paddleocr is None:
                self._paddleocr = PaddleOCR(lang=options.language, use_gpu=False)

            # Perform OCR
            result = self._paddleocr.ocr(image)

            # Parse results
            ocr_results = []
            for line in result:
                for word_info in line:
                    bbox = word_info[0]
                    text = word_info[1][0]
                    confidence = word_info[1][1]

                    if confidence < options.confidence_threshold:
                        continue

                    # Convert bbox to region
                    x = min(p[0] for p in bbox) + region.x
                    y = min(p[1] for p in bbox) + region.y
                    w = max(p[0] for p in bbox) - min(p[0] for p in bbox)
                    h = max(p[1] for p in bbox) - min(p[1] for p in bbox)

                    text_region = Region(int(x), int(y), int(w), int(h))

                    result = OCRResult(text=text, confidence=confidence, region=text_region)
                    ocr_results.append(result)

            return ocr_results

        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
            return []

    def _match_text(
        self, search_text: str, ocr_results: list[OCRResult], options: TextFindOptions
    ) -> list[Match]:
        """Match search text against OCR results.

        Args:
            search_text: Text to search for
            ocr_results: OCR results to search in
            options: Text options

        Returns:
            List of matches
        """
        matches = []

        for ocr_result in ocr_results:
            similarity = self._calculate_text_similarity(search_text, ocr_result.text, options)

            if similarity >= options.similarity:
                match = Match(
                    target=Location(region=ocr_result.region),
                    score=similarity,
                    ocr_text=ocr_result.text if options.return_text_content else "",
                )
                matches.append(match)

        return matches

    def _calculate_text_similarity(
        self, search_text: str, found_text: str, options: TextFindOptions
    ) -> float:
        """Calculate similarity between search and found text.

        Args:
            search_text: Text to search for
            found_text: Text found by OCR
            options: Text options

        Returns:
            Similarity score (0.0-1.0)
        """
        # Preprocessing
        if not options.case_sensitive:
            search_text = search_text.lower()
            found_text = found_text.lower()

        if options.ignore_whitespace:
            search_text = "".join(search_text.split())
            found_text = "".join(found_text.split())

        # Match based on type
        if options.match_type == TextMatchType.EXACT:
            return 1.0 if search_text == found_text else 0.0

        elif options.match_type == TextMatchType.CONTAINS:
            return 1.0 if search_text in found_text else 0.0

        elif options.match_type == TextMatchType.STARTS_WITH:
            return 1.0 if found_text.startswith(search_text) else 0.0

        elif options.match_type == TextMatchType.ENDS_WITH:
            return 1.0 if found_text.endswith(search_text) else 0.0

        elif options.match_type == TextMatchType.REGEX:
            try:
                pattern = re.compile(search_text)
                return 1.0 if pattern.search(found_text) else 0.0
            except re.error:
                logger.warning(f"Invalid regex: {search_text}")
                return 0.0

        elif options.match_type == TextMatchType.FUZZY:
            # Use sequence matcher for fuzzy matching
            return SequenceMatcher(None, search_text, found_text).ratio()

        return 0.0

    def _preprocess_image(
        self, image: np.ndarray[Any, Any], options: TextFindOptions
    ) -> np.ndarray[Any, Any]:
        """Preprocess image for OCR.

        Args:
            image: Input image
            options: Text options

        Returns:
            Preprocessed image
        """
        import cv2

        result = image.copy()

        for preprocessing in options.preprocessing:
            if preprocessing == TextPreprocessing.GRAYSCALE:
                if len(result.shape) == 3:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

            elif preprocessing == TextPreprocessing.BINARIZE:
                if len(result.shape) == 3:
                    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                _, result = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            elif preprocessing == TextPreprocessing.DENOISE:
                if len(result.shape) == 2:
                    result = cv2.medianBlur(result, 3)
                else:
                    result = cv2.bilateralFilter(result, 9, 75, 75)

            elif preprocessing == TextPreprocessing.ENHANCE:
                # Enhance contrast
                if len(result.shape) == 2:
                    result = cv2.equalizeHist(result)
                else:
                    # Convert to LAB and enhance L channel
                    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
                    l_channel, a, b = cv2.split(lab)
                    l_channel = cv2.equalizeHist(l_channel)
                    result = cv2.cvtColor(cv2.merge([l_channel, a, b]), cv2.COLOR_LAB2BGR)

        # Scale image if needed
        if options.scale_factor != 1.0:
            width = int(result.shape[1] * options.scale_factor)
            height = int(result.shape[0] * options.scale_factor)
            result = cv2.resize(result, (width, height), interpolation=cv2.INTER_CUBIC)

        return result

    def _capture_region(self, region: Region) -> np.ndarray[Any, Any] | None:
        """Capture image from region.

        Args:
            region: Region to capture

        Returns:
            Captured image or None
        """
        # This would capture the actual region
        # For now, return None as placeholder
        logger.debug(f"Capturing region for OCR: {region}")
        return None

    def _get_cache_key(self, region: Region, options: TextFindOptions) -> str:
        """Generate cache key for OCR results.

        Args:
            region: Region being processed
            options: Text options

        Returns:
            Cache key
        """
        key_parts = [
            str(region),
            options.ocr_engine.name,
            options.language,
            str(options.scale_factor),
            str(options.preprocessing),
        ]
        return "_".join(key_parts)

    def clear_cache(self):
        """Clear OCR cache."""
        self._ocr_cache.clear()
        logger.debug("OCR cache cleared")
