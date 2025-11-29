"""Find text orchestrator.

Main coordinator that brings together OCR engines and text matchers.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ......model.element.location import Location
from ......model.element.region import Region
from ......model.match.match import Match
from .....object_collection import ObjectCollection
from ...options.text_find_options import OCREngine, TextFindOptions, TextMatchType
from .image_preprocessor import ImagePreprocessor
from .ocr_engines import (
    BaseOCREngine,
    EasyOCREngine,
    OCRResult,
    PaddleOCREngine,
    TesseractEngine,
)
from .text_matchers import (
    BaseMatcher,
    ContainsMatcher,
    EndsWithMatcher,
    ExactMatcher,
    FuzzyMatcher,
    RegexMatcher,
    StartsWithMatcher,
)

logger = logging.getLogger(__name__)


@dataclass
class FindTextOrchestrator:
    """Orchestrates text finding using OCR and matching strategies.

    Coordinates between OCR engines, text matchers, and preprocessing
    to find text in images.
    """

    # OCR cache
    _ocr_cache: dict[str, list[OCRResult]] = field(default_factory=dict)

    def find(self, object_collection: ObjectCollection, options: TextFindOptions) -> list[Match]:
        """Find text using OCR and matching.

        Args:
            object_collection: Objects containing text to find
            options: Text find configuration

        Returns:
            List of matches found
        """
        # Extract search texts
        search_texts = self._get_search_texts(object_collection, options)
        if not search_texts:
            logger.warning("No text to search for")
            return []

        # Get search regions
        search_regions = self._get_search_regions(options)

        # Create OCR engine
        ocr_engine = self._create_ocr_engine(options)
        if ocr_engine is None:
            logger.error("No OCR engine available")
            return []

        # Create text matcher
        text_matcher = self._create_text_matcher(options)

        # Create preprocessor
        preprocessor = ImagePreprocessor(options.preprocessing, options.scale_factor)

        # Perform OCR on all regions
        ocr_results = []
        for region in search_regions:
            # Check cache
            cache_key = self._get_cache_key(region, options)
            if options.use_cache and cache_key in self._ocr_cache:
                ocr_results.extend(self._ocr_cache[cache_key])
            else:
                # Capture and preprocess region
                image = self._capture_region(region)
                if image is None:
                    continue

                processed_image = preprocessor.preprocess(image)

                # Perform OCR
                results = ocr_engine.extract_text(
                    processed_image, region, options.language, options.confidence_threshold
                )
                ocr_results.extend(results)

                # Update cache
                if options.use_cache:
                    self._ocr_cache[cache_key] = results

        # Match text against OCR results
        matches = []
        for search_text in search_texts:
            text_matches = self._match_text(search_text, ocr_results, text_matcher, options)
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
            # Use full screen (placeholder)
            regions.append(Region(0, 0, 1920, 1080))

        return regions

    def _create_ocr_engine(self, options: TextFindOptions) -> BaseOCREngine | None:
        """Create OCR engine based on options.

        Args:
            options: Text options

        Returns:
            OCR engine instance or None
        """
        if options.ocr_engine == OCREngine.TESSERACT:
            engine = TesseractEngine(
                psm_mode=options.psm_mode,
                oem_mode=options.oem_mode,
                whitelist_chars=options.whitelist_chars,
                blacklist_chars=options.blacklist_chars,
                return_word_boxes=options.return_word_boxes,
            )
            if engine.is_available():
                return engine

        elif options.ocr_engine == OCREngine.EASYOCR:
            engine = EasyOCREngine()  # type: ignore[assignment]
            if engine.is_available():
                return engine
            # Fallback to Tesseract
            logger.debug("EasyOCR not available, falling back to Tesseract")
            fallback = TesseractEngine()  # type: ignore[assignment]
            if fallback.is_available():
                return fallback

        elif options.ocr_engine == OCREngine.PADDLEOCR:
            engine = PaddleOCREngine(use_gpu=False)  # type: ignore[assignment]
            if engine.is_available():
                return engine
            # Fallback to Tesseract
            logger.debug("PaddleOCR not available, falling back to Tesseract")
            fallback = TesseractEngine()  # type: ignore[assignment]
            if fallback.is_available():
                return fallback

        logger.error(f"OCR engine not available: {options.ocr_engine}")
        return None

    def _create_text_matcher(self, options: TextFindOptions) -> BaseMatcher:
        """Create text matcher based on options.

        Args:
            options: Text options

        Returns:
            Text matcher instance
        """
        case_sensitive = options.case_sensitive
        ignore_whitespace = options.ignore_whitespace

        if options.match_type == TextMatchType.EXACT:
            return ExactMatcher(case_sensitive, ignore_whitespace)

        elif options.match_type == TextMatchType.CONTAINS:
            return ContainsMatcher(case_sensitive, ignore_whitespace)

        elif options.match_type == TextMatchType.STARTS_WITH:
            return StartsWithMatcher(case_sensitive, ignore_whitespace)

        elif options.match_type == TextMatchType.ENDS_WITH:
            return EndsWithMatcher(case_sensitive, ignore_whitespace)

        elif options.match_type == TextMatchType.REGEX:
            return RegexMatcher(case_sensitive, ignore_whitespace)

        elif options.match_type == TextMatchType.FUZZY:
            return FuzzyMatcher(case_sensitive, ignore_whitespace, options.fuzzy_threshold)

        # Default to contains matcher
        return ContainsMatcher(case_sensitive, ignore_whitespace)

    def _match_text(
        self,
        search_text: str,
        ocr_results: list[OCRResult],
        matcher: BaseMatcher,
        options: TextFindOptions,
    ) -> list[Match]:
        """Match search text against OCR results.

        Args:
            search_text: Text to search for
            ocr_results: OCR results to search in
            matcher: Text matcher to use
            options: Text options

        Returns:
            List of matches
        """
        matches = []

        for ocr_result in ocr_results:
            # Calculate similarity using matcher
            similarity = matcher.match(search_text, ocr_result.text)

            # Check if similarity meets threshold
            if similarity >= options.similarity:
                match = Match(
                    target=Location(region=ocr_result.region),
                    score=similarity,
                    ocr_text=ocr_result.text if options.return_text_content else "",
                )
                matches.append(match)

        return matches

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

    def clear_cache(self) -> None:
        """Clear OCR cache."""
        self._ocr_cache.clear()
        logger.debug("OCR cache cleared")
