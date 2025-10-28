"""Main image finding orchestrator."""

import logging
from typing import Any

import cv2
import numpy as np

from ......model.element.pattern import Pattern
from ......model.match.match import Match
from .....object_collection import ObjectCollection
from ...options.pattern_find_options import PatternFindOptions
from .async_support import AsyncFinder
from .image_capture import RegionCapturer, ScreenCapturer
from .match_method_registry import MatchMethodRegistry
from .matchers import MultiScaleMatcher, SingleScaleMatcher

logger = logging.getLogger(__name__)


class FindImageOrchestrator:
    """Coordinates image finding operations.

    Main entry point for template-based image matching.
    Delegates to specialized components for capture, matching, and async execution.
    """

    def __init__(self) -> None:
        """Initialize orchestrator with capture and async support."""
        self.screen_capturer = ScreenCapturer()
        self.region_capturer = RegionCapturer()
        self.async_finder = AsyncFinder()
        self._check_opencv()

    def _check_opencv(self) -> bool:
        """Check OpenCV availability.

        Returns:
            True if OpenCV available, False otherwise
        """
        try:
            import cv2  # noqa: F401

            logger.debug("OpenCV available for image matching")
            return True
        except ImportError:
            logger.error("OpenCV not available, image matching disabled")
            return False

    def find(self, object_collection: ObjectCollection, options: PatternFindOptions) -> list[Match]:
        """Find images using template matching.

        Args:
            object_collection: Objects containing patterns to find
            options: Pattern matching configuration

        Returns:
            List of matches found
        """
        # Get patterns to search for
        patterns = self._extract_patterns(object_collection, options)
        if not patterns:
            logger.warning("No patterns to find")
            return []

        # Capture search images
        search_images = self._capture_search_images(options)
        if not search_images:
            logger.error("Failed to capture search images")
            return []

        # Search for each pattern
        matches = []
        for pattern in patterns:
            pattern_matches = self._find_pattern(pattern, search_images, options)
            matches.extend(pattern_matches)

            # Early termination
            if options.search_type.name == "FIRST" and matches:
                break

        return matches

    async def find_async(
        self,
        object_collection: ObjectCollection,
        options: PatternFindOptions,
    ) -> list[Match]:
        """Find images asynchronously with parallel pattern matching.

        Args:
            object_collection: Objects containing patterns to find
            options: Pattern matching configuration

        Returns:
            List of matches found
        """
        # Get patterns
        patterns = self._extract_patterns(object_collection, options)

        # Use async finder for parallel execution
        return await self.async_finder.find_patterns(
            patterns=patterns,
            capture_func=lambda: self._capture_search_images(options),
            search_func=self._find_pattern,
            options=options,
        )

    def _extract_patterns(
        self, object_collection: ObjectCollection, options: PatternFindOptions
    ) -> list[Pattern]:
        """Extract patterns from object collection and options.

        Args:
            object_collection: Object collection
            options: Pattern options

        Returns:
            List of patterns to find
        """
        patterns = []

        # From options
        patterns.extend(options.patterns)

        # From state images in options
        for state_image in options.state_images:
            if hasattr(state_image, "patterns"):
                patterns.extend(state_image.patterns)

        # From object collection
        for state_image in object_collection.state_images:
            if hasattr(state_image, "patterns"):
                patterns.extend(state_image.patterns)

        return patterns

    def _capture_search_images(self, options: PatternFindOptions) -> list[np.ndarray[Any, Any]]:
        """Capture images to search within.

        Args:
            options: Pattern options with search regions

        Returns:
            List of search images
        """
        if options.search_regions:
            return self.region_capturer.capture_multiple(options.search_regions)
        else:
            img = self.screen_capturer.capture()
            return [img] if img is not None else []

    def _find_pattern(
        self,
        pattern: Pattern,
        search_images: list[np.ndarray[Any, Any]],
        options: PatternFindOptions,
    ) -> list[Match]:
        """Find single pattern in search images.

        Args:
            pattern: Pattern to find
            search_images: Images to search within
            options: Pattern matching configuration

        Returns:
            Matches for this pattern
        """
        # Load pattern image
        template = self._load_pattern_image(pattern)
        if template is None:
            logger.warning(f"Could not load pattern image for {pattern}")
            return []

        # Preprocess template
        template = self._preprocess_image(template, options)

        # Get matcher
        cv2_method = MatchMethodRegistry.get_cv2_method(options.match_method)
        matcher = self._create_matcher(options.scale_invariant, cv2_method)

        # Search in each image
        matches = []
        for search_img in search_images:
            # Preprocess search image
            processed_img = self._preprocess_image(search_img, options)

            # Find matches
            img_matches = matcher.find_matches(template, processed_img, options)
            matches.extend(img_matches)

        return matches

    def _create_matcher(self, scale_invariant: bool, cv2_method: int):
        """Create appropriate matcher based on options.

        Args:
            scale_invariant: Whether to use multi-scale matching
            cv2_method: OpenCV method constant

        Returns:
            Matcher instance
        """
        if scale_invariant:
            return MultiScaleMatcher(cv2_method)
        else:
            return SingleScaleMatcher(cv2_method)

    def _preprocess_image(
        self, image: np.ndarray[Any, Any], options: PatternFindOptions
    ) -> np.ndarray[Any, Any]:
        """Apply preprocessing to image.

        Args:
            image: Image to preprocess
            options: Preprocessing options

        Returns:
            Preprocessed image
        """
        processed = image

        # Grayscale conversion
        if options.use_grayscale and len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # Edge detection
        if options.use_edges:
            processed = cv2.Canny(processed, options.edge_threshold1, options.edge_threshold2)

        return processed

    def _load_pattern_image(self, pattern: Pattern) -> np.ndarray[Any, Any] | None:
        """Load pattern image from Pattern object.

        IMPLEMENTATION REQUIRED: This method needs to extract the image data from
        the Pattern object. Pattern objects contain image paths or embedded image
        data that must be loaded into a numpy array for template matching.

        Args:
            pattern: Pattern to load

        Returns:
            Image array or None if loading fails

        Raises:
            NotImplementedError: This method must be implemented to enable image finding
        """
        logger.error(
            f"Pattern image loading not implemented. Cannot load pattern: {pattern}. "
            "FindImageOrchestrator._load_pattern_image() must be implemented."
        )
        return None
