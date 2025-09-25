"""Find image implementation - ported from Qontinui framework.

Core image matching using template matching and other strategies.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .....model.element.location import Location
from .....model.element.pattern import Pattern
from .....model.element.region import Region
from .....model.match.match import Match
from ....object_collection import ObjectCollection
from ..options.pattern_find_options import MatchMethod, PatternFindOptions

logger = logging.getLogger(__name__)


@dataclass
class FindImage:
    """Core image matching implementation.

    Port of FindImage from Qontinui framework class.

    Implements template matching with support for scale and rotation
    invariance, multiple match methods, and various preprocessing options.
    """

    # OpenCV availability (will be set dynamically)
    _cv2_available: bool = field(default=False, init=False)

    def __post_init__(self):
        """Check for OpenCV availability."""
        try:
            import cv2  # noqa: F401

            self._cv2_available = True
            logger.debug("OpenCV available for image matching")
        except ImportError:
            logger.warning("OpenCV not available, image matching limited")

    def find(self, object_collection: ObjectCollection, options: PatternFindOptions) -> list[Match]:
        """Find images using template matching.

        Args:
            object_collection: Objects containing patterns to find
            options: Pattern find configuration

        Returns:
            List of matches found
        """
        if not self._cv2_available:
            logger.error("OpenCV required for image matching")
            return []

        matches = []

        # Get screen or search regions
        search_images = self._get_search_images(options)

        # Get patterns to find
        patterns = self._get_patterns(object_collection, options)

        if not patterns:
            logger.warning("No patterns to find")
            return []

        # Search for each pattern
        for pattern in patterns:
            pattern_matches = self._find_pattern(pattern, search_images, options)
            matches.extend(pattern_matches)

            # Early termination if enough matches found
            if options.search_type.name == "FIRST" and matches:
                break

        return matches

    def _get_patterns(
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

    def _get_search_images(self, options: PatternFindOptions) -> list[np.ndarray]:
        """Get images to search in.

        Args:
            options: Pattern options with search regions

        Returns:
            List of search images
        """

        search_images = []

        if options.search_regions:
            # Capture specific regions
            for region in options.search_regions:
                img = self._capture_region(region)
                if img is not None:
                    search_images.append(img)
        else:
            # Capture full screen
            img = self._capture_screen()
            if img is not None:
                search_images.append(img)

        return search_images

    def _find_pattern(
        self, pattern: Pattern, search_images: list[np.ndarray], options: PatternFindOptions
    ) -> list[Match]:
        """Find a single pattern in search images.

        Args:
            pattern: Pattern to find
            search_images: Images to search in
            options: Pattern options

        Returns:
            Matches for this pattern
        """
        import cv2

        matches = []
        template = self._load_pattern_image(pattern)

        if template is None:
            logger.warning("Could not load pattern image")
            return []

        # Preprocess template if needed
        if options.use_grayscale:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        if options.use_edges:
            template = cv2.Canny(template, options.edge_threshold1, options.edge_threshold2)

        for search_img in search_images:
            # Preprocess search image to match template
            if options.use_grayscale and len(search_img.shape) == 3:
                search_img = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)

            if options.use_edges:
                search_img = cv2.Canny(search_img, options.edge_threshold1, options.edge_threshold2)

            # Apply template matching
            if options.scale_invariant:
                img_matches = self._multiscale_match(template, search_img, options)
            else:
                img_matches = self._single_scale_match(template, search_img, options)

            matches.extend(img_matches)

        return matches

    def _single_scale_match(
        self, template: np.ndarray, image: np.ndarray, options: PatternFindOptions
    ) -> list[Match]:
        """Perform single-scale template matching.

        Args:
            template: Template image
            image: Image to search in
            options: Pattern options

        Returns:
            Matches found
        """
        import cv2

        # Get OpenCV match method
        method = self._get_cv2_method(options.match_method)

        # Perform template matching
        result = cv2.matchTemplate(image, template, method)

        # Find locations above threshold
        matches = []
        h, w = template.shape[:2]

        if options.search_type.name == "ALL":
            # Find all matches above threshold
            locations = np.where(result >= options.similarity)
            for pt in zip(*locations[::-1], strict=False):
                match = Match(
                    target=Location(region=Region(pt[0], pt[1], w, h)),
                    score=float(result[pt[1], pt[0]]),
                )
                matches.append(match)
        else:
            # Find best match
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # For SQDIFF methods, minimum is best match
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                confidence = 1 - min_val if method == cv2.TM_SQDIFF_NORMED else 1.0
                top_left = min_loc
            else:
                confidence = max_val
                top_left = max_loc

            if confidence >= options.similarity:
                match = Match(
                    target=Location(region=Region(top_left[0], top_left[1], w, h)),
                    score=confidence,
                )
                matches.append(match)

        return matches

    def _multiscale_match(
        self, template: np.ndarray, image: np.ndarray, options: PatternFindOptions
    ) -> list[Match]:
        """Perform multi-scale template matching.

        Args:
            template: Template image
            image: Image to search in
            options: Pattern options

        Returns:
            Matches found at various scales
        """
        import cv2

        best_matches = []

        # Try different scales
        scale = options.min_scale
        while scale <= options.max_scale:
            # Resize template
            new_width = int(template.shape[1] * scale)
            new_height = int(template.shape[0] * scale)

            if new_width < 10 or new_height < 10:
                scale += options.scale_step
                continue

            scaled_template = cv2.resize(template, (new_width, new_height))

            # Find matches at this scale
            matches = self._single_scale_match(scaled_template, image, options)

            # Keep track of best matches
            best_matches.extend(matches)

            # Early termination if good match found
            if matches and options.early_termination_threshold > 0:
                if any(m.similarity >= options.early_termination_threshold for m in matches):
                    logger.debug(f"Early termination at scale {scale}")
                    break

            scale += options.scale_step

        return best_matches

    def _get_cv2_method(self, method: MatchMethod) -> int:
        """Convert MatchMethod enum to OpenCV constant.

        Args:
            method: Match method enum

        Returns:
            OpenCV method constant
        """
        import cv2

        mapping = {
            MatchMethod.CORRELATION: cv2.TM_CCORR,
            MatchMethod.CORRELATION_NORMED: cv2.TM_CCORR_NORMED,
            MatchMethod.CORRELATION_COEFFICIENT: cv2.TM_CCOEFF,
            MatchMethod.CORRELATION_COEFFICIENT_NORMED: cv2.TM_CCOEFF_NORMED,
            MatchMethod.SQUARED_DIFFERENCE: cv2.TM_SQDIFF,
            MatchMethod.SQUARED_DIFFERENCE_NORMED: cv2.TM_SQDIFF_NORMED,
        }

        return mapping.get(method, cv2.TM_CCOEFF_NORMED)

    def _load_pattern_image(self, pattern: Pattern) -> np.ndarray | None:
        """Load pattern image.

        Args:
            pattern: Pattern to load

        Returns:
            Image array or None
        """
        # This would load the actual pattern image
        # For now, return None as placeholder
        logger.debug("Loading pattern image")
        return None

    def _capture_screen(self) -> np.ndarray | None:
        """Capture full screen.

        Returns:
            Screen image or None
        """
        # This would capture the actual screen
        # For now, return None as placeholder
        logger.debug("Capturing screen")
        return None

    def _capture_region(self, region: Region) -> np.ndarray | None:
        """Capture specific region.

        Args:
            region: Region to capture

        Returns:
            Region image or None
        """
        # This would capture the specified region
        # For now, return None as placeholder
        logger.debug(f"Capturing region: {region}")
        return None


@dataclass
class ImageFinder:
    """Modern image finder implementation.

    Port of ImageFinder from Qontinui framework (V2).
    Currently delegates to FindImage but provides hook for
    future ML-based implementations.
    """

    # Delegate to legacy implementation for now
    _legacy_finder: FindImage = field(default_factory=FindImage)

    # Future ML implementation
    _ml_finder: Any | None = None

    # Configuration
    use_ml_if_available: bool = True

    def find(self, object_collection: ObjectCollection, options: PatternFindOptions) -> list[Match]:
        """Find images using best available method.

        Args:
            object_collection: Objects to find
            options: Pattern options

        Returns:
            List of matches
        """
        # Check if ML finder is available and should be used
        if self.use_ml_if_available and self._ml_finder is not None:
            logger.debug("Using ML-based image finder")
            return self._ml_finder.find(object_collection, options)

        # Fall back to legacy template matching
        logger.debug("Using template matching finder")
        return self._legacy_finder.find(object_collection, options)

    def set_ml_finder(self, ml_finder: Any):
        """Set ML-based finder implementation.

        Args:
            ml_finder: ML finder implementation
        """
        self._ml_finder = ml_finder
        logger.info("ML finder registered")
