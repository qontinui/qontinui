"""MaskedFind - Enhanced find operations with mask support.

Extends the Find system to support masked pattern matching.
"""

from typing import Any

import numpy as np

from ..model.element import Image, Location, Pattern, Region
from .find_image import FindImage
from .match import Match
from .matches import Matches


class MaskedFind(FindImage):
    """Enhanced find class that supports masked patterns.

    This class extends FindImage to support mask-based pattern matching,
    allowing for more accurate matching by focusing only on relevant pixels.

    Note: All Patterns now have mask support built-in. This class is retained
    for compatibility but delegates to the standard Pattern class.
    """

    def __init__(self, pattern: Image | Pattern | str | None = None) -> None:
        """Initialize MaskedFind with optional pattern.

        Args:
            pattern: Image, Pattern, or path to find
        """
        super().__init__()

        self._pattern: Pattern | None = None
        self._use_mask = True
        self._mask_threshold = 0.5  # Threshold for considering mask pixels active

        if pattern:
            if isinstance(pattern, Pattern):
                self.pattern(pattern)
            elif isinstance(pattern, str):
                self.image(Image.from_file(pattern))
            else:
                self.image(pattern)

    def pattern(self, pattern: Pattern) -> "MaskedFind":
        """Set the Pattern to find (fluent).

        Args:
            pattern: Pattern to search for

        Returns:
            Self for chaining
        """
        # Call parent to maintain compatibility
        super().pattern(pattern)
        self._pattern = pattern
        # Set similarity threshold from pattern
        if pattern.similarity:
            self.similarity(pattern.similarity)
        return self

    def use_mask(self, enable: bool = True) -> "MaskedFind":
        """Enable/disable mask usage (fluent).

        Args:
            enable: True to use mask, False for full rectangle matching

        Returns:
            Self for chaining
        """
        self._use_mask = enable
        return self

    def mask_threshold(self, threshold: float) -> "MaskedFind":
        """Set threshold for active mask pixels (fluent).

        Args:
            threshold: Threshold value (0.0-1.0) for considering mask pixels active

        Returns:
            Self for chaining
        """
        self._mask_threshold = max(0.0, min(1.0, threshold))
        return self

    def _calculate_masked_similarity(
        self,
        template: np.ndarray[Any, Any],
        template_mask: np.ndarray[Any, Any],
        search_image: np.ndarray[Any, Any],
        x: int,
        y: int,
    ) -> float:
        """Calculate similarity using masked matching.

        Args:
            template: Template image array
            template_mask: Template mask array
            search_image: Image to search in
            x: X coordinate in search image
            y: Y coordinate in search image

        Returns:
            Similarity score (0.0-1.0)
        """
        h, w = template.shape[:2]

        # Check bounds
        if x + w > search_image.shape[1] or y + h > search_image.shape[0]:
            return 0.0

        # Extract region from search image
        region = search_image[y : y + h, x : x + w]

        # Apply mask
        active_mask = template_mask > self._mask_threshold

        if len(template.shape) == 3:
            # Expand mask for color images
            active_mask_expanded = np.expand_dims(active_mask, axis=2)
            active_mask_expanded = np.repeat(active_mask_expanded, template.shape[2], axis=2)
        else:
            active_mask_expanded = active_mask

        # Calculate masked difference
        masked_template = template * active_mask_expanded
        masked_region = region * active_mask_expanded

        # Calculate similarity only for active pixels
        active_pixels = np.sum(active_mask)
        if active_pixels == 0:
            return 0.0

        # Convert to float for precise calculation
        masked_template = masked_template.astype(np.float32)
        masked_region = masked_region.astype(np.float32)

        # Calculate difference
        diff = np.abs(masked_template - masked_region)

        # Normalize and calculate similarity
        normalized_diff = diff / 255.0
        weighted_diff_sum = np.sum(normalized_diff * active_mask_expanded)
        avg_diff = weighted_diff_sum / (active_pixels * (3 if len(template.shape) == 3 else 1))

        similarity = 1.0 - avg_diff
        return float(similarity)

    def find(self) -> Match | None:
        """Find single best match using masked pattern matching.

        Returns:
            Best match found or None
        """
        if self._pattern and self._use_mask:
            # Use masked matching
            matches = self._find_with_mask(find_all=False)
            return matches[0] if matches else None
        else:
            # Fall back to standard matching
            return super().find()

    def find_all_matches(self) -> Matches:
        """Find all matches using masked pattern matching.

        Returns:
            All matches found
        """
        if self._pattern and self._use_mask:
            # Use masked matching
            matches = self._find_with_mask(find_all=True)
            return Matches(matches)
        else:
            # Fall back to standard matching
            return super().find_all_matches()

    def _find_with_mask(self, find_all: bool = False) -> list[Match]:
        """Internal method to find matches using mask.

        Args:
            find_all: Whether to find all matches or just the best

        Returns:
            List of matches found
        """
        if not self._pattern:
            return []

        # Get search image (would come from screenshot or screen capture)
        # This is a placeholder - in production, would get actual screen image
        search_image = self._get_search_image()
        if search_image is None:
            return []

        template = self._pattern.pixel_data
        mask = self._pattern.mask

        matches = []
        best_match = None
        best_similarity = 0.0

        # Sliding window search
        h, w = template.shape[:2]
        search_h, search_w = search_image.shape[:2]

        # Define search region
        if self._search_region:
            if isinstance(self._search_region, Region):
                regions = [self._search_region]
            else:
                # SearchRegions case
                regions = [Region(0, 0, search_w, search_h)]
        else:
            regions = [Region(0, 0, search_w, search_h)]

        for region in regions:
            for y in range(region.y, min(region.y + region.height - h, search_h - h)):
                for x in range(region.x, min(region.x + region.width - w, search_w - w)):
                    similarity = self._calculate_masked_similarity(
                        template, mask, search_image, x, y
                    )

                    if similarity >= self._min_similarity:
                        from ..model.match import Match as MatchObject

                        match_obj = MatchObject(
                            target=Location(region=Region(x, y, w, h)),
                            score=similarity,
                            name=self._pattern.name,
                        )
                        match = Match(match_object=match_obj)

                        if find_all:
                            matches.append(match)
                        elif similarity > best_similarity:
                            best_match = match
                            best_similarity = similarity

        if find_all:
            # Sort by score
            matches.sort(key=lambda m: m.score, reverse=True)
            return matches
        else:
            return [best_match] if best_match else []

    def _get_search_image(self) -> np.ndarray[Any, Any] | None:
        """Get the image to search in.

        Returns:
            Search image as numpy array, or None if not available
        """
        # In production, this would:
        # 1. Take a screenshot if no specific image provided
        # 2. Load the search image if provided
        # 3. Apply any preprocessing (grayscale, edges, etc.)

        # For now, return None as placeholder
        return None

    def optimize_pattern(
        self,
        positive_samples: list[np.ndarray[Any, Any]],
        negative_samples: list[np.ndarray[Any, Any]] | None = None,
    ) -> "MaskedFind":
        """Optimize the pattern's mask based on samples (fluent).

        Args:
            positive_samples: Images that should match
            negative_samples: Images that should not match

        Returns:
            Self for chaining
        """
        if self._pattern:
            optimized_mask, metrics = self._pattern.optimize_mask(
                positive_samples,
                negative_samples,
                method="discriminative" if negative_samples else "stability",
            )
            self._pattern.update_mask(optimized_mask)
            print(f"Mask optimized: density={metrics.get('mask_density', 0):.2%}")

        return self


class MaskedFindBuilder:
    """Builder for creating MaskedFind operations with various configurations."""

    def __init__(self) -> None:
        """Initialize the builder."""
        self._find = MaskedFind()

    def with_pattern(self, pattern: Pattern | Image | str) -> "MaskedFindBuilder":
        """Set the pattern to find.

        Args:
            pattern: Pattern to search for

        Returns:
            Self for chaining
        """
        if isinstance(pattern, Pattern):
            self._find.pattern(pattern)
        elif isinstance(pattern, str):
            self._find.image(Image.from_file(pattern))
        else:
            self._find.image(pattern)
        return self

    def with_mask(self, enable: bool = True) -> "MaskedFindBuilder":
        """Enable or disable mask usage.

        Args:
            enable: Whether to use mask

        Returns:
            Self for chaining
        """
        self._find.use_mask(enable)
        return self

    def with_similarity(self, threshold: float) -> "MaskedFindBuilder":
        """Set similarity threshold.

        Args:
            threshold: Similarity threshold (0.0-1.0)

        Returns:
            Self for chaining
        """
        self._find.similarity(threshold)
        return self

    def in_region(self, region: Region) -> "MaskedFindBuilder":
        """Set search region.

        Args:
            region: Region to search in

        Returns:
            Self for chaining
        """
        self._find.search_region(region)
        return self

    def build(self) -> MaskedFind:
        """Build and return the configured MaskedFind.

        Returns:
            Configured MaskedFind instance
        """
        result: MaskedFind = self._find
        return result
