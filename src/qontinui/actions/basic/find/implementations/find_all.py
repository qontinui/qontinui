"""Find all implementation - ported from Qontinui framework.

Exhaustive pattern matching to find all occurrences.
"""

import logging
from dataclasses import dataclass, field
from typing import cast

from .....model.element.location import Location
from .....model.element.region import Region
from .....model.match.match import Match
from ....object_collection import ObjectCollection
from ..options.pattern_find_options import PatternFindOptions

# Note: FindImage from implementations/find_image was removed (legacy FindImageOrchestrator)
# FindAll needs to be refactored to use RealFindImplementation or removed
# from .find_image import FindImage

logger = logging.getLogger(__name__)


@dataclass
class FindAll:
    """Exhaustive pattern matching implementation.

    Port of FindAll from Qontinui framework class.

    Finds all occurrences of patterns with advanced filtering
    to remove duplicates and overlapping matches.
    """

    # Delegate to FindImage for actual matching (DISABLED - FindImage removed)
    # _image_finder: FindImage = field(default_factory=FindImage)

    # Configuration
    min_distance_between_matches: int = 10  # Minimum pixels between matches
    overlap_threshold: float = 0.3  # Maximum allowed overlap
    use_sliding_window: bool = False  # Use sliding window for thorough search
    window_step: int = 5  # Step size for sliding window

    def find(self, object_collection: ObjectCollection, options: PatternFindOptions) -> list[Match]:
        """Find all occurrences of patterns.

        Args:
            object_collection: Objects to find
            options: Pattern find configuration

        Returns:
            List of all matches found

        Note:
            This class is currently disabled as it depends on the removed FindImageOrchestrator.
            Needs refactoring to use RealFindImplementation.
        """
        raise NotImplementedError(
            "FindAll is disabled - it depends on removed FindImageOrchestrator. "
            "Use RealFindImplementation instead or refactor this class."
        )

    def _sliding_window_search(
        self, object_collection: ObjectCollection, options: PatternFindOptions
    ) -> list[Match]:
        """Perform sliding window search for thorough coverage.

        Args:
            object_collection: Objects to find
            options: Pattern options

        Returns:
            All matches found
        """
        all_matches = []

        # Get search regions
        search_regions = (
            options.search_regions
            if options.search_regions
            else [Region(0, 0, 1920, 1080)]  # Placeholder for full screen
        )

        for region in search_regions:
            # Get pattern dimensions
            for pattern in options.patterns:
                pattern_width = 50  # Placeholder - would get actual pattern size
                pattern_height = 50

                # Slide window across region
                for y in range(
                    region.y,
                    region.y + region.height - pattern_height,
                    self.window_step,
                ):
                    for x in range(
                        region.x,
                        region.x + region.width - pattern_width,
                        self.window_step,
                    ):
                        # Create window region
                        window = Region(x, y, pattern_width * 2, pattern_height * 2)

                        # Search in window
                        window_options = PatternFindOptions()
                        window_options.patterns = [pattern]
                        window_options.add_search_region(window)
                        window_options.similarity = options.similarity

                        matches = self._image_finder.find(object_collection, window_options)  # type: ignore[attr-defined]
                        all_matches.extend(matches)

        return all_matches

    def _filter_duplicates(self, matches: list[Match], options: PatternFindOptions) -> list[Match]:
        """Remove duplicate matches at same location.

        Args:
            matches: Matches to filter
            options: Pattern options

        Returns:
            Filtered matches
        """
        if not matches:
            return matches

        unique_matches = []
        seen_locations: set[tuple[int, int]] = set()

        for match in matches:
            if match.target:
                # Round location to grid to catch near-duplicates
                grid_x = round(match.target.x / 5) * 5
                grid_y = round(match.target.y / 5) * 5
                location_key = (grid_x, grid_y)

                if location_key not in seen_locations:
                    seen_locations.add(location_key)
                    unique_matches.append(match)
            else:
                unique_matches.append(match)

        logger.debug(f"Filtered {len(matches) - len(unique_matches)} duplicate matches")
        return unique_matches

    def _filter_overlapping(self, matches: list[Match], options: PatternFindOptions) -> list[Match]:
        """Remove overlapping matches based on IoU.

        Args:
            matches: Matches to filter
            options: Pattern options

        Returns:
            Filtered matches
        """
        if not matches:
            return matches

        # Sort by similarity (best first)
        sorted_matches = sorted(matches, key=lambda m: m.similarity, reverse=True)

        filtered: list[Match] = []
        for match in sorted_matches:
            # Check overlap with already accepted matches
            is_overlapping = False

            for accepted in filtered:
                if self._calculate_overlap(match.region, accepted.region) > self.overlap_threshold:
                    is_overlapping = True
                    break

            if not is_overlapping:
                filtered.append(match)

        logger.debug(f"Filtered {len(matches) - len(filtered)} overlapping matches")
        return filtered

    def _filter_by_distance(self, matches: list[Match], options: PatternFindOptions) -> list[Match]:
        """Filter matches that are too close together.

        Args:
            matches: Matches to filter
            options: Pattern options

        Returns:
            Filtered matches
        """
        if not matches or self.min_distance_between_matches <= 0:
            return matches

        # Sort by similarity
        sorted_matches = sorted(matches, key=lambda m: m.similarity, reverse=True)

        filtered: list[Match] = []
        for match in sorted_matches:
            too_close = False

            for accepted in filtered:
                distance = self._calculate_distance(match.target, accepted.target)
                if distance < self.min_distance_between_matches:
                    too_close = True
                    break

            if not too_close:
                filtered.append(match)

        logger.debug(f"Filtered {len(matches) - len(filtered)} matches too close together")
        return filtered

    def _sort_matches(self, matches: list[Match]) -> list[Match]:
        """Sort matches by position (top-left to bottom-right).

        Args:
            matches: Matches to sort

        Returns:
            Sorted matches
        """

        def sort_key(match: Match) -> tuple[int, int]:
            if match.target:
                return (match.target.y, match.target.x)
            return (0, 0)

        return sorted(matches, key=sort_key)

    def _calculate_overlap(self, region1: Region | None, region2: Region | None) -> float:
        """Calculate overlap ratio between two regions.

        Args:
            region1: First region
            region2: Second region

        Returns:
            Overlap ratio (0.0-1.0)
        """
        if not region1 or not region2:
            return 0.0

        # Calculate intersection
        x1 = max(region1.x, region2.x)
        y1 = max(region1.y, region2.y)
        x2 = min(region1.x + region1.width, region2.x + region2.width)
        y2 = min(region1.y + region1.height, region2.y + region2.height)

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union
        area1 = region1.width * region1.height
        area2 = region2.width * region2.height
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def _calculate_distance(self, loc1: Location | None, loc2: Location | None) -> float:
        """Calculate distance between two locations.

        Args:
            loc1: First location
            loc2: Second location

        Returns:
            Euclidean distance
        """
        if not loc1 or not loc2:
            return float("inf")

        dx = loc1.x - loc2.x
        dy = loc1.y - loc2.y
        return cast(float, (dx * dx + dy * dy) ** 0.5)


@dataclass
class FindAllBuilder:
    """Builder for FindAll configuration.

    Provides fluent interface for FindAll setup.
    """

    _find_all: FindAll = field(default_factory=FindAll)

    def with_min_distance(self, distance: int) -> "FindAllBuilder":
        """Set minimum distance between matches.

        Args:
            distance: Minimum pixel distance

        Returns:
            Self for fluent interface
        """
        self._find_all.min_distance_between_matches = distance
        return self

    def with_overlap_threshold(self, threshold: float) -> "FindAllBuilder":
        """Set maximum allowed overlap.

        Args:
            threshold: Overlap threshold (0.0-1.0)

        Returns:
            Self for fluent interface
        """
        self._find_all.overlap_threshold = threshold
        return self

    def enable_sliding_window(self, step: int = 5) -> "FindAllBuilder":
        """Enable sliding window search.

        Args:
            step: Window step size

        Returns:
            Self for fluent interface
        """
        self._find_all.use_sliding_window = True
        self._find_all.window_step = step
        return self

    def build(self) -> FindAll:
        """Build FindAll instance.

        Returns:
            Configured FindAll
        """
        return self._find_all
