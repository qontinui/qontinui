"""FindExecutor orchestrates find operations using specialized components.

Coordinates screenshot capture, pattern matching, and match filtering
to execute complete find operations. Uses dependency injection for
testability and flexibility.
"""

from __future__ import annotations

from typing import Any

from ..model.element import Pattern, Region
from ..model.search_regions import SearchRegions
from .filters.match_filter import MatchFilter
from .match import Match
from .matchers.image_matcher import ImageMatcher
from .screenshot.screenshot_provider import ScreenshotProvider


class FindExecutor:
    """Orchestrates find operations using specialized components.

    Coordinates the complete find workflow:
    1. Capture screenshot using ScreenshotProvider
    2. Find matches using ImageMatcher
    3. Apply filters in sequence using MatchFilters
    4. Return filtered results

    Uses dependency injection for all components, allowing easy testing
    and customization of behavior.

    Example:
        >>> from qontinui.find.screenshot import PureActionsScreenshotProvider
        >>> from qontinui.find.matchers import TemplateMatcher
        >>> from qontinui.find.filters import SimilarityFilter, NMSFilter
        >>>
        >>> # Configure components
        >>> provider = PureActionsScreenshotProvider()
        >>> matcher = TemplateMatcher()
        >>> filters = [
        ...     SimilarityFilter(min_similarity=0.8),
        ...     NMSFilter(iou_threshold=0.3)
        ... ]
        >>>
        >>> # Create executor
        >>> executor = FindExecutor(
        ...     screenshot_provider=provider,
        ...     matcher=matcher,
        ...     filters=filters
        ... )
        >>>
        >>> # Execute find operation
        >>> pattern = Pattern.from_file("button.png")
        >>> matches = executor.execute(
        ...     pattern=pattern,
        ...     similarity=0.8,
        ...     find_all=True
        ... )

    Attributes:
        screenshot_provider: Component for capturing screenshots
        matcher: Component for finding pattern matches
        filters: List of filters to apply to matches
    """

    def __init__(
        self,
        screenshot_provider: ScreenshotProvider,
        matcher: ImageMatcher,
        filters: list[MatchFilter] | None = None,
    ) -> None:
        """Initialize FindExecutor with components.

        Args:
            screenshot_provider: Component for capturing screenshots.
                                Must implement ScreenshotProvider interface.
            matcher: Component for finding pattern matches.
                    Must implement ImageMatcher interface.
            filters: Optional list of filters to apply to matches.
                    Filters are applied in the order provided.
                    If None, no filtering is applied.

        Raises:
            ValueError: If screenshot_provider or matcher is None
        """
        if screenshot_provider is None:
            raise ValueError("screenshot_provider cannot be None")
        if matcher is None:
            raise ValueError("matcher cannot be None")

        self.screenshot_provider = screenshot_provider
        self.matcher = matcher
        self.filters = filters or []

    def execute(
        self,
        pattern: Pattern,
        search_region: Region | SearchRegions | None = None,
        similarity: float = 0.8,
        find_all: bool = False,
    ) -> list[Match]:
        """Execute find operation using configured components.

        Orchestrates the complete find workflow:
        1. Capture screenshot from the specified region (or entire screen)
        2. Use matcher to find pattern matches in the screenshot
        3. Apply all filters in sequence to refine results
        4. Return final filtered matches

        Args:
            pattern: Pattern to search for. Must have valid pixel data.
            search_region: Optional region to search within.
                          Can be a single Region or SearchRegions with multiple areas.
                          If None, searches the entire screen.
            similarity: Minimum similarity threshold (0.0 to 1.0).
                       Higher values require closer matches.
            find_all: If True, find all matches above threshold.
                     If False, return only the best match.

        Returns:
            List of Match objects sorted by similarity (highest first).
            Empty list if no matches meet the criteria.

        Raises:
            ValueError: If pattern is None or has no pixel data
            RuntimeError: If screenshot capture or matching fails

        Example:
            >>> pattern = Pattern.from_file("icon.png")
            >>> region = Region(x=100, y=100, width=800, height=600)
            >>> matches = executor.execute(
            ...     pattern=pattern,
            ...     search_region=region,
            ...     similarity=0.9,
            ...     find_all=True
            ... )
            >>> print(f"Found {len(matches)} matches")
            >>> for match in matches:
            ...     print(f"  Match at ({match.x}, {match.y}), "
            ...           f"similarity={match.similarity:.2f}")
        """
        if pattern is None:
            raise ValueError("pattern cannot be None")
        if pattern.pixel_data is None:
            raise ValueError("pattern must have pixel data")

        # Step 1: Capture screenshot
        screenshot = self._capture_screenshot(search_region)

        # Step 2: Find matches using matcher
        # Convert search_region to tuple format for matcher
        search_region_tuple = self._convert_search_region(search_region)
        matches = self.matcher.find_matches(
            screenshot=screenshot,
            pattern=pattern,
            find_all=find_all,
            similarity=similarity,
            search_region=search_region_tuple,
        )

        # Step 3: Apply filters in sequence
        filtered_matches = self._apply_filters(matches)

        return filtered_matches

    def _capture_screenshot(self, search_region: Region | SearchRegions | None) -> Any:
        """Capture screenshot using the configured provider.

        Args:
            search_region: Optional region to capture.
                          For SearchRegions, captures the bounding box.
                          For Region, captures that specific region.
                          If None, captures entire screen.

        Returns:
            Screenshot image (format depends on provider)

        Raises:
            RuntimeError: If screenshot capture fails
        """
        # Determine which region to capture
        capture_region: Region | None = None

        if isinstance(search_region, Region):
            # Single region - capture it directly
            capture_region = search_region
        elif isinstance(search_region, SearchRegions) and search_region.regions:
            # Multiple regions - capture bounding box containing all regions
            capture_region = self._calculate_bounding_box(search_region)

        # Capture screenshot
        return self.screenshot_provider.capture(capture_region)

    def _calculate_bounding_box(self, search_regions: SearchRegions) -> Region:
        """Calculate bounding box that contains all search regions.

        Args:
            search_regions: Collection of regions

        Returns:
            Region representing the bounding box

        Raises:
            ValueError: If search_regions has no regions
        """
        if not search_regions.regions:
            raise ValueError("SearchRegions must contain at least one region")

        # Find min/max coordinates across all regions
        min_x = min(r.x for r in search_regions.regions)
        min_y = min(r.y for r in search_regions.regions)
        max_x = max(r.x + r.width for r in search_regions.regions)
        max_y = max(r.y + r.height for r in search_regions.regions)

        return Region(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
        )

    def _convert_search_region(
        self, search_region: Region | SearchRegions | None
    ) -> tuple[int, int, int, int] | None:
        """Convert search region to tuple format for matcher.

        Note: For SearchRegions with multiple regions, this only passes
        the first region to the matcher. Region filtering should be
        handled by RegionFilter in the filters list.

        Args:
            search_region: Region or SearchRegions to convert

        Returns:
            Tuple of (x, y, width, height) or None
        """
        if search_region is None:
            return None

        if isinstance(search_region, Region):
            return (
                search_region.x,
                search_region.y,
                search_region.width,
                search_region.height,
            )

        if isinstance(search_region, SearchRegions) and search_region.regions:
            # For multiple regions, pass the first one to matcher
            # RegionFilter should be used to filter by all regions
            first_region = search_region.regions[0]
            return (
                first_region.x,
                first_region.y,
                first_region.width,
                first_region.height,
            )

        return None

    def _apply_filters(self, matches: list[Match]) -> list[Match]:
        """Apply all filters in sequence to the matches.

        Filters are applied in the order they were provided during
        initialization. Each filter receives the output from the
        previous filter.

        Args:
            matches: List of matches to filter

        Returns:
            Filtered list of matches

        Raises:
            ValueError: If a filter raises a ValueError
            RuntimeError: If a filter fails unexpectedly
        """
        if not self.filters:
            return matches

        result = matches
        for filter_instance in self.filters:
            try:
                result = filter_instance.filter(result)
            except ValueError as e:
                # Re-raise validation errors from filters
                raise ValueError(f"Filter {type(filter_instance).__name__} failed: {e}") from e
            except Exception as e:
                # Wrap unexpected errors
                raise RuntimeError(
                    f"Filter {type(filter_instance).__name__} raised unexpected error: {e}"
                ) from e

        return result

    def add_filter(self, match_filter: MatchFilter) -> None:
        """Add a filter to the executor's filter chain.

        The filter will be applied after all existing filters.

        Args:
            match_filter: Filter to add

        Raises:
            ValueError: If match_filter is None
        """
        if match_filter is None:
            raise ValueError("match_filter cannot be None")
        self.filters.append(match_filter)

    def clear_filters(self) -> None:
        """Remove all filters from the executor."""
        self.filters.clear()

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"FindExecutor("
            f"screenshot_provider={type(self.screenshot_provider).__name__}, "
            f"matcher={type(self.matcher).__name__}, "
            f"filters={[type(f).__name__ for f in self.filters]})"
        )
