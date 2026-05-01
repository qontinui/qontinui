"""FindExecutor orchestrates find operations using specialized components.

Coordinates screenshot capture, pattern matching, and match filtering
to execute complete find operations. Uses dependency injection for
testability and flexibility.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..model.element import Location, Pattern, Region
from ..model.match import Match as MatchObject
from ..model.search_regions import SearchRegions
from .filters.match_filter import MatchFilter
from .match import Match
from .matchers.image_matcher import ImageMatcher
from .screenshot.screenshot_provider import ScreenshotProvider

if TYPE_CHECKING:
    from .backends.cascade import CascadeDetector, MatchSettings

logger = logging.getLogger(__name__)


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
        cascade_detector: CascadeDetector | None = None,
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
            cascade_detector: Optional CascadeDetector for graduated
                             fallback across multiple detection backends.
                             When provided, the executor tries the cascade
                             first and falls back to the raw matcher if the
                             cascade returns no results.

        Raises:
            ValueError: If screenshot_provider or matcher is None
        """
        if screenshot_provider is None:
            raise ValueError("screenshot_provider cannot be None")
        if matcher is None:
            raise ValueError("matcher cannot be None")

        self.screenshot_provider = screenshot_provider
        self.matcher = matcher
        self.filters = list(filters) if filters else []
        self.cascade_detector = cascade_detector

    @classmethod
    def with_cascade(
        cls,
        screenshot_provider: ScreenshotProvider,
        matcher: ImageMatcher | None = None,
        filters: list[MatchFilter] | None = None,
        accessibility_capture: Any = None,
        ocr_engine: Any = None,
        llm_client: Any = None,
    ) -> FindExecutor:
        """Create a FindExecutor with an auto-configured CascadeDetector.

        This is the recommended way to create a FindExecutor for production
        use. The cascade provides graduated fallback across all available
        detection backends (template → edge → feature → invariant → QATM →
        OCR → OmniParser → VLM).

        Args:
            screenshot_provider: Component for capturing screenshots.
            matcher: Pattern matcher. Defaults to TemplateMatcher if None.
            filters: Optional list of match filters.
            accessibility_capture: Optional IAccessibilityCapture for
                                   accessibility-tree-first detection.
            ocr_engine: Optional IOCREngine for text-based detection.
            llm_client: Optional VisionLLMClient for VLM fallback.

        Returns:
            Configured FindExecutor with CascadeDetector enabled.
        """
        if matcher is None:
            from .matchers.template_matcher import TemplateMatcher

            matcher = TemplateMatcher()

        try:
            from .backends.cascade import CascadeDetector

            cascade = CascadeDetector(
                accessibility_capture=accessibility_capture,
                ocr_engine=ocr_engine,
                llm_client=llm_client,
            )
        except Exception:
            logger.warning(
                "Could not create CascadeDetector, cascade fallback disabled",
                exc_info=True,
            )
            cascade = None

        return cls(
            screenshot_provider=screenshot_provider,
            matcher=matcher,
            filters=filters,
            cascade_detector=cascade,
        )

    def execute(
        self,
        pattern: Pattern,
        search_region: Region | SearchRegions | None = None,
        similarity: float = 0.8,
        find_all: bool = False,
        match_settings: MatchSettings | None = None,
    ) -> list[Match]:
        """Execute find operation using configured components.

        Orchestrates the complete find workflow:
        1. Capture screenshot from the specified region (or entire screen)
        2. If a CascadeDetector is configured, try it first
        3. Otherwise (or on cascade miss), use the raw ImageMatcher
        4. Apply all filters in sequence to refine results
        5. Return final filtered matches

        Args:
            pattern: Pattern to search for. Must have valid pixel data.
            search_region: Optional region to search within.
                          Can be a single Region or SearchRegions with multiple areas.
                          If None, searches the entire screen.
            similarity: Minimum similarity threshold (0.0 to 1.0).
                       Higher values require closer matches.
            find_all: If True, find all matches above threshold.
                     If False, return only the best match.
            match_settings: Optional per-target settings for the CascadeDetector.

        Returns:
            List of Match objects sorted by similarity (highest first).
            Empty list if no matches meet the criteria.

        Raises:
            ValueError: If pattern is None or has no pixel data
            RuntimeError: If screenshot capture or matching fails
        """
        if pattern is None:
            raise ValueError("pattern cannot be None")
        if pattern.pixel_data is None:
            raise ValueError("pattern must have pixel data")

        # Step 1: Capture screenshot
        screenshot = self._capture_screenshot(search_region)
        search_region_tuple = self._convert_search_region(search_region)

        # Step 2: Try CascadeDetector if available
        # Use explicit match_settings, falling back to pattern-level settings
        effective_settings = match_settings or getattr(pattern, "match_settings", None)
        if self.cascade_detector is not None:
            cascade_matches = self._try_cascade(
                pattern,
                screenshot,
                search_region_tuple,
                similarity,
                find_all,
                effective_settings,
            )
            if cascade_matches:
                return self._apply_filters(cascade_matches)

        # Step 3: Fall back to raw ImageMatcher
        matches = self.matcher.find_matches(
            screenshot=screenshot,
            pattern=pattern,
            find_all=find_all,
            similarity=similarity,
            search_region=search_region_tuple,
        )

        # Step 4: Apply filters in sequence
        return self._apply_filters(matches)

    def _try_cascade(
        self,
        pattern: Pattern,
        screenshot: Any,
        search_region: tuple[int, int, int, int] | None,
        similarity: float,
        find_all: bool,
        match_settings: MatchSettings | None,
    ) -> list[Match]:
        """Try the CascadeDetector and convert results to Match objects.

        Returns:
            List of Match objects, or empty list if cascade found nothing.
        """
        assert self.cascade_detector is not None

        config: dict[str, Any] = {
            "needle_type": "template",
            "min_confidence": similarity,
            "find_all": find_all,
            "search_region": search_region,
        }
        if match_settings is not None:
            config["match_settings"] = match_settings

        try:
            detection_results = self.cascade_detector.find(pattern, screenshot, config)
        except Exception:
            logger.warning(
                "CascadeDetector failed, falling back to matcher", exc_info=True
            )
            return []

        # Convert DetectionResult → Match
        matches: list[Match] = []
        for dr in detection_results:
            region = Region(x=dr.x, y=dr.y, width=dr.width, height=dr.height)
            cx, cy = dr.center
            from ..model.match import MatchMetadata

            meta = MatchMetadata(
                backend_metadata=dict(dr.metadata) if dr.metadata else {}
            )
            match_obj = MatchObject(
                target=Location(x=cx, y=cy, region=region),
                score=dr.confidence,
                name=pattern.name,
                metadata=meta,
            )
            matches.append(Match(match_obj))

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches

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
                raise ValueError(
                    f"Filter {type(filter_instance).__name__} failed: {e}"
                ) from e
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
