"""Find class - ported from Qontinui framework.

Base class for all find operations with builder pattern.
"""

from __future__ import annotations

import time
from typing import Any

from ..actions import FindOptions
from ..model.element import Image, Pattern, Region
from ..model.search_regions import SearchRegions
from ..reporting.events import EventType, emit_event
from .filters import NMSFilter, RegionFilter, SimilarityFilter
from .find_executor import FindExecutor
from .find_results import FindResults
from .match import Match
from .matchers import TemplateMatcher
from .matches import Matches
from .screenshot import CachedScreenshotProvider, PureActionsScreenshotProvider

# Type alias for search regions
SearchRegion = Region | SearchRegions


class Find:
    """Base find class with builder pattern.

    Port of Find from Qontinui framework class.
    Provides fluent interface for configuring and executing find operations.
    """

    def __init__(self, target: Pattern | Image | str | None = None) -> None:
        """Initialize Find with optional target.

        Args:
            target: Pattern, Image, or image path to find
        """
        # Pattern now has mask support built-in
        self._target: Pattern | None
        if isinstance(target, Pattern):
            self._target = target
        else:
            self._target = self._convert_to_pattern(target) if target else None

        self._options = FindOptions()
        self._search_region: SearchRegion | None = None
        self._screenshot: Any | None = None
        self._method = "template"  # Default matching method

    def _convert_to_pattern(self, target: Pattern | Image | str) -> Pattern:
        """Convert various input types to Pattern.

        Args:
            target: Pattern, Image, or path string

        Returns:
            Pattern object with full mask
        """
        if isinstance(target, Pattern):
            return target
        elif isinstance(target, Image):
            return Pattern.from_image(target)
        elif isinstance(target, str):
            return Pattern.from_file(target)
        else:
            raise ValueError(f"Invalid target type: {type(target)}")

    def pattern(self, pattern: Pattern) -> Find:
        """Set the pattern to find (fluent).

        Args:
            pattern: Pattern to search for

        Returns:
            Self for chaining
        """
        self._target = pattern
        return self

    def image(self, image: Image) -> Find:
        """Set the image to find (fluent).

        Args:
            image: Image to search for

        Returns:
            Self for chaining
        """
        self._target = Pattern.from_image(image)
        return self

    def similarity(self, similarity: float) -> Find:
        """Set minimum similarity threshold (fluent).

        Args:
            similarity: Minimum similarity (0.0 to 1.0)

        Returns:
            Self for chaining
        """
        self._options.min_similarity(similarity)
        if self._target:
            self._target.similarity = similarity
        return self

    def search_region(self, region: Region | SearchRegions) -> Find:
        """Set search region (fluent).

        Args:
            region: Region to search within

        Returns:
            Self for chaining
        """
        if isinstance(region, SearchRegions):
            self._search_region = region
            # SearchRegions doesn't have x, y, width, height directly
            # Don't set individual region in options
        elif isinstance(region, Region):
            self._search_region = Region(region.x, region.y, region.width, region.height)
            self._options.search_region(region.x, region.y, region.width, region.height)
        return self

    def timeout(self, seconds: float) -> Find:
        """Set timeout for find operation (fluent).

        Args:
            seconds: Timeout in seconds

        Returns:
            Self for chaining
        """
        self._options.timeout(seconds)
        return self

    def max_matches(self, count: int) -> Find:
        """Set maximum number of matches to find (fluent).

        Args:
            count: Maximum matches

        Returns:
            Self for chaining
        """
        self._options.max_matches(count)
        return self

    def find_all(self, find_all: bool = True) -> Find:
        """Enable/disable finding all matches (fluent).

        Args:
            find_all: True to find all matches

        Returns:
            Self for chaining
        """
        self._options.find_all(find_all)
        return self

    def method(self, method: str) -> Find:
        """Set matching method (fluent).

        Args:
            method: Matching method ('template', 'feature', 'ai')

        Returns:
            Self for chaining
        """
        self._method = method
        return self

    def screenshot(self, screenshot: Any) -> Find:
        """Set screenshot to search in (fluent).

        Args:
            screenshot: Screenshot image

        Returns:
            Self for chaining
        """
        self._screenshot = screenshot
        return self

    def sort_by(self, criteria: str) -> Find:
        """Set sort criteria for results (fluent).

        Args:
            criteria: Sort criteria ('similarity', 'position', 'size')

        Returns:
            Self for chaining
        """
        self._options.sort_by(criteria)
        return self

    def cache_result(self, cache: bool = True) -> Find:
        """Enable/disable result caching (fluent).

        Args:
            cache: True to cache results

        Returns:
            Self for chaining
        """
        self._options.cache_result(cache)
        return self

    def use_cache(self, use: bool = True) -> Find:
        """Enable/disable cache usage (fluent).

        Args:
            use: True to use cached results

        Returns:
            Self for chaining
        """
        self._options.use_cache(use)
        return self

    def execute(self) -> FindResults:
        """Execute the find operation using FindExecutor.

        Returns:
            FindResults with matches
        """
        if not self._target:
            return FindResults.empty()

        start_time = time.time()

        # Configure screenshot provider with caching
        base_provider = PureActionsScreenshotProvider()
        screenshot_provider = CachedScreenshotProvider(
            provider=base_provider, ttl_seconds=0.1  # Cache for 100ms
        )

        # Configure template matcher
        matcher = TemplateMatcher(method="TM_CCOEFF_NORMED", nms_overlap_threshold=0.3)

        # Configure filters
        filters = []

        # Similarity filter (always applied)
        filters.append(SimilarityFilter(min_similarity=self._options._min_similarity))

        # NMS filter (for find_all mode)
        if self._options._find_all:
            filters.append(NMSFilter(iou_threshold=0.3))  # type: ignore[arg-type]

        # Region filter (if using SearchRegions with multiple regions)
        if isinstance(self._search_region, SearchRegions) and self._search_region.regions:
            filters.append(RegionFilter(self._search_region))  # type: ignore[arg-type]

        # Create executor
        executor = FindExecutor(
            screenshot_provider=screenshot_provider, matcher=matcher, filters=filters  # type: ignore[arg-type]
        )

        # Execute find operation
        match_list = executor.execute(
            pattern=self._target,
            search_region=self._search_region,
            similarity=self._options._min_similarity,
            find_all=self._options._find_all,
        )

        # Convert to Matches collection
        matches = Matches(match_list)

        # Sort matches if configured
        sort_criteria = self._options._sort_by
        if sort_criteria == "similarity":
            matches.sort_by_similarity()
        elif sort_criteria == "position":
            matches.sort_by_position()

        # Apply max matches limit
        if not self._options._find_all and matches.size() > 0:
            first_match = matches.first
            if first_match is not None:
                matches = Matches([first_match])
        elif self._options._max_matches < matches.size():
            matches = Matches(matches.to_list()[: self._options._max_matches])

        duration = time.time() - start_time

        # Emit match attempted event for reporting
        self._emit_match_event(matches, match_list)

        # Convert SearchRegions to Region if needed for results
        search_region_for_results = (
            self._search_region.regions[0]
            if isinstance(self._search_region, SearchRegions) and self._search_region.regions
            else (self._search_region if isinstance(self._search_region, Region) else None)
        )

        return FindResults(
            matches=matches,
            pattern=self._target,
            search_region=search_region_for_results,
            duration=duration,
            screenshot=(screenshot_provider._cache.image if screenshot_provider._cache else None),
            method=self._method,
        )

    def find(self) -> Match | None:
        """Find first/best match.

        Returns:
            First match or None
        """
        results = self.find_all(False).execute()
        return results.first_match

    def find_all_matches(self) -> Matches:
        """Find all matches.

        Returns:
            Matches collection
        """
        results = self.find_all(True).execute()
        return results.matches

    def exists(self) -> bool:
        """Check if pattern exists.

        Returns:
            True if pattern is found
        """
        match = self.find()
        return match is not None and match.exists()

    def wait_until_exists(self, timeout: float = 10.0) -> Match | None:
        """Wait until pattern appears.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            Match when found or None if timeout
        """
        start_time = time.time()
        check_interval = 0.5

        while time.time() - start_time < timeout:
            match = self.find()
            if match and match.exists():
                return match
            time.sleep(check_interval)

        return None

    def wait_until_vanishes(self, timeout: float = 10.0) -> bool:
        """Wait until pattern disappears.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if pattern vanished, False if timeout
        """
        start_time = time.time()
        check_interval = 0.5

        while time.time() - start_time < timeout:
            if not self.exists():
                return True
            time.sleep(check_interval)

        return False

    def _emit_match_event(self, matches: Matches, raw_matches: list[Match]) -> None:
        """Emit match attempted event for reporting.

        Args:
            matches: Final filtered matches
            raw_matches: Raw matches before final filtering
        """
        if not self._target:
            return

        # Get best match from raw matches for reporting
        best_match = raw_matches[0] if raw_matches else None
        best_confidence = best_match.similarity if best_match else 0.0
        threshold_passed = best_confidence >= self._options._min_similarity

        # Emit diagnostic event
        image_id = self._target.name if hasattr(self._target, "name") else "unknown"
        emit_event(
            EventType.MATCH_ATTEMPTED,
            data={
                "image_id": image_id,
                "image_name": (self._target.name if hasattr(self._target, "name") else None),
                "template_dimensions": {
                    "width": (
                        self._target.pixel_data.shape[1]
                        if self._target.pixel_data is not None
                        else 0
                    ),
                    "height": (
                        self._target.pixel_data.shape[0]
                        if self._target.pixel_data is not None
                        else 0
                    ),
                },
                "best_match_location": (
                    {
                        "x": best_match.x if best_match else 0,
                        "y": best_match.y if best_match else 0,
                        "region": (
                            {
                                "x": (
                                    best_match.region.x if best_match and best_match.region else 0
                                ),
                                "y": (
                                    best_match.region.y if best_match and best_match.region else 0
                                ),
                                "width": (
                                    best_match.region.width
                                    if best_match and best_match.region
                                    else 0
                                ),
                                "height": (
                                    best_match.region.height
                                    if best_match and best_match.region
                                    else 0
                                ),
                            }
                            if best_match and best_match.region
                            else None
                        ),
                    }
                    if best_match
                    else None
                ),
                "best_match_confidence": float(best_confidence),
                "similarity_threshold": float(self._options._min_similarity),
                "threshold_passed": bool(threshold_passed),
                "match_method": self._method,
                "find_all_mode": self._options._find_all,
                "num_matches_found": matches.size(),
            },
        )

    def __str__(self) -> str:
        """String representation."""
        target_name = self._target.name if self._target else "None"
        return f"Find(target='{target_name}', method='{self._method}')"

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"Find(target={self._target}, " f"method='{self._method}', " f"options={self._options})"
        )
