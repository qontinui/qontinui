"""Find class - ported from Qontinui framework.

Base class for all find operations with builder pattern.
"""

from __future__ import annotations

import time
from typing import Any

from ..actions import FindOptions
from ..model.element import Image, Location, Pattern, Region
from ..model.search_regions import SearchRegions
from .find_results import FindResults
from .match import Match
from .matches import Matches

# Type alias for search regions
SearchRegion = Region | SearchRegions


class Find:
    """Base find class with builder pattern.

    Port of Find from Qontinui framework class.
    Provides fluent interface for configuring and executing find operations.
    """

    def __init__(self, target: Pattern | Image | str | None = None):
        """Initialize Find with optional target.

        Args:
            target: Pattern, Image, or image path to find
        """
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
            Pattern object
        """
        if isinstance(target, Pattern):
            return target
        elif isinstance(target, Image):
            return Pattern(image=target)
        elif isinstance(target, str):
            # Create Image from path
            image = Image(name=target, path=target)
            return Pattern(image=image)
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
        self._target = Pattern(image=image)
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
            self._target = self._target.with_similarity(similarity)
        return self

    def search_region(self, region: Region | SearchRegions) -> Find:
        """Set search region (fluent).

        Args:
            region: Region to search within

        Returns:
            Self for chaining
        """
        if isinstance(region, SearchRegion):
            self._search_region = region
        else:
            self._search_region = SearchRegion(region.x, region.y, region.width, region.height)
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
        """Execute the find operation.

        Returns:
            FindResults with matches
        """
        if not self._target:
            return FindResults.empty()

        start_time = time.time()

        # Get screenshot if not provided
        if self._screenshot is None:
            self._screenshot = self._capture_screenshot()

        # Perform the actual finding
        matches = self._perform_find()

        # Sort matches if configured
        sort_criteria = self._options._sort_by
        if sort_criteria == "similarity":
            matches.sort_by_similarity()
        elif sort_criteria == "position":
            matches.sort_by_position()

        # Apply max matches limit
        if not self._options._find_all and matches.size() > 0:
            matches = Matches([matches.first])
        elif self._options._max_matches < matches.size():
            matches = Matches(matches.to_list()[: self._options._max_matches])

        duration = time.time() - start_time

        return FindResults(
            matches=matches,
            pattern=self._target,
            search_region=self._search_region,
            duration=duration,
            screenshot=self._screenshot,
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

    def _capture_screenshot(self) -> Any:
        """Capture screenshot for finding.

        Returns:
            Screenshot image
        """
        # This would capture actual screenshot
        # For now, return placeholder
        from ..actions.pure import PureActions

        pure = PureActions()
        result = pure.capture_screen(
            region=self._search_region.to_tuple() if self._search_region else None
        )
        return result.data if result.success else None

    def _perform_find(self) -> Matches:
        """Perform the actual pattern matching.

        Returns:
            Matches found
        """
        # This would perform actual matching using OpenCV, etc.
        # For now, return empty matches as placeholder
        # In real implementation, this would:
        # 1. Use template matching for self._method == "template"
        # 2. Use feature matching for self._method == "feature"
        # 3. Use AI model for self._method == "ai"

        # Placeholder implementation
        from ..model.element import MatchObject

        # Simulate finding a match at a location
        if self._target and self._screenshot:
            # Create a simulated match
            match_obj = MatchObject(
                location=Location(100, 100),
                region=Region(100, 100, self._target.image.width, self._target.image.height),
                similarity=0.95,
                pattern=self._target,
                screenshot=self._screenshot,
                timestamp=time.time(),
            )
            return Matches([Match(match_obj)])

        return Matches()

    def __str__(self) -> str:
        """String representation."""
        target_name = self._target.name if self._target else "None"
        return f"Find(target='{target_name}', method='{self._method}')"

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"Find(target={self._target}, " f"method='{self._method}', " f"options={self._options})"
        )
