"""Find class - ported from Qontinui framework.

Base class for all find operations with builder pattern.

MIGRATION NOTE: This class now delegates to the new FindAction system.
This provides backward compatibility while the codebase is migrated.
"""

from __future__ import annotations

import time
from typing import Any

from ..actions.find import FindAction
from ..actions.find import FindOptions as NewFindOptions
from ..model.element import Image, Pattern, Region
from ..model.match import Match as ModelMatch
from ..model.search_regions import SearchRegions
from .find_results import FindResults
from .match import Match
from .matches import Matches

# Type alias for search regions
SearchRegion = Region | SearchRegions


class Find:
    """Base find class with builder pattern.

    MIGRATION: This class now delegates to FindAction for all operations.
    The builder pattern is preserved for backward compatibility.
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

        # Options storage for builder pattern
        self._min_similarity: float = 0.8
        self._search_region: SearchRegion | None = None
        self._timeout: float = 0.0
        self._max_matches: int = 100
        self._find_all_mode: bool = False
        self._sort_by: str = "similarity"
        self._cache_result: bool = False
        self._use_cache: bool = False
        self._screenshot: Any | None = None
        self._method = "template"  # Default matching method

        # The new system instance
        self._find_action = FindAction()

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
        self._min_similarity = similarity
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
        elif isinstance(region, Region):
            self._search_region = Region(region.x, region.y, region.width, region.height)
        return self

    def timeout(self, seconds: float) -> Find:
        """Set timeout for find operation (fluent).

        Args:
            seconds: Timeout in seconds

        Returns:
            Self for chaining
        """
        self._timeout = seconds
        return self

    def max_matches(self, count: int) -> Find:
        """Set maximum number of matches to find (fluent).

        Args:
            count: Maximum matches

        Returns:
            Self for chaining
        """
        self._max_matches = count
        return self

    def find_all(self, find_all: bool = True) -> Find:
        """Enable/disable finding all matches (fluent).

        Args:
            find_all: True to find all matches

        Returns:
            Self for chaining
        """
        self._find_all_mode = find_all
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
        self._sort_by = criteria
        return self

    def cache_result(self, cache: bool = True) -> Find:
        """Enable/disable result caching (fluent).

        Args:
            cache: True to cache results

        Returns:
            Self for chaining
        """
        self._cache_result = cache
        return self

    def use_cache(self, use: bool = True) -> Find:
        """Enable/disable cache usage (fluent).

        Args:
            use: True to use cached results

        Returns:
            Self for chaining
        """
        self._use_cache = use
        return self

    async def execute(self) -> FindResults:
        """Execute the find operation by delegating to FindAction.

        MIGRATION: This method now delegates to the new FindAction system.
        The result is converted back to FindResults for backward compatibility.

        Returns:
            FindResults with matches
        """
        if not self._target:
            return FindResults.empty()

        start_time = time.time()

        # Build FindOptions for the new system
        search_region = None
        if isinstance(self._search_region, Region):
            search_region = self._search_region
        elif isinstance(self._search_region, SearchRegions) and self._search_region.regions:
            # Use first region from SearchRegions
            search_region = self._search_region.regions[0]

        options = NewFindOptions(
            similarity=self._min_similarity,
            find_all=self._find_all_mode,
            search_region=search_region,
            timeout=self._timeout,
            collect_debug=True,  # Always collect debug for visual reporting
        )

        # Delegate to FindAction
        result = await self._find_action.find(self._target, options)

        # Convert new Match objects to old Match wrapper objects
        old_matches = self._convert_matches(result.matches.to_list())

        # Create Matches collection
        matches = Matches(old_matches)

        # Sort matches if configured
        if self._sort_by == "similarity":
            matches.sort_by_similarity()
        elif self._sort_by == "position":
            matches.sort_by_position()

        # Apply max matches limit
        if not self._find_all_mode and matches.size() > 0:
            first_match = matches.first
            if first_match is not None:
                matches = Matches([first_match])
        elif self._max_matches < matches.size():
            matches = Matches(matches.to_list()[: self._max_matches])

        duration = time.time() - start_time

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
            screenshot=None,  # Screenshot is handled internally by FindAction
            method=self._method,
        )

    def _convert_matches(self, model_matches: list[ModelMatch]) -> list[Match]:
        """Convert new system Match objects to old Match wrapper objects.

        Args:
            model_matches: List of model/match/Match objects

        Returns:
            List of find/match/Match wrapper objects
        """
        return [Match(match_object=m) for m in model_matches]

    async def find(self) -> Match | None:
        """Find first/best match.

        Returns:
            First match or None
        """
        results = await self.find_all(False).execute()
        return results.first_match

    async def find_all_matches(self) -> Matches:
        """Find all matches.

        Returns:
            Matches collection
        """
        results = await self.find_all(True).execute()
        return results.matches

    async def exists(self) -> bool:
        """Check if pattern exists.

        Returns:
            True if pattern is found
        """
        match = await self.find()
        return match is not None and match.exists()

    async def wait_until_exists(self, timeout: float = 10.0) -> Match | None:
        """Wait until pattern appears.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            Match when found or None if timeout
        """
        import asyncio

        start_time = time.time()
        check_interval = 0.5

        while time.time() - start_time < timeout:
            match = await self.find()
            if match and match.exists():
                return match
            await asyncio.sleep(check_interval)

        return None

    async def wait_until_vanishes(self, timeout: float = 10.0) -> bool:
        """Wait until pattern disappears.

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            True if pattern vanished, False if timeout
        """
        import asyncio

        start_time = time.time()
        check_interval = 0.5

        while time.time() - start_time < timeout:
            if not await self.exists():
                return True
            await asyncio.sleep(check_interval)

        return False

    def __str__(self) -> str:
        """String representation."""
        target_name = self._target.name if self._target else "None"
        return f"Find(target='{target_name}', method='{self._method}')"

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"Find(target={self._target}, "
            f"method='{self._method}', "
            f"similarity={self._min_similarity})"
        )
