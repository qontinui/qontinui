"""SearchRegionOnObject class - ported from Qontinui framework.

Defines search regions relative to state objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from ..state.state_object import StateObject
from .location import Position
from .region import Region

if TYPE_CHECKING:
    from .search_region import SearchRegion


class SearchStrategy(Enum):
    """Strategy for defining search regions."""

    RELATIVE = auto()  # Relative to object position
    ABSOLUTE = auto()  # Absolute screen coordinates
    EXPANDED = auto()  # Expanded from object bounds
    CONTRACTED = auto()  # Contracted from object bounds
    ADJACENT = auto()  # Adjacent to object (above, below, left, right)


@dataclass
class SearchRegionOnObject:
    """Defines search regions relative to state objects.

    Port of SearchRegionOnObject from Qontinui framework class.

    SearchRegionOnObject allows defining search areas that are dynamically
    positioned relative to other UI elements. This is essential for adaptive
    automation where absolute positions change but relative positions remain
    consistent.

    Key features:
    - Dynamic positioning based on object location
    - Multiple search strategies (relative, expanded, adjacent)
    - Chaining multiple regions for complex searches
    - Fallback regions when primary search fails

    Common use cases:
    - Search for label next to input field
    - Find button below text
    - Locate icon within container
    - Search in expanding/collapsing panels
    - Find elements in scrollable lists

    Example:
        # Search for submit button below form fields
        search_region = SearchRegionOnObject(form_object)
        search_region.set_strategy(SearchStrategy.ADJACENT)
        search_region.set_direction(Direction.BELOW)
        search_region.set_offset(10)  # 10 pixels below form

        # Find button in the defined region
        button = find.in_region(search_region.get_search_region())
    """

    # Base state object to position relative to
    base_object: StateObject | None = None

    # Search strategy to use
    strategy: SearchStrategy = SearchStrategy.RELATIVE

    # Offset from base object (x, y)
    offset_x: int = 0
    offset_y: int = 0

    # Size adjustments (for EXPANDED/CONTRACTED strategies)
    expand_by: int = 0  # Pixels to expand/contract by

    # Position relative to object (for ADJACENT strategy)
    adjacent_position: Position = Position.CENTER
    adjacent_distance: int = 0  # Distance from object edge
    adjacent_width: int | None = None  # Width of adjacent region
    adjacent_height: int | None = None  # Height of adjacent region

    # Absolute region (for ABSOLUTE strategy)
    absolute_region: Region | None = None

    # Additional search regions (for multi-region searches)
    additional_regions: list[SearchRegionOnObject] = field(default_factory=list)

    # Fallback regions if primary search fails
    fallback_regions: list[SearchRegionOnObject] = field(default_factory=list)

    # Search parameters
    similarity: float = 0.7
    timeout: float = 5.0

    # Cache for computed regions
    _cached_region: SearchRegion | None = field(default=None, init=False)
    _cache_valid: bool = field(default=False, init=False)

    def set_base_object(self, obj: StateObject) -> SearchRegionOnObject:
        """Set the base object for relative positioning.

        Args:
            obj: State object to position relative to

        Returns:
            Self for fluent interface
        """
        self.base_object = obj
        self._invalidate_cache()
        return self

    def set_strategy(self, strategy: SearchStrategy) -> SearchRegionOnObject:
        """Set the search strategy.

        Args:
            strategy: Strategy to use

        Returns:
            Self for fluent interface
        """
        self.strategy = strategy
        self._invalidate_cache()
        return self

    def set_offset(self, x: int, y: int) -> SearchRegionOnObject:
        """Set offset from base object.

        Args:
            x: X offset in pixels
            y: Y offset in pixels

        Returns:
            Self for fluent interface
        """
        self.offset_x = x
        self.offset_y = y
        self._invalidate_cache()
        return self

    def expand(self, pixels: int) -> SearchRegionOnObject:
        """Expand search region by specified pixels.

        Args:
            pixels: Pixels to expand by (negative to contract)

        Returns:
            Self for fluent interface
        """
        self.strategy = SearchStrategy.EXPANDED
        self.expand_by = pixels
        self._invalidate_cache()
        return self

    def contract(self, pixels: int) -> SearchRegionOnObject:
        """Contract search region by specified pixels.

        Args:
            pixels: Pixels to contract by

        Returns:
            Self for fluent interface
        """
        self.strategy = SearchStrategy.CONTRACTED
        self.expand_by = -pixels
        self._invalidate_cache()
        return self

    def adjacent_to(
        self,
        position: Position,
        distance: int = 0,
        width: int | None = None,
        height: int | None = None,
    ) -> SearchRegionOnObject:
        """Set search region adjacent to object.

        Args:
            position: Position relative to object (TOP, BOTTOM, LEFT, RIGHT)
            distance: Distance from object edge
            width: Width of adjacent region (None = same as object)
            height: Height of adjacent region (None = same as object)

        Returns:
            Self for fluent interface
        """
        self.strategy = SearchStrategy.ADJACENT
        self.adjacent_position = position
        self.adjacent_distance = distance
        self.adjacent_width = width
        self.adjacent_height = height
        self._invalidate_cache()
        return self

    def set_absolute(self, region: Region) -> SearchRegionOnObject:
        """Set absolute search region.

        Args:
            region: Absolute region to search in

        Returns:
            Self for fluent interface
        """
        self.strategy = SearchStrategy.ABSOLUTE
        self.absolute_region = region
        self._invalidate_cache()
        return self

    def add_region(self, region: SearchRegionOnObject) -> SearchRegionOnObject:
        """Add additional search region.

        Args:
            region: Additional region to search

        Returns:
            Self for fluent interface
        """
        self.additional_regions.append(region)
        return self

    def add_fallback(self, region: SearchRegionOnObject) -> SearchRegionOnObject:
        """Add fallback search region.

        Args:
            region: Fallback region if primary fails

        Returns:
            Self for fluent interface
        """
        self.fallback_regions.append(region)
        return self

    def set_similarity(self, similarity: float) -> SearchRegionOnObject:
        """Set similarity threshold for searches.

        Args:
            similarity: Similarity threshold (0.0-1.0)

        Returns:
            Self for fluent interface
        """
        self.similarity = similarity
        return self

    def set_timeout(self, timeout: float) -> SearchRegionOnObject:
        """Set timeout for searches.

        Args:
            timeout: Timeout in seconds

        Returns:
            Self for fluent interface
        """
        self.timeout = timeout
        return self

    def get_search_region(self) -> SearchRegion | None:
        """Get the computed search region.

        Returns:
            SearchRegion or None if cannot compute
        """
        if self._cache_valid and self._cached_region:
            return self._cached_region

        region = self._compute_region()
        if region:
            self._cached_region = SearchRegion(
                region.x, region.y, region.width, region.height, self.similarity
            )
            self._cache_valid = True
            return self._cached_region

        return None

    def get_all_regions(self) -> list[SearchRegion]:
        """Get all search regions including additional ones.

        Returns:
            List of all search regions
        """
        regions = []

        # Primary region
        primary = self.get_search_region()
        if primary:
            regions.append(primary)

        # Additional regions
        for additional in self.additional_regions:
            region = additional.get_search_region()
            if region:
                regions.append(region)

        return regions

    def get_fallback_regions(self) -> list[SearchRegion]:
        """Get fallback search regions.

        Returns:
            List of fallback regions
        """
        regions = []
        for fallback in self.fallback_regions:
            region = fallback.get_search_region()
            if region:
                regions.append(region)
        return regions

    def _compute_region(self) -> Region | None:
        """Compute the search region based on strategy.

        Returns:
            Computed region or None
        """
        if self.strategy == SearchStrategy.ABSOLUTE:
            return self.absolute_region

        if not self.base_object:
            return None

        # Get base region from object
        base_region = self._get_base_region()
        if not base_region:
            return None

        if self.strategy == SearchStrategy.RELATIVE:
            return self._compute_relative_region(base_region)
        elif self.strategy == SearchStrategy.EXPANDED:
            return self._compute_expanded_region(base_region)
        elif self.strategy == SearchStrategy.CONTRACTED:
            return self._compute_contracted_region(base_region)
        elif self.strategy == SearchStrategy.ADJACENT:
            return self._compute_adjacent_region(base_region)

        return base_region

    def _get_base_region(self) -> Region | None:
        """Get base region from state object.

        Returns:
            Base region or None
        """
        if not self.base_object:
            return None

        # Try to get region from state object
        if hasattr(self.base_object, "search_region"):
            return self.base_object.search_region
        elif hasattr(self.base_object, "region"):
            return self.base_object.region
        elif hasattr(self.base_object, "get_region"):
            return self.base_object.get_region()

        return None

    def _compute_relative_region(self, base: Region) -> Region:
        """Compute relative region with offset.

        Args:
            base: Base region

        Returns:
            Offset region
        """
        return Region(base.x + self.offset_x, base.y + self.offset_y, base.width, base.height)

    def _compute_expanded_region(self, base: Region) -> Region:
        """Compute expanded region.

        Args:
            base: Base region

        Returns:
            Expanded region
        """
        return base.grow(self.expand_by)

    def _compute_contracted_region(self, base: Region) -> Region:
        """Compute contracted region.

        Args:
            base: Base region

        Returns:
            Contracted region
        """
        return base.grow(-abs(self.expand_by))

    def _compute_adjacent_region(self, base: Region) -> Region:
        """Compute adjacent region based on position.

        Args:
            base: Base region

        Returns:
            Adjacent region
        """
        width = self.adjacent_width or base.width
        height = self.adjacent_height or base.height

        if self.adjacent_position == Position.TOP:
            return Region(base.x, base.y - height - self.adjacent_distance, width, height)
        elif self.adjacent_position == Position.BOTTOM:
            return Region(base.x, base.bottom + self.adjacent_distance, width, height)
        elif self.adjacent_position == Position.LEFT:
            return Region(base.x - width - self.adjacent_distance, base.y, width, height)
        elif self.adjacent_position == Position.RIGHT:
            return Region(base.right + self.adjacent_distance, base.y, width, height)
        else:
            # For other positions, return offset region
            return self._compute_relative_region(base)

    def _invalidate_cache(self):
        """Invalidate cached region."""
        self._cache_valid = False
        self._cached_region = None

    def is_valid(self) -> bool:
        """Check if search region is valid.

        Returns:
            True if can compute valid region
        """
        return self.get_search_region() is not None

    def copy(self) -> SearchRegionOnObject:
        """Create a copy of this search region.

        Returns:
            New SearchRegionOnObject instance
        """
        new_region = SearchRegionOnObject(
            base_object=self.base_object,
            strategy=self.strategy,
            offset_x=self.offset_x,
            offset_y=self.offset_y,
            expand_by=self.expand_by,
            adjacent_position=self.adjacent_position,
            adjacent_distance=self.adjacent_distance,
            adjacent_width=self.adjacent_width,
            adjacent_height=self.adjacent_height,
            absolute_region=self.absolute_region,
            similarity=self.similarity,
            timeout=self.timeout,
        )
        new_region.additional_regions = self.additional_regions.copy()
        new_region.fallback_regions = self.fallback_regions.copy()
        return new_region

    def __str__(self) -> str:
        """String representation."""
        return f"SearchRegionOnObject(strategy={self.strategy.name}, base={self.base_object is not None})"

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"SearchRegionOnObject(strategy={self.strategy}, "
            f"has_base={self.base_object is not None}, "
            f"additional={len(self.additional_regions)}, "
            f"fallbacks={len(self.fallback_regions)})"
        )
