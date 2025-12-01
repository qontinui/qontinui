"""Search regions - ported from Brobot framework.

Manages multiple search areas for pattern matching, supporting both
dynamic and fixed location patterns.
"""

from dataclasses import dataclass, field
from typing import Optional

from .element.region import Region


@dataclass
class SearchRegions:
    """Container for defining screen regions to search within.

    Port of SearchRegions from Brobot framework. This class provides sophisticated
    control over where patterns are searched for on the screen, supporting both
    dynamic and fixed location patterns.

    Key features:
    - Multiple Regions: Supports searching in multiple rectangular areas
    - Fixed Region: Special handling for patterns that always appear in the same location
    - Dynamic Selection: Can select from available regions for load distribution

    Attributes:
        regions: List of regions to search within
        fixed_region: The fixed region for patterns that always appear in the same location
    """

    regions: list[Region] = field(default_factory=list)
    """The list of regions to search within."""

    fixed_region: Region | None = None
    """The fixed region is defined when an image with a fixed location is found.
    This region is then used in future FIND operations. Initialized to None to
    distinguish between 'not set' and 'explicitly set to a region'."""

    def __init__(self, other: Optional["SearchRegions | list[Region]"] = None) -> None:
        """Initialize search regions.

        Args:
            other: Another SearchRegions instance to copy from, or a list of Regions to use
        """
        if other:
            if isinstance(other, list):
                self.regions = list(other)
                self.fixed_region = None
            elif isinstance(other, SearchRegions):
                self.regions = [
                    Region(x=r.x, y=r.y, width=r.width, height=r.height)
                    for r in other.regions
                ]
                self.fixed_region = (
                    Region(
                        x=other.fixed_region.x,
                        y=other.fixed_region.y,
                        width=other.fixed_region.width,
                        height=other.fixed_region.height,
                    )
                    if other.fixed_region
                    else None
                )
            else:
                self.regions = []
                self.fixed_region = None
        else:
            self.regions = []
            self.fixed_region = None

    def is_fixed_region_set(self) -> bool:
        """Check if the fixed region has been set.

        Returns:
            True if fixed region is defined, False otherwise
        """
        return self.fixed_region is not None

    def reset_fixed_region(self) -> None:
        """Reset the fixed region, allowing the pattern to be found anywhere."""
        self.fixed_region = None

    def set_fixed_region(self, region: Region) -> None:
        """Set the fixed region for this pattern.

        Args:
            region: The fixed region where this pattern is always found
        """
        self.fixed_region = region

    def add_search_regions(self, *regions: Region) -> "SearchRegions":
        """Add one or more regions to the search areas.

        Args:
            *regions: The region(s) to add

        Returns:
            This instance for method chaining
        """
        for region in regions:
            if region is not None:
                self.regions.append(region)
        return self

    def add_region(self, region: Region) -> "SearchRegions":
        """Add a region to the search areas.

        Args:
            region: The region to add

        Returns:
            This instance for method chaining
        """
        return self.add_search_regions(region)

    def set_regions(self, regions: list[Region]) -> None:
        """Set the search regions.

        Args:
            regions: List of regions to use for searching
        """
        self.regions = list(regions)

    def get_regions(self, fixed: bool) -> list[Region]:
        """Get the search regions based on whether the pattern has a fixed location.

        When the pattern has a fixed location:
        - If a fixed region is defined, it is returned
        - Otherwise, all search regions are returned

        Args:
            fixed: Whether the pattern has a fixed location

        Returns:
            List of regions to search
        """
        if not fixed:
            return list(self.regions)

        # Only return fixed region if it's been explicitly set
        if self.fixed_region is not None:
            return [self.fixed_region]

        return list(self.regions)

    def get_fixed_if_defined_or_random_region(self, fixed: bool) -> Region:
        """Get the fixed region if defined, otherwise return a random search region.

        Args:
            fixed: Whether the pattern has a fixed location

        Returns:
            A search region
        """
        import random

        region_list = self.get_regions(fixed)
        if not region_list:
            return Region()  # Full screen default
        return random.choice(region_list)

    def get_one_region(self) -> Region:
        """Return the fixed region if defined, otherwise the first defined region.

        Returns:
            One region, or a full-screen region if none are defined
        """
        region_list = self.get_regions(True)
        if len(region_list) == 1:
            return region_list[0]
        for region in region_list:
            if region is not None:
                return region
        return Region()  # Full screen default

    def get_regions_for_search(self) -> list[Region]:
        """Get regions for searching, with full-screen default if none are configured.

        Returns:
            List of search regions (never empty)
        """
        if self.fixed_region is not None:
            return [self.fixed_region]
        if self.regions:
            return list(self.regions)
        return [Region()]  # Full screen default

    def clear(self) -> "SearchRegions":
        """Clear all search regions.

        Returns:
            This instance for method chaining
        """
        self.regions.clear()
        return self

    def is_empty(self) -> bool:
        """Check if there are no search regions defined.

        Returns:
            True if no regions are defined, False otherwise
        """
        return len(self.regions) == 0 and self.fixed_region is None

    def is_defined(self, fixed: bool) -> bool:
        """Check if regions are defined based on whether pattern has fixed position.

        Args:
            fixed: Whether the pattern has a fixed position

        Returns:
            True if fixed and fixed region is set, or if not fixed and any region is defined
        """
        if self.fixed_region is not None:
            return True
        if fixed:
            return False  # Should be fixed but the region is not defined
        return len(self.regions) > 0

    def get_fixed_region(self) -> Region | None:
        """Get the fixed region if defined.

        Returns:
            Fixed region if available, None otherwise
        """
        return self.fixed_region
