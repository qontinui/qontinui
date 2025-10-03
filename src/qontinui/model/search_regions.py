"""Search regions - ported from Qontinui framework.

Defines regions to search within.
"""

from dataclasses import dataclass, field
from typing import Optional

from .element.region import Region


@dataclass
class SearchRegions:
    """Container for defining screen regions to search within.

    Port of SearchRegions from Qontinui framework class.

    This class encapsulates one or more regions that constrain where
    find operations should look for matches. By default, the entire
    screen is searched, but limiting the search area can improve both
    performance and accuracy.
    """

    regions: list[Region] = field(default_factory=list)
    """The list of regions to search within."""

    def __init__(self, other: Optional["SearchRegions | list[Region]"] = None):
        """Initialize search regions.

        Args:
            other: Another SearchRegions instance to copy from, or a list of Regions to use
        """
        if other:
            if isinstance(other, list):
                self.regions = other
            else:
                self.regions = [
                    Region(x=r.x, y=r.y, width=r.width, height=r.height) for r in other.regions
                ]
        else:
            self.regions = []

    def add_region(self, region: Region) -> "SearchRegions":
        """Add a region to the search areas.

        Args:
            region: The region to add

        Returns:
            This instance for method chaining
        """
        self.regions.append(region)
        return self

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
        return len(self.regions) == 0

    def get_fixed_region(self) -> Region | None:
        """Get the first region if defined.

        Returns:
            First region if available, None otherwise
        """
        return self.regions[0] if self.regions else None
