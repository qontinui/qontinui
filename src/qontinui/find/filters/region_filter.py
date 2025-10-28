"""Search region filter for spatial match filtering.

Filters matches based on whether they fall within specified search regions.
Supports multiple regions with containment checking based on match center point.
"""

from __future__ import annotations

from ...model.element import Region
from ...model.search_regions import SearchRegions
from ..match import Match
from .match_filter import MatchFilter


class RegionFilter(MatchFilter):
    """Filter matches by search region containment.

    Keeps only matches whose center point falls within at least one
    of the specified search regions. This is useful for limiting
    matches to specific areas of interest in the screenshot.

    The filter uses the match's center point (not bounding box) for
    containment testing. A match is kept if its center is inside any
    of the search regions.

    Example:
        >>> # Single region
        >>> region = Region(x=100, y=100, width=200, height=200)
        >>> filter = RegionFilter(region)
        >>> filtered = filter.filter(matches)
        >>>
        >>> # Multiple regions
        >>> regions = SearchRegions([region1, region2, region3])
        >>> filter = RegionFilter(regions)
        >>> filtered = filter.filter(matches)
    """

    def __init__(self, search_region: Region | SearchRegions) -> None:
        """Initialize region filter.

        Args:
            search_region: Single region or collection of regions to filter by.
                          Matches must have centers inside at least one region.

        Raises:
            ValueError: If search_region is None or invalid
        """
        if search_region is None:
            raise ValueError("search_region cannot be None")

        # Normalize to SearchRegions for uniform handling
        if isinstance(search_region, Region):
            self.search_regions = SearchRegions([search_region])
        elif isinstance(search_region, SearchRegions):
            self.search_regions = search_region
        else:
            raise ValueError(
                f"search_region must be Region or SearchRegions, got {type(search_region)}"
            )

        # Validate that we have at least one region
        if not self.search_regions.regions:
            raise ValueError("search_region must contain at least one region")

    def filter(self, matches: list[Match]) -> list[Match]:
        """Filter matches by search region containment.

        Args:
            matches: List of matches to filter

        Returns:
            Filtered list containing only matches whose centers fall
            within at least one search region

        Raises:
            ValueError: If a match lacks required region information
        """
        if not matches:
            return matches

        filtered_matches: list[Match] = []
        for match in matches:
            if match.region is None:
                # Skip matches without regions (cannot determine containment)
                continue

            try:
                # Get match center point
                match_center = match.center

                # Check if center is in ANY of the search regions
                for search_region in self.search_regions.regions:
                    if search_region.contains(match_center):
                        filtered_matches.append(match)
                        break  # Found in at least one region, move to next match

            except (AttributeError, ValueError) as e:
                # Match center or region contains() failed
                raise ValueError(f"Failed to check match containment: {e}") from e

        return filtered_matches
