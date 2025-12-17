"""Region collection builder for ObjectCollection.

Handles Regions and Locations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...model.element import Region
    from ...model.state import StateLocation, StateRegion


class RegionCollectionBuilder:
    """Builder for region-related objects in ObjectCollection.

    Handles:
    - Regions
    - StateRegions
    - Locations
    - StateLocations
    - Grid subregions
    """

    def __init__(self) -> None:
        """Initialize builder with empty lists."""
        self.state_locations: list[StateLocation] = []
        self.state_regions: list[StateRegion] = []

    def with_locations(self, *locations) -> RegionCollectionBuilder:
        """Add locations to collection.

        Args:
            locations: Variable number of Location or StateLocation objects

        Returns:
            This builder for method chaining
        """
        from ...model.element import Location
        from ...model.state import StateLocation

        for location in locations:
            if isinstance(location, Location):
                # Convert Location to StateLocation
                state_location = StateLocation(location=location, name=location.name)
                self.state_locations.append(state_location)
            elif isinstance(location, StateLocation):
                self.state_locations.append(location)
        return self

    def set_locations(self, locations: list[StateLocation]) -> RegionCollectionBuilder:
        """Set locations list.

        Args:
            locations: List of StateLocation objects

        Returns:
            This builder for method chaining
        """
        self.state_locations = locations
        return self

    def with_regions(self, *regions) -> RegionCollectionBuilder:
        """Add regions to collection.

        Args:
            regions: Variable number of Region or StateRegion objects

        Returns:
            This builder for method chaining
        """
        from ...model.element import Region
        from ...model.state import StateRegion

        for region in regions:
            if isinstance(region, Region):
                # Convert Region to StateRegion
                state_region = StateRegion(region=region, name=getattr(region, "name", None))
                self.state_regions.append(state_region)
            elif isinstance(region, StateRegion):
                self.state_regions.append(region)
        return self

    def set_regions(self, regions: list[StateRegion]) -> RegionCollectionBuilder:
        """Set regions list.

        Args:
            regions: List of StateRegion objects

        Returns:
            This builder for method chaining
        """
        self.state_regions = regions
        return self

    def with_grid_subregions(self, rows: int, columns: int, *regions) -> RegionCollectionBuilder:
        """Add grid subregions from regions.

        Args:
            rows: Number of rows in grid
            columns: Number of columns in grid
            regions: Variable number of Region or StateRegion objects

        Returns:
            This builder for method chaining
        """
        from ...model.element import Region
        from ...model.state import StateRegion

        for region in regions:
            if isinstance(region, Region):
                # Split region into grid
                grid_regions = self._create_grid_regions(region, rows, columns)
                for grid_region in grid_regions:
                    state_region = StateRegion(region=grid_region)
                    self.state_regions.append(state_region)
            elif isinstance(region, StateRegion):
                # Split state region's underlying region into grid
                grid_regions = self._create_grid_regions(region.get_search_region(), rows, columns)
                for grid_region in grid_regions:
                    state_region = StateRegion(region=grid_region)
                    self.state_regions.append(state_region)
        return self

    def _create_grid_regions(self, region: Region, rows: int, columns: int) -> list[Region]:
        """Create grid of subregions from a region.

        Args:
            region: Region to split
            rows: Number of rows
            columns: Number of columns

        Returns:
            List of grid subregions
        """
        from ...model.element import Region

        grid_regions = []
        cell_width = region.width // columns
        cell_height = region.height // rows

        for row in range(rows):
            for col in range(columns):
                x = region.x + (col * cell_width)
                y = region.y + (row * cell_height)
                grid_region = Region(x=x, y=y, width=cell_width, height=cell_height)
                grid_regions.append(grid_region)

        return grid_regions

    def build(self) -> tuple[list[StateLocation], list[StateRegion]]:
        """Build and return the locations and regions lists.

        Returns:
            Tuple of (state_locations, state_regions) copies
        """
        return self.state_locations.copy(), self.state_regions.copy()
