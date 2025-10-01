"""OverlappingGrids class - ported from Qontinui framework.

Creates two offset grids that overlap to provide comprehensive coverage of grid-like interfaces.
"""

from .grid import Grid, GridBuilder
from .region import Region


class OverlappingGrids:
    """Creates two offset grids that overlap to provide comprehensive coverage.

    Port of OverlappingGrids from Qontinui framework class.

    OverlappingGrids addresses a common challenge in grid-based GUI automation: elements
    that span cell boundaries or are positioned at cell intersections. By creating two grids
    offset by half a cell width and height, this class ensures that any element in a
    grid-based layout falls clearly within at least one cell, improving targeting accuracy.

    Grid configuration:
    - Primary Grid: Original grid aligned with the region
    - Inner Grid: Offset by half cell dimensions in both directions
    - Same cell dimensions for both grids
    - Inner grid is smaller due to half-cell borders

    Problem scenarios solved:
    - Icons positioned at cell boundaries
    - UI elements that span multiple cells
    - Click targets at grid intersections
    - Partially visible elements at grid edges
    - Misaligned content in imperfect grids

    Common applications:
    - Desktop icon grids with large icons
    - Game boards where pieces sit on intersections
    - Tile-based maps with overlapping elements
    - Menu grids with hover effects extending beyond cells
    - Calendar interfaces with multi-day events

    Coverage guarantee:
    - Any point in the original region is covered by at least one cell
    - Most points are covered by cells from both grids
    - Elements at boundaries are fully contained in at least one cell
    - Reduces click accuracy issues near cell edges

    Example - Desktop icons at cell boundaries:
        # Original grid might split an icon between cells
        desktop = GridBuilder() \\
            .set_region(desktop_region) \\
            .set_columns(8) \\
            .set_rows(6) \\
            .build()

        # Overlapping grids ensure each icon falls within a cell
        overlap = OverlappingGrids(desktop)
        all_cells = overlap.get_all_regions()
    """

    def __init__(self, grid: Grid):
        """Initialize with a base grid.

        Args:
            grid: Base grid to create overlapping version of
        """
        self.grid = grid

        # Create inner grid offset by half cell dimensions
        overlap_reg = Region(
            grid.region.x + grid.cell_width // 2,
            grid.region.y + grid.cell_height // 2,
            grid.region.width - grid.cell_width,
            grid.region.height - grid.cell_height,
        )

        self.inner_grid = (
            GridBuilder()
            .set_region(overlap_reg)
            .set_cell_width(grid.cell_width)
            .set_cell_height(grid.cell_height)
            .build()
        )

    def get_all_regions(self) -> list[Region]:
        """Get all regions from both grids sorted by Y coordinate.

        Returns:
            Combined list of regions from primary and inner grids
        """
        regions = list(self.grid.grid_regions)
        regions.extend(self.inner_grid.grid_regions)
        # Sort by Y coordinate for top-to-bottom processing
        regions.sort(key=lambda r: r.y)
        return regions

    def get_primary_regions(self) -> list[Region]:
        """Get regions from primary grid only.

        Returns:
            List of regions from primary grid
        """
        return self.grid.grid_regions.copy()

    def get_inner_regions(self) -> list[Region]:
        """Get regions from inner grid only.

        Returns:
            List of regions from inner grid
        """
        return self.inner_grid.grid_regions.copy()

    def get_region_count(self) -> int:
        """Get total number of regions.

        Returns:
            Combined count from both grids
        """
        return len(self.grid.grid_regions) + len(self.inner_grid.grid_regions)

    def contains_point(self, x: int, y: int) -> list[Region]:
        """Find all cells containing a point.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            List of regions containing the point
        """
        containing_regions = []

        # Check primary grid
        for region in self.grid.grid_regions:
            if region.contains_point(x, y):
                containing_regions.append(region)

        # Check inner grid
        for region in self.inner_grid.grid_regions:
            if region.contains_point(x, y):
                containing_regions.append(region)

        return containing_regions

    def get_best_cell_for_point(self, x: int, y: int) -> Region:
        """Get the cell that best contains a point (most centered).

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Region that best contains the point
        """
        cells = self.contains_point(x, y)
        if not cells:
            # Return nearest cell if point is outside all cells
            all_regions = self.get_all_regions()
            return min(all_regions, key=lambda r: r.distance_from_center(x, y))

        # Return cell where point is most centered
        return min(cells, key=lambda r: r.distance_from_center(x, y))

    def print(self) -> None:
        """Print grid information for debugging."""
        g = self.grid.region
        i = self.inner_grid.region
        print(f"cell w.h = {self.grid.cell_width}.{self.grid.cell_height}")
        print(f"main grid x.y.w.h = {g.x}.{g.y}.{g.width}.{g.height}")
        print(f"inner grid x.y.w.h = {i.x}.{i.y}.{i.width}.{i.height}")
        print(f"main grid cells: {len(self.grid.grid_regions)}")
        print(f"inner grid cells: {len(self.inner_grid.grid_regions)}")
        print(f"total cells: {self.get_region_count()}")

    def __str__(self) -> str:
        """String representation."""
        return (
            f"OverlappingGrids(primary={self.grid.rows}x{self.grid.cols}, "
            f"inner={self.inner_grid.rows}x{self.inner_grid.cols}, "
            f"total_cells={self.get_region_count()})"
        )

    def __repr__(self) -> str:
        """Developer representation."""
        return f"OverlappingGrids(grid={self.grid!r}, " f"inner_grid={self.inner_grid!r})"
