"""Grid class - ported from Qontinui framework.

Divides a screen region into a matrix of cells for systematic interaction.
"""

from dataclasses import dataclass, field

from .region import Region


@dataclass
class Grid:
    """Divides a screen region into a matrix of cells for systematic interaction.

    Port of Grid from Qontinui framework class.

    Grid provides a structured way to interact with regularly arranged GUI elements
    such as icon grids, table cells, calendar dates, or tile-based interfaces. By
    dividing a region into rows and columns, Grid enables precise targeting of
    individual elements within repetitive layouts without defining each element
    separately.

    Definition modes:
    - By Dimensions: Specify number of rows and columns
    - By Cell Size: Specify cell width and height
    - Hybrid approach with intelligent remainder handling

    Remainder handling strategies:
    - Adjust Region: Shrink region to fit exact grid (adjust_region_to_grids=True)
    - Expand Last Cells: Rightmost and bottom cells absorb extra space
    - Smart Expansion: Add extra row/column if remainder > half cell size

    Common use cases:
    - Desktop icon grids and application launchers
    - Calendar date selection (7 columns × n rows)
    - Spreadsheet or table cell navigation
    - Game boards (chess, tic-tac-toe, etc.)
    - Photo galleries and thumbnail grids
    - Virtual keyboards and keypads
    """

    region: Region
    cell_width: int
    cell_height: int
    rows: int
    cols: int
    grid_regions: list[Region] = field(default_factory=list)

    def print(self) -> None:
        """Print grid structure for debugging."""
        print(f"region = {self.region}")
        for r in range(self.rows):
            for c in range(self.cols):
                reg = self.grid_regions[self.cols * r + c]
                print(f"{reg.x}.{reg.y}_{reg.width}.{reg.height} ", end="")
            print()

    def get_cell(self, row: int, col: int) -> Region | None:
        """Get region for specific cell.

        Args:
            row: Row index (0-based)
            col: Column index (0-based)

        Returns:
            Region at specified position or None if out of bounds
        """
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return None
        return self.grid_regions[self.cols * row + col]

    def get_cell_by_index(self, index: int) -> Region | None:
        """Get region by linear index.

        Args:
            index: Linear index in grid (0-based)

        Returns:
            Region at index or None if out of bounds
        """
        if index < 0 or index >= len(self.grid_regions):
            return None
        return self.grid_regions[index]

    @classmethod
    def builder(cls) -> "GridBuilder":
        """Create a new GridBuilder.

        Returns:
            New GridBuilder instance
        """
        return GridBuilder()


class GridBuilder:
    """Builder for Grid class.

    Port of Grid from Qontinui framework.Builder.
    """

    def __init__(self):
        """Initialize builder with default values."""
        self.region: Region | None = None
        self.cell_width: int = 0
        self.cell_height: int = 0
        self.rows: int = 0
        self.cols: int = 0
        self.adjust_region_to_grids: bool = False
        self.grid_regions: list[Region] = []

    def set_region(self, region: Region) -> "GridBuilder":
        """Set the region to divide into grid (fluent).

        Args:
            region: Region to divide

        Returns:
            Self for chaining
        """
        self.region = region
        return self

    def set_cell_width(self, cell_width: int) -> "GridBuilder":
        """Set cell width (fluent).

        Args:
            cell_width: Width of each cell

        Returns:
            Self for chaining
        """
        self.cell_width = cell_width
        return self

    def set_cell_height(self, cell_height: int) -> "GridBuilder":
        """Set cell height (fluent).

        Args:
            cell_height: Height of each cell

        Returns:
            Self for chaining
        """
        self.cell_height = cell_height
        return self

    def set_rows(self, rows: int) -> "GridBuilder":
        """Set number of rows (fluent).

        Args:
            rows: Number of rows

        Returns:
            Self for chaining
        """
        self.rows = rows
        return self

    def set_columns(self, cols: int) -> "GridBuilder":
        """Set number of columns (fluent).

        Args:
            cols: Number of columns

        Returns:
            Self for chaining
        """
        self.cols = cols
        return self

    def set_adjust_region_to_grids(self, adjust: bool) -> "GridBuilder":
        """Set whether to adjust region to fit grids exactly (fluent).

        Args:
            adjust: True to shrink region to fit grids

        Returns:
            Self for chaining
        """
        self.adjust_region_to_grids = adjust
        return self

    def _set_grid_regions(self) -> None:
        """Calculate and create grid regions."""
        if self.region is None:
            self.region = Region(0, 0, 0, 0)

        # Calculate dimensions based on what's provided
        if self.cell_height > 0 and self.cell_width > 0:
            # Width and height are defined - calculate rows/cols
            self.cols = self.region.width // self.cell_width
            self.rows = self.region.height // self.cell_height
        else:
            # Use rows and columns - calculate cell size
            if self.cols > 0:
                self.cell_width = self.region.width // self.cols
            if self.rows > 0:
                self.cell_height = self.region.height // self.rows

        # Ensure at least 1 row and column
        self.cols = max(self.cols, 1)
        self.rows = max(self.rows, 1)

        # Adjust region if requested
        if self.adjust_region_to_grids:
            self.region = Region(
                self.region.x,
                self.region.y,
                self.cols * self.cell_width,
                self.rows * self.cell_height,
            )

        # Calculate remainder pixels
        w_remainder = self.region.width - self.cols * self.cell_width
        h_remainder = self.region.height - self.rows * self.cell_height

        # Add extra column/row if remainder is > half cell size
        if w_remainder > self.cell_width // 2:
            self.cols += 1
        if h_remainder > self.cell_height // 2:
            self.rows += 1

        # Calculate size of rightmost and bottommost cells (may be different)
        rightmost_cell_width = self.region.width - (self.cols - 1) * self.cell_width
        bottommost_cell_height = self.region.height - (self.rows - 1) * self.cell_height

        x = self.region.x
        y = self.region.y

        # Create grid regions
        # All rows except the last
        for r in range(self.rows - 1):
            # All columns except the last
            for c in range(self.cols - 1):
                self.grid_regions.append(
                    Region(
                        x + c * self.cell_width,
                        y + r * self.cell_height,
                        self.cell_width,
                        self.cell_height,
                    )
                )
            # Last column (might be wider)
            self.grid_regions.append(
                Region(
                    x + (self.cols - 1) * self.cell_width,
                    y + r * self.cell_height,
                    rightmost_cell_width,
                    self.cell_height,
                )
            )

        # Bottom row
        for c in range(self.cols - 1):
            self.grid_regions.append(
                Region(
                    x + c * self.cell_width,
                    y + (self.rows - 1) * self.cell_height,
                    self.cell_width,
                    bottommost_cell_height,
                )
            )

        # Bottom-right cell (might be both wider and taller)
        self.grid_regions.append(
            Region(
                x + (self.cols - 1) * self.cell_width,
                y + (self.rows - 1) * self.cell_height,
                rightmost_cell_width,
                bottommost_cell_height,
            )
        )

    def build(self) -> Grid:
        """Build the Grid instance.

        Returns:
            Configured Grid instance
        """
        if self.region is None:
            raise ValueError("Region must be set before building Grid")
        self._set_grid_regions()
        return Grid(
            region=self.region,
            cell_width=self.cell_width,
            cell_height=self.cell_height,
            rows=self.rows,
            cols=self.cols,
            grid_regions=self.grid_regions,
        )
