"""Relative region positioning utilities.

Provides utilities for defining regions relative to other regions
with various spatial relationships.
"""

from enum import Enum

from ..model.element import Region


class Direction(Enum):
    """Directions for relative positioning."""

    ABOVE = "above"
    BELOW = "below"
    LEFT = "left"
    RIGHT = "right"

    ABOVE_LEFT = "above_left"
    ABOVE_RIGHT = "above_right"
    BELOW_LEFT = "below_left"
    BELOW_RIGHT = "below_right"


class RelativeRegion:
    """Utilities for relative region positioning.

    Following Brobot principles:
    - Intuitive spatial relationships
    - Support for all directions
    - Configurable gaps and alignment
    """

    @staticmethod
    def get_adjacent(
        reference: Region, direction: Direction, width: int, height: int, gap: int = 0
    ) -> Region:
        """Get a region adjacent to a reference region.

        Args:
            reference: Reference region
            direction: Direction relative to reference
            width: Width of new region
            height: Height of new region
            gap: Gap between regions

        Returns:
            New adjacent region
        """
        x, y = 0, 0

        if direction == Direction.ABOVE:
            x = reference.x
            y = reference.y - height - gap
        elif direction == Direction.BELOW:
            x = reference.x
            y = reference.y + reference.height + gap
        elif direction == Direction.LEFT:
            x = reference.x - width - gap
            y = reference.y
        elif direction == Direction.RIGHT:
            x = reference.x + reference.width + gap
            y = reference.y
        elif direction == Direction.ABOVE_LEFT:
            x = reference.x - width - gap
            y = reference.y - height - gap
        elif direction == Direction.ABOVE_RIGHT:
            x = reference.x + reference.width + gap
            y = reference.y - height - gap
        elif direction == Direction.BELOW_LEFT:
            x = reference.x - width - gap
            y = reference.y + reference.height + gap
        elif direction == Direction.BELOW_RIGHT:
            x = reference.x + reference.width + gap
            y = reference.y + reference.height + gap

        return Region(x=x, y=y, width=width, height=height)

    @staticmethod
    def get_grid_region(
        reference: Region, row: int, col: int, rows: int, cols: int, gap: int = 0
    ) -> Region:
        """Get a region in a grid layout relative to reference.

        Args:
            reference: Reference region defining grid bounds
            row: Row index (0-based)
            col: Column index (0-based)
            rows: Total number of rows
            cols: Total number of columns
            gap: Gap between grid cells

        Returns:
            Region for specified grid cell
        """
        # Calculate cell dimensions
        total_gap_width = gap * (cols - 1)
        total_gap_height = gap * (rows - 1)

        cell_width = (reference.width - total_gap_width) // cols
        cell_height = (reference.height - total_gap_height) // rows

        # Calculate position
        x = reference.x + col * (cell_width + gap)
        y = reference.y + row * (cell_height + gap)

        return Region(x=x, y=y, width=cell_width, height=cell_height)

    @staticmethod
    def get_columns(reference: Region, num_columns: int, gap: int = 0) -> list[Region]:
        """Divide a region into columns.

        Args:
            reference: Reference region to divide
            num_columns: Number of columns
            gap: Gap between columns

        Returns:
            List of column regions
        """
        columns = []

        total_gap = gap * (num_columns - 1)
        column_width = (reference.width - total_gap) // num_columns

        for i in range(num_columns):
            x = reference.x + i * (column_width + gap)
            columns.append(Region(x=x, y=reference.y, width=column_width, height=reference.height))

        return columns

    @staticmethod
    def get_rows(reference: Region, num_rows: int, gap: int = 0) -> list[Region]:
        """Divide a region into rows.

        Args:
            reference: Reference region to divide
            num_rows: Number of rows
            gap: Gap between rows

        Returns:
            List of row regions
        """
        rows = []

        total_gap = gap * (num_rows - 1)
        row_height = (reference.height - total_gap) // num_rows

        for i in range(num_rows):
            y = reference.y + i * (row_height + gap)
            rows.append(Region(x=reference.x, y=y, width=reference.width, height=row_height))

        return rows

    @staticmethod
    def align_horizontal(regions: list[Region], gap: int = 0, y: int | None = None) -> list[Region]:
        """Align regions horizontally.

        Args:
            regions: Regions to align
            gap: Gap between regions
            y: Y position (uses first region's y if None)

        Returns:
            List of aligned regions
        """
        if not regions:
            return []

        aligned = []
        current_x = regions[0].x
        y_pos = y if y is not None else regions[0].y

        for region in regions:
            aligned.append(
                Region(
                    x=current_x, y=y_pos, width=region.width, height=region.height, name=region.name
                )
            )
            current_x += region.width + gap

        return aligned

    @staticmethod
    def align_vertical(regions: list[Region], gap: int = 0, x: int | None = None) -> list[Region]:
        """Align regions vertically.

        Args:
            regions: Regions to align
            gap: Gap between regions
            x: X position (uses first region's x if None)

        Returns:
            List of aligned regions
        """
        if not regions:
            return []

        aligned = []
        current_y = regions[0].y
        x_pos = x if x is not None else regions[0].x

        for region in regions:
            aligned.append(
                Region(
                    x=x_pos, y=current_y, width=region.width, height=region.height, name=region.name
                )
            )
            current_y += region.height + gap

        return aligned

    @staticmethod
    def center_in(inner: Region, outer: Region) -> Region:
        """Center a region inside another region.

        Args:
            inner: Region to center
            outer: Container region

        Returns:
            Centered region
        """
        x = outer.x + (outer.width - inner.width) // 2
        y = outer.y + (outer.height - inner.height) // 2

        return Region(x=x, y=y, width=inner.width, height=inner.height, name=inner.name)
