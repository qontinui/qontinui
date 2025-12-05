"""Transformation operations for Region.

Handles region modifications like growing, shrinking, offsetting, and splitting.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .region import Region


class RegionTransforms:
    """Transformation operations for regions following Single Responsibility Principle."""

    @staticmethod
    def grow(region: Region, pixels: int) -> Region:
        """Grow region by specified pixels in all directions.

        Args:
            region: Region to grow
            pixels: Number of pixels to grow

        Returns:
            New grown region
        """
        from .region import Region

        return Region(
            x=region.x - pixels,
            y=region.y - pixels,
            width=region.width + 2 * pixels,
            height=region.height + 2 * pixels,
            name=region.name,
        )

    @staticmethod
    def shrink(region: Region, pixels: int) -> Region:
        """Shrink region by specified pixels in all directions.

        Args:
            region: Region to shrink
            pixels: Number of pixels to shrink

        Returns:
            New shrunk region
        """
        return RegionTransforms.grow(region, -pixels)

    @staticmethod
    def offset(region: Region, dx: int, dy: int) -> Region:
        """Create new region with offset.

        Args:
            region: Region to offset
            dx: X offset
            dy: Y offset

        Returns:
            New offset region
        """
        from .region import Region

        return Region(
            x=region.x + dx,
            y=region.y + dy,
            width=region.width,
            height=region.height,
            name=region.name,
        )

    @staticmethod
    def split_horizontal(region: Region, parts: int) -> list[Region]:
        """Split region horizontally into equal parts.

        Args:
            region: Region to split
            parts: Number of parts

        Returns:
            List of regions
        """
        from .region import Region

        regions = []
        part_height = region.height // parts

        for i in range(parts):
            y = region.y + i * part_height
            h = part_height if i < parts - 1 else region.height - i * part_height
            regions.append(Region(x=region.x, y=y, width=region.width, height=h))

        return regions

    @staticmethod
    def split_vertical(region: Region, parts: int) -> list[Region]:
        """Split region vertically into equal parts.

        Args:
            region: Region to split
            parts: Number of parts

        Returns:
            List of regions
        """
        from .region import Region

        regions = []
        part_width = region.width // parts

        for i in range(parts):
            x = region.x + i * part_width
            w = part_width if i < parts - 1 else region.width - i * part_width
            regions.append(Region(x=x, y=region.y, width=w, height=region.height))

        return regions

    @staticmethod
    def above(region: Region, distance: int) -> Region:
        """Create a region above the given region.

        Args:
            region: Source region
            distance: Height of the new region

        Returns:
            New region above the source
        """
        from .region import Region

        return Region(
            x=region.x,
            y=region.y - distance,
            width=region.width,
            height=distance,
        )

    @staticmethod
    def below(region: Region, distance: int) -> Region:
        """Create a region below the given region.

        Args:
            region: Source region
            distance: Height of the new region

        Returns:
            New region below the source
        """
        from .region import Region

        return Region(
            x=region.x,
            y=region.y + region.height,
            width=region.width,
            height=distance,
        )

    @staticmethod
    def left_of(region: Region, distance: int) -> Region:
        """Create a region to the left of the given region.

        Args:
            region: Source region
            distance: Width of the new region

        Returns:
            New region to the left of the source
        """
        from .region import Region

        return Region(
            x=region.x - distance,
            y=region.y,
            width=distance,
            height=region.height,
        )

    @staticmethod
    def right_of(region: Region, distance: int) -> Region:
        """Create a region to the right of the given region.

        Args:
            region: Source region
            distance: Width of the new region

        Returns:
            New region to the right of the source
        """
        from .region import Region

        return Region(
            x=region.x + region.width,
            y=region.y,
            width=distance,
            height=region.height,
        )
