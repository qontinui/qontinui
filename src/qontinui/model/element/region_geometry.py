"""Geometric operations for Region.

Handles spatial relationships between regions.
"""


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .location import Location
    from .region import Region


class RegionGeometry:
    """Geometric operations for regions following Single Responsibility Principle."""

    @staticmethod
    def contains(region: Region, point: Location | tuple[int, int]) -> bool:
        """Check if a point is inside a region.

        Args:
            region: Region to check
            point: Location or (x, y) tuple

        Returns:
            True if point is inside
        """
        if isinstance(point, tuple):
            px, py = point
        else:
            px, py = point.x, point.y

        return (
            region.x <= px <= region.x + region.width and region.y <= py <= region.y + region.height
        )

    @staticmethod
    def contains_point(region: Region, x: int, y: int) -> bool:
        """Check if a point is inside a region.

        Args:
            region: Region to check
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is inside region
        """
        return region.x <= x < region.x + region.width and region.y <= y < region.y + region.height

    @staticmethod
    def overlaps(region1: Region, region2: Region) -> bool:
        """Check if two regions overlap.

        Args:
            region1: First region
            region2: Second region

        Returns:
            True if regions overlap
        """
        return not (
            region1.x + region1.width < region2.x
            or region2.x + region2.width < region1.x
            or region1.y + region1.height < region2.y
            or region2.y + region2.height < region1.y
        )

    @staticmethod
    def intersection(region1: Region, region2: Region) -> Region | None:
        """Get intersection of two regions.

        Args:
            region1: First region
            region2: Second region

        Returns:
            Intersection region or None if no overlap
        """
        if not RegionGeometry.overlaps(region1, region2):
            return None

        from .region import Region

        x1 = max(region1.x, region2.x)
        y1 = max(region1.y, region2.y)
        x2 = min(region1.x + region1.width, region2.x + region2.width)
        y2 = min(region1.y + region1.height, region2.y + region2.height)

        return Region.from_bounds(x1, y1, x2, y2)

    @staticmethod
    def union(region1: Region, region2: Region) -> Region:
        """Get union of two regions.

        Args:
            region1: First region
            region2: Second region

        Returns:
            Union region containing both
        """
        from .region import Region

        x1 = min(region1.x, region2.x)
        y1 = min(region1.y, region2.y)
        x2 = max(region1.x + region1.width, region2.x + region2.width)
        y2 = max(region1.y + region1.height, region2.y + region2.height)

        return Region.from_bounds(x1, y1, x2, y2)

    @staticmethod
    def distance_to(region1: Region, region2: Region) -> float:
        """Calculate distance between two regions.

        Args:
            region1: First region
            region2: Second region

        Returns:
            Distance between closest edges, 0 if overlapping
        """
        import math

        # If regions overlap, distance is 0
        if RegionGeometry.overlaps(region1, region2):
            return 0.0

        # Calculate horizontal distance
        if region1.x + region1.width < region2.x:
            dx = region2.x - (region1.x + region1.width)
        elif region2.x + region2.width < region1.x:
            dx = region1.x - (region2.x + region2.width)
        else:
            dx = 0

        # Calculate vertical distance
        if region1.y + region1.height < region2.y:
            dy = region2.y - (region1.y + region1.height)
        elif region2.y + region2.height < region1.y:
            dy = region1.y - (region2.y + region2.height)
        else:
            dy = 0

        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def distance_from_center(region: Region, x: int, y: int) -> float:
        """Calculate distance from center of region to a point.

        Args:
            region: Region to measure from
            x: X coordinate
            y: Y coordinate

        Returns:
            Distance from center to point
        """
        import math

        center_x = region.x + region.width // 2
        center_y = region.y + region.height // 2
        dx = center_x - x
        dy = center_y - y
        return math.sqrt(dx * dx + dy * dy)
