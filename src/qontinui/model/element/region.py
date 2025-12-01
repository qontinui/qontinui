"""Region - ported from Qontinui framework.

Represents a rectangular area on the screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from .region_factory import RegionFactory
from .region_geometry import RegionGeometry
from .region_transforms import RegionTransforms

if TYPE_CHECKING:
    from .location import Location


@dataclass
class Region:
    """Represents a rectangular area on the screen.

    Port of Region from Qontinui framework class.

    A Region is a fundamental data type that defines a rectangular area using x,y coordinates
    for the top-left corner and width,height dimensions. It serves as the spatial foundation for
    GUI element location and interaction in the model-based approach.

    In the context of model-based GUI automation, Regions are used to:
    - Define search areas for finding GUI elements (images, text, patterns)
    - Represent the boundaries of matched GUI elements
    - Specify areas for mouse and keyboard interactions
    - Create spatial relationships between GUI elements in States

    This class now delegates specialized operations to helper classes following
    Single Responsibility Principle:
    - RegionGeometry: Spatial relationships (contains, overlaps, intersection, union)
    - RegionTransforms: Modifications (grow, shrink, offset, split, directional)
    - RegionFactory: Construction methods (from_xywh, from_bounds, from_locations)
    """

    x: int = 0
    """X coordinate of top-left corner."""

    y: int = 0
    """Y coordinate of top-left corner."""

    width: int = 0
    """Width of the region."""

    height: int = 0
    """Height of the region."""

    name: str | None = None
    """Optional name for this region."""

    def __post_init__(self) -> None:
        """Initialize with screen dimensions if width/height are 0."""
        if self.width == 0 and self.height == 0:
            self.width, self.height = RegionFactory.get_screen_dimensions()

    # Core properties
    @property
    def right(self) -> int:
        """Get the right edge x-coordinate.

        Returns:
            X coordinate of right edge
        """
        return self.x + self.width

    @property
    def bottom(self) -> int:
        """Get the bottom edge y-coordinate.

        Returns:
            Y coordinate of bottom edge
        """
        return self.y + self.height

    @property
    def left(self) -> int:
        """Get the left edge x-coordinate (alias for x).

        Returns:
            X coordinate of left edge
        """
        return self.x

    @property
    def top(self) -> int:
        """Get the top edge y-coordinate (alias for y).

        Returns:
            Y coordinate of top edge
        """
        return self.y

    @property
    def w(self) -> int:
        """Alias for width (Brobot compatibility)."""
        return self.width

    @w.setter
    def w(self, value: int) -> None:
        self.width = value

    @property
    def h(self) -> int:
        """Alias for height (Brobot compatibility)."""
        return self.height

    @h.setter
    def h(self, value: int) -> None:
        self.height = value

    @property
    def area(self) -> int:
        """Get area of the region.

        Returns:
            Area in pixels
        """
        return self.width * self.height

    @property
    def center(self) -> Location:
        """Get center location of this region.

        Returns:
            Center location
        """
        from .location import Location

        return Location(x=self.x + self.width // 2, y=self.y + self.height // 2)

    @property
    def top_left(self) -> Location:
        """Get top-left corner location.

        Returns:
            Top-left location
        """
        from .location import Location

        return Location(x=self.x, y=self.y)

    @property
    def top_right(self) -> Location:
        """Get top-right corner location.

        Returns:
            Top-right location
        """
        from .location import Location

        return Location(x=self.x + self.width, y=self.y)

    @property
    def bottom_left(self) -> Location:
        """Get bottom-left corner location.

        Returns:
            Bottom-left location
        """
        from .location import Location

        return Location(x=self.x, y=self.y + self.height)

    @property
    def bottom_right(self) -> Location:
        """Get bottom-right corner location.

        Returns:
            Bottom-right location
        """
        from .location import Location

        return Location(x=self.x + self.width, y=self.y + self.height)

    # Factory methods (delegated to RegionFactory)
    @classmethod
    def from_xywh(cls, x: int, y: int, w: int, h: int) -> Region:
        """Create Region from x, y, width, height.

        Args:
            x: X coordinate
            y: Y coordinate
            w: Width
            h: Height

        Returns:
            New Region instance
        """
        return RegionFactory.from_xywh(x, y, w, h)

    @classmethod
    def from_bounds(cls, x1: int, y1: int, x2: int, y2: int) -> Region:
        """Create Region from bounding coordinates.

        Args:
            x1: Left x coordinate
            y1: Top y coordinate
            x2: Right x coordinate
            y2: Bottom y coordinate

        Returns:
            New Region instance
        """
        return RegionFactory.from_bounds(x1, y1, x2, y2)

    @classmethod
    def from_locations(cls, loc1: Location, loc2: Location) -> Region:
        """Create Region as bounding box of two locations.

        Args:
            loc1: First location
            loc2: Second location

        Returns:
            New Region containing both locations
        """
        return RegionFactory.from_locations(loc1, loc2)

    # Geometry operations (delegated to RegionGeometry)
    def contains(self, point: Location | tuple[int, int]) -> bool:
        """Check if a point is inside this region.

        Args:
            point: Location or (x, y) tuple

        Returns:
            True if point is inside
        """
        return RegionGeometry.contains(self, point)

    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is inside this region.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is inside region
        """
        return RegionGeometry.contains_point(self, x, y)

    def overlaps(self, other: Region) -> bool:
        """Check if this region overlaps with another.

        Args:
            other: Other region

        Returns:
            True if regions overlap
        """
        return RegionGeometry.overlaps(self, other)

    def intersection(self, other: Region) -> Region | None:
        """Get intersection with another region.

        Args:
            other: Other region

        Returns:
            Intersection region or None if no overlap
        """
        return RegionGeometry.intersection(self, other)

    def union(self, other: Region) -> Region:
        """Get union with another region.

        Args:
            other: Other region

        Returns:
            Union region containing both
        """
        return RegionGeometry.union(self, other)

    def distance_to(self, other: Region) -> float:
        """Calculate distance between this region and another.

        Args:
            other: Other region

        Returns:
            Distance between closest edges, 0 if overlapping
        """
        return RegionGeometry.distance_to(self, other)

    def distance_from_center(self, x: int, y: int) -> float:
        """Calculate distance from center of this region to a point.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Distance from center to point
        """
        return RegionGeometry.distance_from_center(self, x, y)

    # Transformation operations (delegated to RegionTransforms)
    def grow(self, pixels: int) -> Region:
        """Grow region by specified pixels in all directions.

        Args:
            pixels: Number of pixels to grow

        Returns:
            New grown region
        """
        return RegionTransforms.grow(self, pixels)

    def shrink(self, pixels: int) -> Region:
        """Shrink region by specified pixels in all directions.

        Args:
            pixels: Number of pixels to shrink

        Returns:
            New shrunk region
        """
        return RegionTransforms.shrink(self, pixels)

    def offset(self, dx: int, dy: int) -> Region:
        """Create new region with offset.

        Args:
            dx: X offset
            dy: Y offset

        Returns:
            New offset region
        """
        return RegionTransforms.offset(self, dx, dy)

    def split_horizontal(self, parts: int) -> list[Region]:
        """Split region horizontally into equal parts.

        Args:
            parts: Number of parts

        Returns:
            List of regions
        """
        return RegionTransforms.split_horizontal(self, parts)

    def split_vertical(self, parts: int) -> list[Region]:
        """Split region vertically into equal parts.

        Args:
            parts: Number of parts

        Returns:
            List of regions
        """
        return RegionTransforms.split_vertical(self, parts)

    def above(self, distance: int) -> Region:
        """Create a region above this one.

        Args:
            distance: Height of the new region

        Returns:
            New region above this one
        """
        return RegionTransforms.above(self, distance)

    def below(self, distance: int) -> Region:
        """Create a region below this one.

        Args:
            distance: Height of the new region

        Returns:
            New region below this one
        """
        return RegionTransforms.below(self, distance)

    def left_of(self, distance: int) -> Region:
        """Create a region to the left of this one.

        Args:
            distance: Width of the new region

        Returns:
            New region to the left of this one
        """
        return RegionTransforms.left_of(self, distance)

    def right_of(self, distance: int) -> Region:
        """Create a region to the right of this one.

        Args:
            distance: Width of the new region

        Returns:
            New region to the right of this one
        """
        return RegionTransforms.right_of(self, distance)

    # Utility methods
    def get_center(self) -> Location:
        """Get center location (method version of center property).

        Returns:
            Center location
        """
        return self.center

    def is_defined(self) -> bool:
        """Check if region is defined (has non-zero area).

        Returns:
            True if region has non-zero width and height
        """
        return self.width > 0 and self.height > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "name": self.name,
        }

    # Dunder methods
    def __str__(self) -> str:
        """String representation."""
        if self.name:
            return f"Region({self.name} at {self.x},{self.y} size {self.width}x{self.height})"
        return f"Region({self.x},{self.y},{self.width}x{self.height})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Region(x={self.x}, y={self.y}, width={self.width}, height={self.height})"

    def __eq__(self, other) -> bool:
        """Check equality."""
        if not isinstance(other, Region):
            return False
        return (
            self.x == other.x
            and self.y == other.y
            and self.width == other.width
            and self.height == other.height
        )

    def __lt__(self, other) -> bool:
        """Compare regions by area."""
        return cast(bool, self.area < other.area)
