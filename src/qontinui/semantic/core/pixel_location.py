"""PixelLocation - Represents complex shapes using pixel-level precision."""


from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage  # type: ignore[import-untyped]

from ...model.element.region import Region


@dataclass
class Point:
    """Simple point representation."""

    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return isinstance(other, Point) and self.x == other.x and self.y == other.y


@dataclass
class PixelLocation:
    """Represents complex shapes using pixel-level precision.

    Can represent any arbitrary shape beyond simple rectangles, using a set of pixels
    that comprise the shape. Provides methods for shape analysis, conversion to
    bounding boxes, and spatial operations.
    """

    pixels: set[Point] = field(default_factory=set)
    """Set of all pixels that comprise this shape."""

    _min_x: int | None = field(default=None, init=False)
    _min_y: int | None = field(default=None, init=False)
    _max_x: int | None = field(default=None, init=False)
    _max_y: int | None = field(default=None, init=False)
    _centroid: Point | None = field(default=None, init=False)

    def __post_init__(self):
        """Calculate bounding box coordinates after initialization."""
        self._update_bounds()

    def _update_bounds(self):
        """Update cached bounding box coordinates."""
        if not self.pixels:
            self._min_x = self._min_y = self._max_x = self._max_y = 0
            self._centroid = Point(0, 0)
            return

        xs = [p.x for p in self.pixels]
        ys = [p.y for p in self.pixels]

        self._min_x = min(xs)
        self._min_y = min(ys)
        self._max_x = max(xs)
        self._max_y = max(ys)

        # Calculate centroid
        avg_x = sum(xs) / len(xs)
        avg_y = sum(ys) / len(ys)
        self._centroid = Point(int(avg_x), int(avg_y))

    @classmethod
    def from_mask(
        cls, mask: np.ndarray[Any, Any], offset_x: int = 0, offset_y: int = 0
    ) -> PixelLocation:
        """Create PixelLocation from a boolean mask array.

        Args:
            mask: 2D boolean array where True indicates pixels in the shape
            offset_x: X offset to add to all pixel coordinates
            offset_y: Y offset to add to all pixel coordinates

        Returns:
            New PixelLocation instance
        """
        pixels = set()
        for y, row in enumerate(mask):
            for x, value in enumerate(row):
                if value:
                    pixels.add(Point(x + offset_x, y + offset_y))
        return cls(pixels=pixels)

    @classmethod
    def from_rectangle(cls, x: int, y: int, width: int, height: int) -> PixelLocation:
        """Create PixelLocation from a rectangle.

        Args:
            x: X coordinate of top-left corner
            y: Y coordinate of top-left corner
            width: Width of rectangle
            height: Height of rectangle

        Returns:
            New PixelLocation instance
        """
        pixels = set()
        for py in range(y, y + height):
            for px in range(x, x + width):
                pixels.add(Point(px, py))
        return cls(pixels=pixels)

    @classmethod
    def from_polygon(cls, vertices: list[tuple[int, int]]) -> PixelLocation:
        """Create PixelLocation from polygon vertices.

        Uses scanline algorithm to fill polygon.

        Args:
            vertices: List of (x, y) tuples defining polygon vertices

        Returns:
            New PixelLocation instance
        """
        if len(vertices) < 3:
            return cls()

        # Find bounding box
        ys = [v[1] for v in vertices]
        min_y, max_y = min(ys), max(ys)

        pixels = set()

        # Scanline fill algorithm
        for y in range(min_y, max_y + 1):
            intersections = []

            for i in range(len(vertices)):
                v1 = vertices[i]
                v2 = vertices[(i + 1) % len(vertices)]

                if v1[1] <= y < v2[1] or v2[1] <= y < v1[1]:
                    # Calculate x intersection
                    t = (y - v1[1]) / (v2[1] - v1[1])
                    x = v1[0] + t * (v2[0] - v1[0])
                    intersections.append(int(x))

            intersections.sort()

            # Fill between pairs of intersections
            for i in range(0, len(intersections), 2):
                if i + 1 < len(intersections):
                    for x in range(intersections[i], intersections[i + 1] + 1):
                        pixels.add(Point(x, y))

        return cls(pixels=pixels)

    @classmethod
    def from_circle(cls, center: tuple[int, int], radius: int) -> PixelLocation:
        """Create PixelLocation from a circle.

        Args:
            center: (x, y) tuple for circle center
            radius: Circle radius

        Returns:
            New PixelLocation instance
        """
        pixels = set()
        cx, cy = center

        # Use midpoint circle algorithm
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                    pixels.add(Point(x, y))

        return cls(pixels=pixels)

    def to_bounding_box(self) -> Region:
        """Convert to smallest Region that contains all pixels.

        Returns:
            Region that bounds all pixels in this shape
        """
        if (
            not self.pixels
            or self._min_x is None
            or self._min_y is None
            or self._max_x is None
            or self._max_y is None
        ):
            return Region(0, 0, 0, 0)

        # Type narrowing - at this point we know all bounds are not None
        min_x: int = self._min_x
        min_y: int = self._min_y
        max_x: int = self._max_x
        max_y: int = self._max_y

        return Region(
            x=min_x,
            y=min_y,
            width=max_x - min_x + 1,
            height=max_y - min_y + 1,
        )

    def get_centroid(self) -> Point:
        """Get the centroid (center of mass) of the shape.

        Returns:
            Point at the centroid of all pixels
        """
        return self._centroid or Point(0, 0)

    def contains(self, point: Point) -> bool:
        """Check if a point is within this shape.

        Args:
            point: Point to check

        Returns:
            True if point is in the shape
        """
        return point in self.pixels

    def overlaps(self, other: PixelLocation) -> bool:
        """Check if this shape overlaps with another.

        Args:
            other: Other PixelLocation to check

        Returns:
            True if shapes have any pixels in common
        """
        return bool(self.pixels & other.pixels)

    def get_overlap_percentage(self, other: PixelLocation) -> float:
        """Calculate percentage of overlap with another shape.

        Args:
            other: Other PixelLocation to compare

        Returns:
            Percentage of this shape that overlaps with other (0.0 to 1.0)
        """
        if not self.pixels:
            return 0.0

        overlap_count = len(self.pixels & other.pixels)
        return overlap_count / len(self.pixels)

    def union(self, other: PixelLocation) -> PixelLocation:
        """Create union of this shape with another.

        Args:
            other: Other PixelLocation to union with

        Returns:
            New PixelLocation containing all pixels from both shapes
        """
        return PixelLocation(pixels=self.pixels | other.pixels)

    def intersection(self, other: PixelLocation) -> PixelLocation:
        """Create intersection of this shape with another.

        Args:
            other: Other PixelLocation to intersect with

        Returns:
            New PixelLocation containing only overlapping pixels
        """
        return PixelLocation(pixels=self.pixels & other.pixels)

    def get_area(self) -> int:
        """Get the area (number of pixels) of the shape.

        Returns:
            Number of pixels in the shape
        """
        return len(self.pixels)

    def get_perimeter(self) -> int:
        """Calculate the perimeter of the shape.

        Counts pixels that have at least one neighbor outside the shape.

        Returns:
            Approximate perimeter in pixels
        """
        perimeter = 0

        for pixel in self.pixels:
            # Check 4-connected neighbors
            neighbors = [
                Point(pixel.x - 1, pixel.y),
                Point(pixel.x + 1, pixel.y),
                Point(pixel.x, pixel.y - 1),
                Point(pixel.x, pixel.y + 1),
            ]

            # Count neighbors not in shape
            for neighbor in neighbors:
                if neighbor not in self.pixels:
                    perimeter += 1
                    break  # Only count pixel once

        return perimeter

    def get_compactness(self) -> float:
        """Measure how compact (round) the shape is.

        Returns value between 0 and 1, where 1 is a perfect circle.

        Returns:
            Compactness measure (0.0 to 1.0)
        """
        area = self.get_area()
        if area == 0:
            return 0.0

        perimeter = self.get_perimeter()
        if perimeter == 0:
            return 0.0

        # Use isoperimetric quotient
        return (4 * math.pi * area) / (perimeter**2)

    def get_contour(self) -> list[Point]:
        """Get ordered list of boundary pixels.

        Returns:
            List of Points forming the contour
        """
        if not self.pixels:
            return []

        # Convert to numpy array for contour finding
        min_x = self._min_x or 0
        min_y = self._min_y or 0
        max_x = self._max_x or 0
        max_y = self._max_y or 0

        width = max_x - min_x + 1
        height = max_y - min_y + 1

        mask = np.zeros((height, width), dtype=bool)
        for pixel in self.pixels:
            mask[pixel.y - min_y, pixel.x - min_x] = True

        # Find contour pixels using edge detection
        edges = mask & ~ndimage.binary_erosion(mask)

        contour = []
        for y in range(height):
            for x in range(width):
                if edges[y, x]:
                    contour.append(Point(x + min_x, y + min_y))

        return contour

    def translate(self, dx: int, dy: int) -> PixelLocation:
        """Create translated copy of this shape.

        Args:
            dx: X translation
            dy: Y translation

        Returns:
            New translated PixelLocation
        """
        new_pixels = {Point(p.x + dx, p.y + dy) for p in self.pixels}
        return PixelLocation(pixels=new_pixels)

    def scale(self, factor: float, center: Point | None = None) -> PixelLocation:
        """Create scaled copy of this shape.

        Args:
            factor: Scale factor (>1 enlarges, <1 shrinks)
            center: Center point for scaling (defaults to centroid)

        Returns:
            New scaled PixelLocation
        """
        if center is None:
            center = self.get_centroid()

        new_pixels = set()
        for pixel in self.pixels:
            # Translate to origin, scale, translate back
            dx = pixel.x - center.x
            dy = pixel.y - center.y
            new_x = int(center.x + dx * factor)
            new_y = int(center.y + dy * factor)
            new_pixels.add(Point(new_x, new_y))

        return PixelLocation(pixels=new_pixels)
