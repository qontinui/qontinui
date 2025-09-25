"""Position models - ported from Qontinui framework.

Represents relative positions within rectangular areas using percentages.
"""

from dataclasses import dataclass
from enum import Enum

from ...util.common.pair import Pair


class PositionName(Enum):
    """Standard relative positions within a rectangular area.

    Port of Positions from Qontinui framework.Name enum.

    Defines semantic names for common locations within any rectangular region.
    """

    TOPLEFT = "TOPLEFT"
    TOPMIDDLE = "TOPMIDDLE"
    TOPRIGHT = "TOPRIGHT"
    MIDDLELEFT = "MIDDLELEFT"
    MIDDLEMIDDLE = "MIDDLEMIDDLE"
    MIDDLERIGHT = "MIDDLERIGHT"
    BOTTOMLEFT = "BOTTOMLEFT"
    BOTTOMMIDDLE = "BOTTOMMIDDLE"
    BOTTOMRIGHT = "BOTTOMRIGHT"


class Positions:
    """Defines standard relative positions within a rectangular area.

    Port of Positions from Qontinui framework class.

    Positions provides a standardized way to reference common locations within any rectangular
    region using semantic names instead of numeric coordinates. This abstraction is fundamental
    to Brobot's approach of making automation scripts more readable and maintainable by using
    human-understandable position references.

    Position mapping:
    - TOPLEFT: (0.0, 0.0) - Upper left corner
    - TOPMIDDLE: (0.5, 0.0) - Center of top edge
    - TOPRIGHT: (1.0, 0.0) - Upper right corner
    - MIDDLELEFT: (0.0, 0.5) - Center of left edge
    - MIDDLEMIDDLE: (0.5, 0.5) - Center of region
    - MIDDLERIGHT: (1.0, 0.5) - Center of right edge
    - BOTTOMLEFT: (0.0, 1.0) - Lower left corner
    - BOTTOMMIDDLE: (0.5, 1.0) - Center of bottom edge
    - BOTTOMRIGHT: (1.0, 1.0) - Lower right corner

    In the model-based approach, Positions enables location specifications that adapt
    automatically to different screen resolutions and element sizes.
    """

    _positions: dict[PositionName, Pair[float, float]] = {
        PositionName.TOPLEFT: Pair.of(0.0, 0.0),
        PositionName.TOPMIDDLE: Pair.of(0.5, 0.0),
        PositionName.TOPRIGHT: Pair.of(1.0, 0.0),
        PositionName.MIDDLELEFT: Pair.of(0.0, 0.5),
        PositionName.MIDDLEMIDDLE: Pair.of(0.5, 0.5),
        PositionName.MIDDLERIGHT: Pair.of(1.0, 0.5),
        PositionName.BOTTOMLEFT: Pair.of(0.0, 1.0),
        PositionName.BOTTOMMIDDLE: Pair.of(0.5, 1.0),
        PositionName.BOTTOMRIGHT: Pair.of(1.0, 1.0),
    }

    @classmethod
    def get_coordinates(cls, position: PositionName) -> Pair[float, float]:
        """Get percentage coordinates for a named position.

        Args:
            position: Position name

        Returns:
            Pair containing (percentW, percentH)
        """
        return cls._positions[position]


@dataclass
class Position:
    """Represents a relative position within a rectangular area using percentage coordinates.

    Port of Position from Qontinui framework class.

    Position provides a resolution-independent way to specify locations within regions,
    matches, or other rectangular areas. By using percentages (0.0 to 1.0) instead of absolute
    pixel coordinates, Position enables automation scripts that adapt automatically to different
    screen sizes, resolutions, and element dimensions.

    Key advantages over absolute positioning:
    - Resolution Independence: Works across different screen sizes without modification
    - Semantic Clarity: Can use named positions like TOPLEFT, CENTER, BOTTOMRIGHT
    - Dynamic Adaptation: Automatically adjusts to changing element sizes
    - Intuitive Specification: Easy to target general areas without pixel precision

    Coordinate system:
    - (0.0, 0.0) = top-left corner
    - (0.5, 0.5) = center (default)
    - (1.0, 1.0) = bottom-right corner
    - Values can exceed 0-1 range for positions outside the area

    In the model-based approach, Position enables abstract spatial reasoning that's independent
    of concrete pixel coordinates. This abstraction is crucial for creating maintainable automation
    that works across different environments and adapts to UI changes without script modifications.
    """

    percent_w: float = 0.5
    """Width percentage (0.0 = left, 1.0 = right)."""

    percent_h: float = 0.5
    """Height percentage (0.0 = top, 1.0 = bottom)."""

    @classmethod
    def from_percentages(cls, percent_w: int, percent_h: int) -> "Position":
        """Create Position from integer percentages.

        Args:
            percent_w: Width percentage (0-100)
            percent_h: Height percentage (0-100)

        Returns:
            Position instance
        """
        return cls(percent_w=percent_w / 100.0, percent_h=percent_h / 100.0)

    @classmethod
    def from_name(
        cls, position_name: PositionName, add_percent_w: float = 0.0, add_percent_h: float = 0.0
    ) -> "Position":
        """Create Position from named position with optional offset.

        Args:
            position_name: Standard position name
            add_percent_w: Width offset to add
            add_percent_h: Height offset to add

        Returns:
            Position instance
        """
        coords = Positions.get_coordinates(position_name)
        return cls(
            percent_w=coords.get_key() + add_percent_w, percent_h=coords.get_value() + add_percent_h
        )

    def add_percent_w(self, add_w: float) -> None:
        """Add to width percentage.

        Args:
            add_w: Amount to add
        """
        self.percent_w += add_w

    def add_percent_h(self, add_h: float) -> None:
        """Add to height percentage.

        Args:
            add_h: Amount to add
        """
        self.percent_h += add_h

    def multiply_percent_w(self, mult: float) -> None:
        """Multiply width percentage.

        Args:
            mult: Multiplication factor
        """
        self.percent_w *= mult

    def multiply_percent_h(self, mult: float) -> None:
        """Multiply height percentage.

        Args:
            mult: Multiplication factor
        """
        self.percent_h *= mult

    def to_tuple(self) -> tuple[float, float]:
        """Convert to tuple.

        Returns:
            (percent_w, percent_h) tuple
        """
        return (self.percent_w, self.percent_h)

    def __str__(self) -> str:
        """String representation."""
        return f"P[{self.percent_w:.1f} {self.percent_h:.1f}]"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Position(percent_w={self.percent_w}, percent_h={self.percent_h})"
