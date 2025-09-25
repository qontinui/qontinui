"""Positions class - ported from Qontinui framework.

Defines standard relative positions within a rectangular area.
"""

from enum import Enum


class PositionName(Enum):
    """Standard position names within a rectangle.

    Port of Positions from Qontinui framework.Name enum.
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

    Key features:
    - Relative Coordinates: All positions use 0.0-1.0 scale for universal applicability
    - Nine-point Grid: Covers the most commonly needed reference points
    - Immutable Mapping: Position definitions are fixed and thread-safe
    - Type Safety: Enum-based names prevent invalid position references
    """

    # Static position mappings
    _positions: dict[PositionName, tuple[float, float]] = {
        PositionName.TOPLEFT: (0.0, 0.0),
        PositionName.TOPMIDDLE: (0.5, 0.0),
        PositionName.TOPRIGHT: (1.0, 0.0),
        PositionName.MIDDLELEFT: (0.0, 0.5),
        PositionName.MIDDLEMIDDLE: (0.5, 0.5),
        PositionName.MIDDLERIGHT: (1.0, 0.5),
        PositionName.BOTTOMLEFT: (0.0, 1.0),
        PositionName.BOTTOMMIDDLE: (0.5, 1.0),
        PositionName.BOTTOMRIGHT: (1.0, 1.0),
    }

    @classmethod
    def get_coordinates(cls, position: PositionName) -> tuple[float, float]:
        """Get relative coordinates for a position.

        Args:
            position: Position name

        Returns:
            Tuple of (x, y) relative coordinates (0.0-1.0)
        """
        return cls._positions[position]

    @classmethod
    def get_x(cls, position: PositionName) -> float:
        """Get relative X coordinate for a position.

        Args:
            position: Position name

        Returns:
            Relative X coordinate (0.0-1.0)
        """
        return cls._positions[position][0]

    @classmethod
    def get_y(cls, position: PositionName) -> float:
        """Get relative Y coordinate for a position.

        Args:
            position: Position name

        Returns:
            Relative Y coordinate (0.0-1.0)
        """
        return cls._positions[position][1]

    @classmethod
    def is_corner(cls, position: PositionName) -> bool:
        """Check if position is a corner.

        Args:
            position: Position to check

        Returns:
            True if position is a corner
        """
        return position in [
            PositionName.TOPLEFT,
            PositionName.TOPRIGHT,
            PositionName.BOTTOMLEFT,
            PositionName.BOTTOMRIGHT,
        ]

    @classmethod
    def is_edge(cls, position: PositionName) -> bool:
        """Check if position is on an edge (not corner or center).

        Args:
            position: Position to check

        Returns:
            True if position is on an edge
        """
        return position in [
            PositionName.TOPMIDDLE,
            PositionName.MIDDLELEFT,
            PositionName.MIDDLERIGHT,
            PositionName.BOTTOMMIDDLE,
        ]

    @classmethod
    def is_center(cls, position: PositionName) -> bool:
        """Check if position is the center.

        Args:
            position: Position to check

        Returns:
            True if position is the center
        """
        return position == PositionName.MIDDLEMIDDLE

    @classmethod
    def apply_to_region(
        cls, position: PositionName, x: int, y: int, width: int, height: int
    ) -> tuple[int, int]:
        """Apply position to a specific region.

        Args:
            position: Position to apply
            x: Region X coordinate
            y: Region Y coordinate
            width: Region width
            height: Region height

        Returns:
            Tuple of absolute (x, y) coordinates
        """
        rel_x, rel_y = cls.get_coordinates(position)
        abs_x = int(x + width * rel_x)
        abs_y = int(y + height * rel_y)
        return (abs_x, abs_y)

    @classmethod
    def from_string(cls, name: str) -> PositionName:
        """Get position from string name.

        Args:
            name: Position name as string

        Returns:
            PositionName enum value

        Raises:
            ValueError: If name is not valid
        """
        try:
            return PositionName[name.upper()]
        except KeyError as e:
            raise ValueError(f"Invalid position name: {name}") from e
