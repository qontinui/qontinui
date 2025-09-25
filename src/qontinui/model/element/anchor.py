"""Anchor - defines a reference point for relative positioning.

Anchors enable flexible positioning by defining named reference points
that can be used to position elements relative to each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .region import Region


class AnchorType(Enum):
    """Types of anchor points."""

    TOP_LEFT = auto()
    TOP_CENTER = auto()
    TOP_RIGHT = auto()
    MIDDLE_LEFT = auto()
    CENTER = auto()
    MIDDLE_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_CENTER = auto()
    BOTTOM_RIGHT = auto()
    CUSTOM = auto()


@dataclass
class Anchor:
    """Defines a reference point for relative positioning.

    An Anchor represents a specific point that can be used as a reference
    for positioning other elements. This enables creating flexible layouts
    that adapt to different screen sizes and resolutions.

    Anchors can be:
    - Predefined positions (corners, edges, center)
    - Custom positions with specific coordinates
    - Named reference points for semantic positioning

    In the model-based approach, anchors are essential for:
    - Defining spatial relationships between GUI elements
    - Creating responsive layouts that adapt to screen changes
    - Establishing consistent positioning rules across states
    """

    name: str
    """Name identifier for this anchor."""

    anchor_type: AnchorType = AnchorType.CENTER
    """Type of anchor point."""

    x: int = 0
    """X coordinate of the anchor point."""

    y: int = 0
    """Y coordinate of the anchor point."""

    reference_element: str | None = None
    """Optional reference to another element this anchor is relative to."""

    offset_x: int = 0
    """X offset from the anchor point."""

    offset_y: int = 0
    """Y offset from the anchor point."""

    def __str__(self) -> str:
        """String representation of the anchor."""
        if self.reference_element:
            return f"Anchor({self.name} at {self.anchor_type.name} of {self.reference_element})"
        return f"Anchor({self.name} at ({self.x}, {self.y}))"

    def get_position(self, region: Region | None = None) -> tuple[int, int]:
        """Calculate the actual position of this anchor.

        Args:
            region: Optional region to calculate position within

        Returns:
            Tuple of (x, y) coordinates
        """
        if region is None:
            return (self.x + self.offset_x, self.y + self.offset_y)

        # Calculate position based on anchor type within region
        base_x, base_y = self._calculate_base_position(region)
        return (base_x + self.offset_x, base_y + self.offset_y)

    def _calculate_base_position(self, region: Region) -> tuple[int, int]:
        """Calculate base position within a region based on anchor type.

        Args:
            region: Region to calculate position within

        Returns:
            Tuple of (x, y) base coordinates
        """
        x, y = region.x, region.y
        w, h = region.width, region.height

        if self.anchor_type == AnchorType.TOP_LEFT:
            return (x, y)
        elif self.anchor_type == AnchorType.TOP_CENTER:
            return (x + w // 2, y)
        elif self.anchor_type == AnchorType.TOP_RIGHT:
            return (x + w, y)
        elif self.anchor_type == AnchorType.MIDDLE_LEFT:
            return (x, y + h // 2)
        elif self.anchor_type == AnchorType.CENTER:
            return (x + w // 2, y + h // 2)
        elif self.anchor_type == AnchorType.MIDDLE_RIGHT:
            return (x + w, y + h // 2)
        elif self.anchor_type == AnchorType.BOTTOM_LEFT:
            return (x, y + h)
        elif self.anchor_type == AnchorType.BOTTOM_CENTER:
            return (x + w // 2, y + h)
        elif self.anchor_type == AnchorType.BOTTOM_RIGHT:
            return (x + w, y + h)
        else:  # CUSTOM
            return (self.x, self.y)

    @classmethod
    def center(cls, name: str = "center") -> Anchor:
        """Create a center anchor.

        Args:
            name: Name for the anchor

        Returns:
            New Anchor at center position
        """
        return cls(name=name, anchor_type=AnchorType.CENTER)

    @classmethod
    def top_left(cls, name: str = "top_left") -> Anchor:
        """Create a top-left anchor.

        Args:
            name: Name for the anchor

        Returns:
            New Anchor at top-left position
        """
        return cls(name=name, anchor_type=AnchorType.TOP_LEFT)

    @classmethod
    def bottom_right(cls, name: str = "bottom_right") -> Anchor:
        """Create a bottom-right anchor.

        Args:
            name: Name for the anchor

        Returns:
            New Anchor at bottom-right position
        """
        return cls(name=name, anchor_type=AnchorType.BOTTOM_RIGHT)

    @classmethod
    def custom(cls, name: str, x: int, y: int) -> Anchor:
        """Create a custom anchor at specific coordinates.

        Args:
            name: Name for the anchor
            x: X coordinate
            y: Y coordinate

        Returns:
            New Anchor at custom position
        """
        return cls(name=name, x=x, y=y, anchor_type=AnchorType.CUSTOM)
