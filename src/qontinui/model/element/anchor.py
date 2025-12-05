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
    """Defines a reference point for region definition and spatial relationships.

    IMPORTANT DESIGN NOTE:
    ======================
    In Qontinui, Anchor and Location serve DIFFERENT purposes:
    - Anchor: Used for REGION DEFINITION and spatial relationships
    - Location: Used for ACTION TARGETS (where to click, type, hover)

    This differs from Brobot where the relationship may be structured differently.
    See LOCATION_ANCHOR_DESIGN.md for full design rationale.

    PRIMARY PURPOSE:
    ================
    Anchors are used to define COMPONENTS OF REGIONS:
    - Two anchors can define opposite corners of a rectangular region
    - Multiple anchors can define complex region boundaries
    - Anchors establish spatial relationships between elements

    REGION COMPONENT SPECIFICATION:
    ================================
    When an Anchor is part of a region definition, it specifies:
    1. WHICH PART of the region it represents (via anchor_type)
       - TOP_LEFT: Upper-left corner of the region
       - BOTTOM_RIGHT: Lower-right corner of the region
       - CENTER: Center point of the region
       - etc.

    2. HOW IT RELATES to other anchors defining the same region
       - Two TOP_LEFT and BOTTOM_RIGHT anchors define a rectangle
       - Multiple anchors can define irregular regions

    Example:
    ```python
    # Define a region between two UI elements
    top_anchor = Anchor(name="header", anchor_type=AnchorType.BOTTOM_LEFT)
    bottom_anchor = Anchor(name="footer", anchor_type=AnchorType.TOP_RIGHT)
    search_region = Region.from_anchors(top_anchor, bottom_anchor)
    ```

    KEY DIFFERENCES FROM LOCATION:
    ===============================
    - Anchor DEFINES region boundaries
    - Anchor IS a component of region definition
    - Anchor IS NOT the target for action operations
    - Location IS for specifying where to perform actions

    When you need to click/type somewhere, use Location.
    When you need to define a region's boundaries, use Anchor.
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
