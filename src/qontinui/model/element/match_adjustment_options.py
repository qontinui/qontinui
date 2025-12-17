"""MatchAdjustmentOptions class - faithful port from Brobot framework.

Configuration for adjusting the position and dimensions of found matches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .location import Location
    from .position import Position


@dataclass
class MatchAdjustmentOptions:
    """Configuration for adjusting the position and dimensions of found matches.

    Faithful port of Brobot's MatchAdjustmentOptions class.

    This class encapsulates all parameters for post-processing the region of a Match.
    It allows for dynamic resizing or targeting of specific points within a match,
    providing flexibility for subsequent actions like clicks or drags.

    Common use cases:
    - Offset click point from matched element (e.g., click 10px to the right)
    - Expand region to include surrounding area
    - Contract region to focus on center
    - Target specific position within match (e.g., top-right corner)
    - Set absolute dimensions regardless of match size

    Example:
        # Click 10 pixels right and 20 pixels down from center of match
        options = MatchAdjustmentOptions(add_x=10, add_y=20)

        # Expand matched region by 50 pixels in all directions
        options = MatchAdjustmentOptions(add_w=100, add_h=100, add_x=-50, add_y=-50)

        # Target top-left corner with 5px offset
        options = MatchAdjustmentOptions(
            target_position=Position(percent_w=0.0, percent_h=0.0),
            target_offset=Location(x=5, y=5)
        )
    """

    # Target position within a match's bounds (e.g., CENTER, TOP_LEFT)
    # This overrides any default position defined in the search pattern
    target_position: Position | None = None

    # Pixel offset from the calculated target position
    # Useful for interacting near, but not directly on, an element
    target_offset: Location | None = None

    # Number of pixels to add to the width of the match region
    add_w: int = 0

    # Number of pixels to add to the height of the match region
    add_h: int = 0

    # Absolute width of the match region, overriding its original width
    # A value less than 0 disables this setting
    absolute_w: int = -1

    # Absolute height of the match region, overriding its original height
    # A value less than 0 disables this setting
    absolute_h: int = -1

    # Number of pixels to add to the x-coordinate of the match region's origin
    add_x: int = 0

    # Number of pixels to add to the y-coordinate of the match region's origin
    add_y: int = 0

    def with_target_position(self, position: Position) -> MatchAdjustmentOptions:
        """Set target position within match (fluent).

        Args:
            position: Target position (e.g., Position with percent_w=0.5, percent_h=0.5 for center)

        Returns:
            Self for chaining
        """
        self.target_position = position
        return self

    def with_target_offset(self, offset: Location) -> MatchAdjustmentOptions:
        """Set pixel offset from target position (fluent).

        Args:
            offset: Offset in pixels from calculated position

        Returns:
            Self for chaining
        """
        self.target_offset = offset
        return self

    def with_add_w(self, pixels: int) -> MatchAdjustmentOptions:
        """Add pixels to match width (fluent).

        Args:
            pixels: Pixels to add to width (negative to reduce)

        Returns:
            Self for chaining
        """
        self.add_w = pixels
        return self

    def with_add_h(self, pixels: int) -> MatchAdjustmentOptions:
        """Add pixels to match height (fluent).

        Args:
            pixels: Pixels to add to height (negative to reduce)

        Returns:
            Self for chaining
        """
        self.add_h = pixels
        return self

    def with_absolute_w(self, pixels: int) -> MatchAdjustmentOptions:
        """Set absolute width for match region (fluent).

        Args:
            pixels: Absolute width in pixels (< 0 to disable)

        Returns:
            Self for chaining
        """
        self.absolute_w = pixels
        return self

    def with_absolute_h(self, pixels: int) -> MatchAdjustmentOptions:
        """Set absolute height for match region (fluent).

        Args:
            pixels: Absolute height in pixels (< 0 to disable)

        Returns:
            Self for chaining
        """
        self.absolute_h = pixels
        return self

    def with_add_x(self, pixels: int) -> MatchAdjustmentOptions:
        """Add pixels to match X coordinate (fluent).

        Args:
            pixels: Pixels to add to X (negative to move left)

        Returns:
            Self for chaining
        """
        self.add_x = pixels
        return self

    def with_add_y(self, pixels: int) -> MatchAdjustmentOptions:
        """Add pixels to match Y coordinate (fluent).

        Args:
            pixels: Pixels to add to Y (negative to move up)

        Returns:
            Self for chaining
        """
        self.add_y = pixels
        return self

    def with_offset(self, x: int, y: int) -> MatchAdjustmentOptions:
        """Set X and Y offset together (fluent).

        Args:
            x: X offset in pixels
            y: Y offset in pixels

        Returns:
            Self for chaining
        """
        self.add_x = x
        self.add_y = y
        return self

    def with_expand(self, pixels: int) -> MatchAdjustmentOptions:
        """Expand matched region by pixels in all directions (fluent).

        Args:
            pixels: Pixels to expand (applies to all sides)

        Returns:
            Self for chaining
        """
        self.add_w = pixels * 2
        self.add_h = pixels * 2
        self.add_x = -pixels
        self.add_y = -pixels
        return self

    def with_contract(self, pixels: int) -> MatchAdjustmentOptions:
        """Contract matched region by pixels in all directions (fluent).

        Args:
            pixels: Pixels to contract (applies to all sides)

        Returns:
            Self for chaining
        """
        self.add_w = -pixels * 2
        self.add_h = -pixels * 2
        self.add_x = pixels
        self.add_y = pixels
        return self

    def copy(self) -> MatchAdjustmentOptions:
        """Create a copy of these adjustment options.

        Returns:
            New MatchAdjustmentOptions instance
        """
        return MatchAdjustmentOptions(
            target_position=self.target_position,
            target_offset=self.target_offset,
            add_w=self.add_w,
            add_h=self.add_h,
            absolute_w=self.absolute_w,
            absolute_h=self.absolute_h,
            add_x=self.add_x,
            add_y=self.add_y,
        )

    def __str__(self) -> str:
        """String representation."""
        parts = []
        if self.add_x != 0 or self.add_y != 0:
            parts.append(f"offset=({self.add_x},{self.add_y})")
        if self.add_w != 0 or self.add_h != 0:
            parts.append(f"size_adjust=({self.add_w},{self.add_h})")
        if self.absolute_w >= 0 or self.absolute_h >= 0:
            parts.append(f"absolute_size=({self.absolute_w},{self.absolute_h})")
        if self.target_position:
            parts.append(f"target_pos={self.target_position}")
        if self.target_offset:
            parts.append(f"target_offset={self.target_offset}")
        return (
            f"MatchAdjustmentOptions({', '.join(parts) if parts else 'no adjustments'})"
        )

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"MatchAdjustmentOptions(add_x={self.add_x}, add_y={self.add_y}, "
            f"add_w={self.add_w}, add_h={self.add_h}, "
            f"absolute_w={self.absolute_w}, absolute_h={self.absolute_h}, "
            f"target_position={self.target_position}, target_offset={self.target_offset})"
        )
