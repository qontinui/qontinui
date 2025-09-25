"""Anchor points for declarative region positioning.

Defines anchor points that can be used to position regions
relative to other regions or the screen.
"""

from enum import Enum

from ..model.element import Region


class AnchorPoint(Enum):
    """Standard anchor points for positioning.

    Following Brobot's anchor system for intuitive positioning.
    """

    # Corner anchors
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"

    # Edge anchors
    TOP_CENTER = "top_center"
    BOTTOM_CENTER = "bottom_center"
    LEFT_CENTER = "left_center"
    RIGHT_CENTER = "right_center"

    # Center
    CENTER = "center"

    # Screen-relative anchors
    SCREEN_TOP_LEFT = "screen_top_left"
    SCREEN_TOP_RIGHT = "screen_top_right"
    SCREEN_BOTTOM_LEFT = "screen_bottom_left"
    SCREEN_BOTTOM_RIGHT = "screen_bottom_right"
    SCREEN_CENTER = "screen_center"


class Anchor:
    """Calculates anchor positions for regions.

    Following Brobot principles:
    - Intuitive anchor point naming
    - Support for all standard positions
    - Screen-relative positioning
    """

    @staticmethod
    def get_point(region: Region, anchor: AnchorPoint) -> tuple[int, int]:
        """Get the coordinates of an anchor point on a region.

        Args:
            region: Region to get anchor point from
            anchor: Which anchor point to get

        Returns:
            (x, y) coordinates of the anchor point
        """
        if anchor == AnchorPoint.TOP_LEFT:
            return (region.x, region.y)
        elif anchor == AnchorPoint.TOP_RIGHT:
            return (region.x + region.width, region.y)
        elif anchor == AnchorPoint.BOTTOM_LEFT:
            return (region.x, region.y + region.height)
        elif anchor == AnchorPoint.BOTTOM_RIGHT:
            return (region.x + region.width, region.y + region.height)
        elif anchor == AnchorPoint.TOP_CENTER:
            return (region.x + region.width // 2, region.y)
        elif anchor == AnchorPoint.BOTTOM_CENTER:
            return (region.x + region.width // 2, region.y + region.height)
        elif anchor == AnchorPoint.LEFT_CENTER:
            return (region.x, region.y + region.height // 2)
        elif anchor == AnchorPoint.RIGHT_CENTER:
            return (region.x + region.width, region.y + region.height // 2)
        elif anchor == AnchorPoint.CENTER:
            return (region.x + region.width // 2, region.y + region.height // 2)
        else:
            # Handle screen-relative anchors
            return Anchor._get_screen_anchor(anchor)

    @staticmethod
    def _get_screen_anchor(anchor: AnchorPoint) -> tuple[int, int]:
        """Get screen-relative anchor coordinates.

        Args:
            anchor: Screen anchor point

        Returns:
            (x, y) coordinates
        """
        # Get screen dimensions
        screen = Region()  # Uses default screen dimensions

        if anchor == AnchorPoint.SCREEN_TOP_LEFT:
            return (0, 0)
        elif anchor == AnchorPoint.SCREEN_TOP_RIGHT:
            return (screen.width, 0)
        elif anchor == AnchorPoint.SCREEN_BOTTOM_LEFT:
            return (0, screen.height)
        elif anchor == AnchorPoint.SCREEN_BOTTOM_RIGHT:
            return (screen.width, screen.height)
        elif anchor == AnchorPoint.SCREEN_CENTER:
            return (screen.width // 2, screen.height // 2)
        else:
            return (0, 0)

    @staticmethod
    def align_to(
        region: Region, target: Region, from_anchor: AnchorPoint, to_anchor: AnchorPoint
    ) -> Region:
        """Align a region to another region using anchor points.

        Args:
            region: Region to position
            target: Target region to align to
            from_anchor: Anchor point on region
            to_anchor: Anchor point on target

        Returns:
            New aligned region
        """
        # Get anchor points
        from_point = Anchor.get_point(region, from_anchor)
        to_point = Anchor.get_point(target, to_anchor)

        # Calculate offset
        offset_x = to_point[0] - from_point[0]
        offset_y = to_point[1] - from_point[1]

        # Create aligned region
        return Region(
            x=region.x + offset_x,
            y=region.y + offset_y,
            width=region.width,
            height=region.height,
            name=region.name,
        )
