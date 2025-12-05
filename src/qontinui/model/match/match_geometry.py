"""Match geometry calculations.

Provides geometric operations and location calculations for Match objects.
"""


from typing import TYPE_CHECKING

from ..element.location import Location

if TYPE_CHECKING:
    from .match import Match


class MatchGeometry:
    """Provides geometric calculations for Match objects."""

    @staticmethod
    def get_center(match: Match) -> Location:
        """Get center location of match.

        Args:
            match: Match object

        Returns:
            Center location
        """
        region = match.get_region()
        if region:
            return Location(x=region.x + region.width // 2, y=region.y + region.height // 2)
        return match.target if match.target else Location(0, 0)

    @staticmethod
    def get_x(match: Match) -> int:
        """Get x coordinate of match.

        Args:
            match: Match object

        Returns:
            X coordinate or 0
        """
        region = match.get_region()
        return region.x if region else 0

    @staticmethod
    def get_y(match: Match) -> int:
        """Get y coordinate of match.

        Args:
            match: Match object

        Returns:
            Y coordinate or 0
        """
        region = match.get_region()
        return region.y if region else 0

    @staticmethod
    def get_width(match: Match) -> int:
        """Get width of match.

        Args:
            match: Match object

        Returns:
            Width or 0
        """
        region = match.get_region()
        return region.width if region else 0

    @staticmethod
    def get_height(match: Match) -> int:
        """Get height of match.

        Args:
            match: Match object

        Returns:
            Height or 0
        """
        region = match.get_region()
        return region.height if region else 0

    @staticmethod
    def get_area(match: Match) -> int:
        """Get area of the match region.

        Args:
            match: Match object

        Returns:
            Area in pixels
        """
        region = match.get_region()
        return region.area if region else 0
