"""Match query operations.

Provides query functionality for match collections.
"""

from ...model.element import Location
from ..match import Match


class MatchQueries:
    """Query operations for match collections.

    Provides methods to retrieve specific matches from collections
    based on various criteria.
    """

    @staticmethod
    def get_at_index(matches: list[Match], index: int) -> Match | None:
        """Get match at specific index.

        Args:
            matches: List of matches
            index: Index of match

        Returns:
            Match at index or None
        """
        if 0 <= index < len(matches):
            return matches[index]
        return None

    @staticmethod
    def get_first(matches: list[Match]) -> Match | None:
        """Get the first match.

        Args:
            matches: List of matches

        Returns:
            First match or None if empty
        """
        return matches[0] if matches else None

    @staticmethod
    def get_last(matches: list[Match]) -> Match | None:
        """Get the last match.

        Args:
            matches: List of matches

        Returns:
            Last match or None if empty
        """
        return matches[-1] if matches else None

    @staticmethod
    def get_best(matches: list[Match]) -> Match | None:
        """Get the match with highest similarity.

        Args:
            matches: List of matches

        Returns:
            Best match or None if empty
        """
        if not matches:
            return None
        return max(matches, key=lambda m: m.similarity)

    @staticmethod
    def get_worst(matches: list[Match]) -> Match | None:
        """Get the match with lowest similarity.

        Args:
            matches: List of matches

        Returns:
            Worst match or None if empty
        """
        if not matches:
            return None
        return min(matches, key=lambda m: m.similarity)

    @staticmethod
    def get_nearest_to(matches: list[Match], location: Location) -> Match | None:
        """Get match nearest to location.

        Args:
            matches: List of matches
            location: Reference location

        Returns:
            Nearest match or None if empty
        """
        if not matches:
            return None
        return min(matches, key=lambda m: m.center.distance_to(location))

    @staticmethod
    def get_farthest_from(matches: list[Match], location: Location) -> Match | None:
        """Get match farthest from location.

        Args:
            matches: List of matches
            location: Reference location

        Returns:
            Farthest match or None if empty
        """
        if not matches:
            return None
        return max(matches, key=lambda m: m.center.distance_to(location))
