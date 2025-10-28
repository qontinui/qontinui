"""Match sorting operations.

Provides sorting functionality for match collections.
"""

from ..match import Match


class MatchSorters:
    """Sort operations for match collections.

    Provides various sorting methods for matches based on
    similarity and position.
    """

    @staticmethod
    def by_similarity(matches: list[Match], reverse: bool = True) -> None:
        """Sort matches by similarity in place.

        Args:
            matches: List of matches to sort
            reverse: True for descending order (best first)
        """
        matches.sort(key=lambda m: m.similarity, reverse=reverse)

    @staticmethod
    def by_position(
        matches: list[Match],
        top_to_bottom: bool = True,
        left_to_right: bool = True,
    ) -> None:
        """Sort matches by position in place.

        Args:
            matches: List of matches to sort
            top_to_bottom: True to sort top to bottom
            left_to_right: True to sort left to right
        """

        def position_key(match: Match):
            y = match.target.y if top_to_bottom else -match.target.y
            x = match.target.x if left_to_right else -match.target.x
            return (y, x)

        matches.sort(key=position_key)
