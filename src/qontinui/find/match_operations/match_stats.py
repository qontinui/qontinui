"""Match statistical and utility operations.

Provides statistical and utility functionality for match collections.
"""

from ..match import Match


class MatchStats:
    """Statistical and utility operations for match collections.

    Provides methods for calculating statistics and checking
    collection state.
    """

    @staticmethod
    def size(matches: list[Match]) -> int:
        """Get number of matches.

        Args:
            matches: List of matches

        Returns:
            Number of matches
        """
        return len(matches)

    @staticmethod
    def is_empty(matches: list[Match]) -> bool:
        """Check if collection is empty.

        Args:
            matches: List of matches

        Returns:
            True if no matches
        """
        return len(matches) == 0

    @staticmethod
    def has_matches(matches: list[Match]) -> bool:
        """Check if collection has matches.

        Args:
            matches: List of matches

        Returns:
            True if has matches
        """
        return len(matches) > 0
