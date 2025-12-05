"""Base filter interface for match filtering.

Provides abstract base class for implementing match filtering strategies.
Each filter should have a single, focused responsibility.
"""


from abc import ABC, abstractmethod

from ..match import Match


class MatchFilter(ABC):
    """Abstract base class for match filters.

    All filters must implement the filter() method which takes a list
    of matches and returns a filtered list.

    Filters should be composable and stateless where possible.
    """

    @abstractmethod
    def filter(self, matches: list[Match]) -> list[Match]:
        """Filter a list of matches according to filter criteria.

        Args:
            matches: List of matches to filter

        Returns:
            Filtered list of matches

        Raises:
            ValueError: If matches list is invalid or contains invalid matches
        """
        pass
