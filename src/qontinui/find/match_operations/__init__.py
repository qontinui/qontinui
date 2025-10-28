"""Match operations module.

Provides helper classes for match collection operations:
- MatchFilters: Filtering operations
- MatchQueries: Query operations
- MatchSorters: Sorting operations
- MatchStats: Statistical operations
"""

from .match_filters import MatchFilters
from .match_queries import MatchQueries
from .match_sorters import MatchSorters
from .match_stats import MatchStats

__all__ = [
    "MatchFilters",
    "MatchQueries",
    "MatchSorters",
    "MatchStats",
]
