"""Match filtering components for the find module.

This package provides composable filters for post-processing match results:

- MatchFilter: Abstract base class for all filters
- NMSFilter: Non-Maximum Suppression for removing overlapping matches
- SimilarityFilter: Filter by minimum similarity threshold
- RegionFilter: Filter by search region containment

Filters are designed to be:
- Focused: Each filter has a single responsibility
- Composable: Multiple filters can be chained together
- Testable: Clear interfaces with well-defined behavior

Example usage:
    >>> from qontinui.find.filters import NMSFilter, SimilarityFilter, RegionFilter
    >>>
    >>> # Apply filters sequentially
    >>> matches = find_all_matches()
    >>> matches = SimilarityFilter(0.8).filter(matches)
    >>> matches = NMSFilter(0.3).filter(matches)
    >>> matches = RegionFilter(search_region).filter(matches)
"""

from .match_filter import MatchFilter
from .nms_filter import NMSFilter
from .region_filter import RegionFilter
from .similarity_filter import SimilarityFilter

__all__ = [
    "MatchFilter",
    "NMSFilter",
    "RegionFilter",
    "SimilarityFilter",
]
