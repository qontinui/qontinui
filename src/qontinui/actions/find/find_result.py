"""FindResult - Result of a find operation.

Returned by both mock and real implementations in identical format.
"""

from dataclasses import dataclass, field
from typing import Any

from ...model.match import Match
from .matches import Matches


@dataclass
class FindResult:
    """Result of a find operation.

    This format is identical whether returned from mock or real execution.
    The action layer above is agnostic to which implementation was used.

    The `matches` field is a `Matches` collection providing rich operations:
    - Sorting: sort_by_similarity(), sort_by_position()
    - Filtering: filter_by_similarity(), filter_by_region(), filter()
    - Queries: best, worst, first, last, nearest_to(), farthest_from()
    """

    matches: Matches = field(default_factory=Matches)
    found: bool = False
    pattern_name: str = ""
    duration_ms: float = 0.0
    debug_data: dict[str, Any] | None = None

    @property
    def best_match(self) -> Match | None:
        """Best match by confidence, or None if not found."""
        return self.matches.best

    @property
    def all_matches(self) -> list[Match]:
        """Get all matches as a list."""
        return self.matches.to_list()

    @property
    def match_count(self) -> int:
        """Get number of matches found."""
        return len(self.matches)
