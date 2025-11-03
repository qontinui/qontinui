"""FindResult - Result of a find operation.

Returned by both mock and real implementations in identical format.
"""

from dataclasses import dataclass
from typing import Any

from ...model.match import Match


@dataclass
class FindResult:
    """Result of a find operation.

    This format is identical whether returned from mock or real execution.
    The action layer above is agnostic to which implementation was used.
    """

    matches: list[Match]
    found: bool
    pattern_name: str
    duration_ms: float
    debug_data: dict[str, Any] | None = None

    @property
    def best_match(self) -> Match | None:
        """Best match by confidence, or None if not found."""
        return self.matches[0] if self.matches else None
