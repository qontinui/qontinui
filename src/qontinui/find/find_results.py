"""FindResults class - ported from Qontinui framework.

Container for results from find operations with metadata.
"""

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from ..model.element import Pattern, Region
from .matches import Matches

if TYPE_CHECKING:
    from .match import Match


@dataclass
class FindResults:
    """Results from a find operation.

    Port of FindResults from Qontinui framework class.
    Contains matches and metadata about the find operation.
    """

    matches: Matches
    pattern: Pattern | None = None
    search_region: Region | None = None
    duration: float = 0.0  # Time taken for find operation
    timestamp: float = field(default_factory=time.time)
    screenshot: Any | None = None  # Screenshot used for finding
    method: str = "template"  # Finding method used
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if find was successful.

        Returns:
            True if matches were found
        """
        return self.matches.has_matches()

    @property
    def found(self) -> bool:
        """Alias for success.

        Returns:
            True if matches were found
        """
        return self.success

    @property
    def count(self) -> int:
        """Number of matches found.

        Returns:
            Number of matches
        """
        return self.matches.size()

    @property
    def first_match(self) -> Optional["Match"]:
        """Get first match.

        Returns:
            First match or None
        """
        return self.matches.first

    @property
    def best_match(self) -> Optional["Match"]:
        """Get best match by similarity.

        Returns:
            Best match or None
        """
        return self.matches.best

    def get_match(self, index: int = 0) -> Optional["Match"]:
        """Get match at specific index.

        Args:
            index: Index of match

        Returns:
            Match at index or None
        """
        return self.matches.get(index)

    def filter_by_similarity(self, min_similarity: float) -> "FindResults":
        """Filter results by minimum similarity.

        Args:
            min_similarity: Minimum similarity threshold

        Returns:
            New FindResults with filtered matches
        """
        return FindResults(
            matches=self.matches.filter_by_similarity(min_similarity),
            pattern=self.pattern,
            search_region=self.search_region,
            duration=self.duration,
            timestamp=self.timestamp,
            screenshot=self.screenshot,
            method=self.method,
            metadata={**self.metadata, "filtered_similarity": min_similarity},
        )

    def filter_by_region(self, region: Region) -> "FindResults":
        """Filter results by region.

        Args:
            region: Region to filter by

        Returns:
            New FindResults with filtered matches
        """
        return FindResults(
            matches=self.matches.filter_by_region(region),
            pattern=self.pattern,
            search_region=self.search_region,
            duration=self.duration,
            timestamp=self.timestamp,
            screenshot=self.screenshot,
            method=self.method,
            metadata={**self.metadata, "filtered_region": region},
        )

    def remove_overlapping(self, overlap_threshold: float = 0.5) -> "FindResults":
        """Remove overlapping matches.

        Args:
            overlap_threshold: Minimum overlap ratio

        Returns:
            New FindResults without overlaps
        """
        return FindResults(
            matches=self.matches.remove_overlapping(overlap_threshold),
            pattern=self.pattern,
            search_region=self.search_region,
            duration=self.duration,
            timestamp=self.timestamp,
            screenshot=self.screenshot,
            method=self.method,
            metadata={**self.metadata, "overlap_removed": True},
        )

    def with_pattern(self, pattern: Pattern) -> "FindResults":
        """Create new results with pattern reference.

        Args:
            pattern: Pattern that was searched

        Returns:
            New FindResults with pattern
        """
        return FindResults(
            matches=self.matches,
            pattern=pattern,
            search_region=self.search_region,
            duration=self.duration,
            timestamp=self.timestamp,
            screenshot=self.screenshot,
            method=self.method,
            metadata=self.metadata,
        )

    def __str__(self) -> str:
        """String representation."""
        pattern_name = self.pattern.name if self.pattern else "Unknown"
        return (
            f"FindResults(pattern='{pattern_name}', "
            f"found={self.count} matches, "
            f"duration={self.duration:.3f}s)"
        )

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"FindResults(matches={self.matches}, "
            f"pattern={self.pattern}, "
            f"duration={self.duration:.3f})"
        )

    def __bool__(self) -> bool:
        """Boolean evaluation - True if matches found."""
        return self.success

    @classmethod
    def empty(cls, pattern: Pattern | None = None) -> "FindResults":
        """Create empty (no matches) results.

        Args:
            pattern: Optional pattern that was searched

        Returns:
            Empty FindResults
        """
        return cls(matches=Matches(), pattern=pattern, metadata={"empty": True})
