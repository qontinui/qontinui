"""FindResult dataclass - Results from vision find operations.

Container for matches found by vision operations with metadata,
supporting filtering, transformation, and analysis operations.
Follows clean code principles with type hints and comprehensive docstrings.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..model.element import Pattern, Region

if TYPE_CHECKING:
    from ..find.match import Match


@dataclass
class FindResult:
    """Result from a vision find operation.

    Container for all matches found and metadata about the search operation.
    Provides convenient properties and transformation methods for working with results.

    Attributes:
        matches: List of Match objects found
        pattern: Pattern that was searched for (optional)
        search_region: Region that was searched (optional)
        strategy: Strategy used for finding (as string)
        duration: Time taken for find operation in seconds
        timestamp: Unix timestamp when search was performed
        screenshot: Screenshot used for finding (optional, image data)
        metadata: Custom metadata dict for extensibility
    """

    matches: list[Match] = field(default_factory=list)
    pattern: Pattern | None = None
    search_region: Region | None = None
    strategy: str = "template"
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)
    screenshot: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        """Check if any matches were found.

        Returns:
            True if at least one match was found
        """
        return len(self.matches) > 0

    @property
    def success(self) -> bool:
        """Alias for found property.

        Returns:
            True if matches were found
        """
        return self.found

    @property
    def count(self) -> int:
        """Get number of matches found.

        Returns:
            Number of matches
        """
        return len(self.matches)

    @property
    def first_match(self) -> Match | None:
        """Get first match found.

        Returns:
            First match or None if no matches
        """
        return self.matches[0] if self.matches else None

    @property
    def best_match(self) -> Match | None:
        """Get best match by similarity score.

        Returns:
            Match with highest similarity or None if no matches
        """
        if not self.matches:
            return None
        return max(self.matches, key=lambda m: m.similarity)

    @property
    def worst_match(self) -> Match | None:
        """Get worst match by similarity score.

        Returns:
            Match with lowest similarity or None if no matches
        """
        if not self.matches:
            return None
        return min(self.matches, key=lambda m: m.similarity)

    def get_match(self, index: int = 0) -> Match | None:
        """Get match at specific index.

        Args:
            index: Index of match (0-based)

        Returns:
            Match at index or None if index out of range
        """
        if 0 <= index < len(self.matches):
            return self.matches[index]
        return None

    def filter_by_similarity(self, min_similarity: float) -> FindResult:
        """Create new result with matches filtered by minimum similarity.

        Args:
            min_similarity: Minimum similarity threshold (0.0-1.0)

        Returns:
            New FindResult with filtered matches

        Raises:
            ValueError: If min_similarity is outside valid range
        """
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError(f"min_similarity must be 0.0-1.0, got {min_similarity}")

        filtered = [m for m in self.matches if m.similarity >= min_similarity]
        return FindResult(
            matches=filtered,
            pattern=self.pattern,
            search_region=self.search_region,
            strategy=self.strategy,
            duration=self.duration,
            timestamp=self.timestamp,
            screenshot=self.screenshot,
            metadata={**self.metadata, "filtered_by_similarity": min_similarity},
        )

    def filter_by_region(self, region: Region) -> FindResult:
        """Create new result with matches filtered by region.

        Args:
            region: Region to filter by

        Returns:
            New FindResult with matches in region

        Raises:
            ValueError: If region is invalid
        """
        if region is None:
            raise ValueError("Region cannot be None")

        filtered = [
            m for m in self.matches if m.region is not None and region.contains_region(m.region)  # type: ignore[attr-defined]
        ]
        return FindResult(
            matches=filtered,
            pattern=self.pattern,
            search_region=self.search_region,
            strategy=self.strategy,
            duration=self.duration,
            timestamp=self.timestamp,
            screenshot=self.screenshot,
            metadata={**self.metadata, "filtered_by_region": str(region)},
        )

    def filter_by_distance(
        self, reference_match: Match, max_distance: float
    ) -> FindResult:
        """Create new result with matches filtered by distance from reference.

        Args:
            reference_match: Reference match for distance calculation
            max_distance: Maximum distance in pixels

        Returns:
            New FindResult with matches within distance

        Raises:
            ValueError: If max_distance is negative
        """
        if max_distance < 0:
            raise ValueError(f"max_distance must be non-negative, got {max_distance}")

        filtered = [
            m
            for m in self.matches
            if reference_match.center.distance_to(m.center) <= max_distance
        ]
        return FindResult(
            matches=filtered,
            pattern=self.pattern,
            search_region=self.search_region,
            strategy=self.strategy,
            duration=self.duration,
            timestamp=self.timestamp,
            screenshot=self.screenshot,
            metadata={**self.metadata, "filtered_by_distance": max_distance},
        )

    def remove_overlapping(self, overlap_threshold: float = 0.5) -> FindResult:
        """Create new result with overlapping matches removed.

        Keeps matches with highest similarity when overlaps detected.

        Args:
            overlap_threshold: Overlap ratio threshold (0.0-1.0, default 0.5)

        Returns:
            New FindResult without overlaps

        Raises:
            ValueError: If overlap_threshold is outside valid range
        """
        if not 0.0 <= overlap_threshold <= 1.0:
            raise ValueError(
                f"overlap_threshold must be 0.0-1.0, got {overlap_threshold}"
            )

        if not self.matches:
            return FindResult(
                matches=[],
                pattern=self.pattern,
                search_region=self.search_region,
                strategy=self.strategy,
                duration=self.duration,
                timestamp=self.timestamp,
                screenshot=self.screenshot,
                metadata={**self.metadata, "overlap_removed": True},
            )

        # Sort by similarity descending
        sorted_matches = sorted(self.matches, key=lambda m: m.similarity, reverse=True)
        kept: list[Any] = []

        for match in sorted_matches:
            # Check if overlaps with any kept match
            overlaps = False
            for kept_match in kept:
                if (
                    match.region is not None
                    and kept_match.region is not None
                    and match.region.overlap_ratio(kept_match.region) >= overlap_threshold  # type: ignore[attr-defined]
                ):
                    overlaps = True
                    break

            if not overlaps:
                kept.append(match)

        return FindResult(
            matches=kept,
            pattern=self.pattern,
            search_region=self.search_region,
            strategy=self.strategy,
            duration=self.duration,
            timestamp=self.timestamp,
            screenshot=self.screenshot,
            metadata={**self.metadata, "overlap_removed": True},
        )

    def sort_by_similarity(self, descending: bool = True) -> FindResult:
        """Create new result with matches sorted by similarity.

        Args:
            descending: Sort in descending order if True, ascending if False

        Returns:
            New FindResult with sorted matches
        """
        sorted_matches = sorted(
            self.matches, key=lambda m: m.similarity, reverse=descending
        )
        return FindResult(
            matches=sorted_matches,
            pattern=self.pattern,
            search_region=self.search_region,
            strategy=self.strategy,
            duration=self.duration,
            timestamp=self.timestamp,
            screenshot=self.screenshot,
            metadata={**self.metadata, "sorted_by_similarity": True},
        )

    def sort_by_location(self, horizontal: bool = True) -> FindResult:
        """Create new result with matches sorted by location.

        Args:
            horizontal: Sort left-to-right if True, top-to-bottom if False

        Returns:
            New FindResult with sorted matches
        """
        if horizontal:
            sorted_matches = sorted(
                self.matches,
                key=lambda m: m.location.x if m.location is not None else 0,
            )
        else:
            sorted_matches = sorted(
                self.matches,
                key=lambda m: m.location.y if m.location is not None else 0,
            )

        return FindResult(
            matches=sorted_matches,
            pattern=self.pattern,
            search_region=self.search_region,
            strategy=self.strategy,
            duration=self.duration,
            timestamp=self.timestamp,
            screenshot=self.screenshot,
            metadata={**self.metadata, "sorted_by_location": True},
        )

    def limit(self, max_matches: int) -> FindResult:
        """Create new result limited to specified number of matches.

        Args:
            max_matches: Maximum matches to keep

        Returns:
            New FindResult with limited matches

        Raises:
            ValueError: If max_matches is negative
        """
        if max_matches < 0:
            raise ValueError(f"max_matches must be non-negative, got {max_matches}")

        limited = self.matches[:max_matches]
        return FindResult(
            matches=limited,
            pattern=self.pattern,
            search_region=self.search_region,
            strategy=self.strategy,
            duration=self.duration,
            timestamp=self.timestamp,
            screenshot=self.screenshot,
            metadata={**self.metadata, "limited": max_matches},
        )

    def with_pattern(self, pattern: Pattern) -> FindResult:
        """Create new result with pattern reference.

        Args:
            pattern: Pattern that was searched

        Returns:
            New FindResult with pattern
        """
        return FindResult(
            matches=self.matches,
            pattern=pattern,
            search_region=self.search_region,
            strategy=self.strategy,
            duration=self.duration,
            timestamp=self.timestamp,
            screenshot=self.screenshot,
            metadata=self.metadata,
        )

    def with_metadata(self, key: str, value: Any) -> FindResult:
        """Create new result with added metadata.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            New FindResult with metadata added
        """
        return FindResult(
            matches=self.matches,
            pattern=self.pattern,
            search_region=self.search_region,
            strategy=self.strategy,
            duration=self.duration,
            timestamp=self.timestamp,
            screenshot=self.screenshot,
            metadata={**self.metadata, key: value},
        )

    def get_average_similarity(self) -> float:
        """Calculate average similarity of all matches.

        Returns:
            Average similarity score (0.0-1.0) or 0.0 if no matches
        """
        if not self.matches:
            return 0.0
        return sum(m.similarity for m in self.matches) / len(self.matches)

    def get_regions(self) -> list[Region]:
        """Get all match regions.

        Returns:
            List of regions for all matches (excludes None regions)
        """
        return [m.region for m in self.matches if m.region is not None]

    def __str__(self) -> str:
        """String representation.

        Returns:
            Human-readable description
        """
        pattern_name = self.pattern.name if self.pattern else "Unknown"
        avg_sim = self.get_average_similarity()
        return (
            f"FindResult(pattern='{pattern_name}', "
            f"found={self.count} matches, "
            f"avg_similarity={avg_sim:.2f}, "
            f"duration={self.duration:.3f}s)"
        )

    def __repr__(self) -> str:
        """Developer representation.

        Returns:
            Detailed description
        """
        return (
            f"FindResult(matches={len(self.matches)}, "
            f"pattern={self.pattern}, "
            f"strategy='{self.strategy}', "
            f"duration={self.duration:.3f}s)"
        )

    def __bool__(self) -> bool:
        """Boolean evaluation - True if matches found.

        Returns:
            True if at least one match exists
        """
        return self.found

    def __len__(self) -> int:
        """Get number of matches.

        Returns:
            Number of matches found
        """
        return len(self.matches)

    def __getitem__(self, index: int) -> Match | None:
        """Get match by index (supports negative indexing).

        Args:
            index: Match index

        Returns:
            Match at index or None

        Raises:
            IndexError: If index out of range
        """
        if -len(self.matches) <= index < len(self.matches):
            return self.matches[index]
        raise IndexError(f"Match index out of range: {index}")

    def __iter__(self):
        """Iterate over matches.

        Returns:
            Iterator over matches
        """
        return iter(self.matches)

    @classmethod
    def empty(
        cls, pattern: Pattern | None = None, strategy: str = "template"
    ) -> FindResult:
        """Create empty (no matches) result.

        Args:
            pattern: Optional pattern that was searched
            strategy: Strategy used (default "template")

        Returns:
            Empty FindResult
        """
        return cls(
            matches=[], pattern=pattern, strategy=strategy, metadata={"empty": True}
        )
