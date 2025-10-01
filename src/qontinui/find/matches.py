"""Matches class - ported from Qontinui framework.

Container for multiple Match objects with utility methods.
"""

from collections.abc import Callable, Iterator

from ..model.element import Location, Region
from .match import Match


class Matches:
    """Container for multiple match results.

    Port of Matches from Qontinui framework class.
    Provides methods for working with collections of matches.
    """

    def __init__(self, matches: list[Match] | None = None):
        """Initialize with optional list of matches.

        Args:
            matches: List of Match objects
        """
        self._matches: list[Match] = matches or []

    def add(self, match: Match) -> "Matches":
        """Add a match to the collection.

        Args:
            match: Match to add

        Returns:
            Self for chaining
        """
        self._matches.append(match)
        return self

    def add_all(self, matches: list[Match]) -> "Matches":
        """Add multiple matches to the collection.

        Args:
            matches: List of matches to add

        Returns:
            Self for chaining
        """
        self._matches.extend(matches)
        return self

    @property
    def first(self) -> Match | None:
        """Get the first match.

        Returns:
            First match or None if empty
        """
        return self._matches[0] if self._matches else None

    @property
    def last(self) -> Match | None:
        """Get the last match.

        Returns:
            Last match or None if empty
        """
        return self._matches[-1] if self._matches else None

    @property
    def best(self) -> Match | None:
        """Get the match with highest similarity.

        Returns:
            Best match or None if empty
        """
        if not self._matches:
            return None
        return max(self._matches, key=lambda m: m.similarity)

    @property
    def worst(self) -> Match | None:
        """Get the match with lowest similarity.

        Returns:
            Worst match or None if empty
        """
        if not self._matches:
            return None
        return min(self._matches, key=lambda m: m.similarity)

    def get(self, index: int) -> Match | None:
        """Get match at specific index.

        Args:
            index: Index of match

        Returns:
            Match at index or None
        """
        if 0 <= index < len(self._matches):
            return self._matches[index]
        return None

    def size(self) -> int:
        """Get number of matches.

        Returns:
            Number of matches
        """
        return len(self._matches)

    def is_empty(self) -> bool:
        """Check if collection is empty.

        Returns:
            True if no matches
        """
        return len(self._matches) == 0

    def has_matches(self) -> bool:
        """Check if collection has matches.

        Returns:
            True if has matches
        """
        return len(self._matches) > 0

    def sort_by_similarity(self, reverse: bool = True) -> "Matches":
        """Sort matches by similarity.

        Args:
            reverse: True for descending order (best first)

        Returns:
            Self for chaining
        """
        self._matches.sort(key=lambda m: m.similarity, reverse=reverse)
        return self

    def sort_by_position(self, top_to_bottom: bool = True, left_to_right: bool = True) -> "Matches":
        """Sort matches by position.

        Args:
            top_to_bottom: True to sort top to bottom
            left_to_right: True to sort left to right

        Returns:
            Self for chaining
        """

        def position_key(match: Match):
            y = match.target.y if top_to_bottom else -match.target.y
            x = match.target.x if left_to_right else -match.target.x
            return (y, x)

        self._matches.sort(key=position_key)
        return self

    def filter_by_similarity(self, min_similarity: float) -> "Matches":
        """Filter matches by minimum similarity.

        Args:
            min_similarity: Minimum similarity threshold

        Returns:
            New Matches with filtered results
        """
        filtered = [m for m in self._matches if m.similarity >= min_similarity]
        return Matches(filtered)

    def filter_by_region(self, region: Region) -> "Matches":
        """Filter matches within a region.

        Args:
            region: Region to filter by

        Returns:
            New Matches with filtered results
        """
        filtered = [m for m in self._matches if region.contains(m.center)]
        return Matches(filtered)

    def filter_by_distance(self, location: Location, max_distance: float) -> "Matches":
        """Filter matches by distance from location.

        Args:
            location: Reference location
            max_distance: Maximum distance

        Returns:
            New Matches with filtered results
        """
        filtered = [m for m in self._matches if m.center.distance_to(location) <= max_distance]
        return Matches(filtered)

    def filter(self, predicate: Callable[[Match], bool]) -> "Matches":
        """Filter matches using custom predicate.

        Args:
            predicate: Function that returns True to keep match

        Returns:
            New Matches with filtered results
        """
        filtered = [m for m in self._matches if predicate(m)]
        return Matches(filtered)

    def remove_overlapping(self, overlap_threshold: float = 0.5) -> "Matches":
        """Remove overlapping matches, keeping best ones.

        Args:
            overlap_threshold: Minimum overlap ratio to consider overlapping

        Returns:
            New Matches without overlaps
        """
        if len(self._matches) <= 1:
            return Matches(self._matches.copy())

        # Sort by similarity (best first)
        sorted_matches = sorted(self._matches, key=lambda m: m.similarity, reverse=True)
        kept_matches: list[Match] = []

        for match in sorted_matches:
            # Check if overlaps with any kept match
            overlaps = False
            # Skip matches without regions
            if match.region is None:
                continue

            for kept in kept_matches:
                if kept.region is None:
                    continue
                intersection = match.region.intersection(kept.region)
                if intersection:
                    overlap_ratio = intersection.area / min(match.region.area, kept.region.area)
                    if overlap_ratio > overlap_threshold:
                        overlaps = True
                        break

            if not overlaps:
                kept_matches.append(match)

        return Matches(kept_matches)

    def nearest_to(self, location: Location) -> Match | None:
        """Get match nearest to location.

        Args:
            location: Reference location

        Returns:
            Nearest match or None if empty
        """
        if not self._matches:
            return None
        return min(self._matches, key=lambda m: m.center.distance_to(location))

    def farthest_from(self, location: Location) -> Match | None:
        """Get match farthest from location.

        Args:
            location: Reference location

        Returns:
            Farthest match or None if empty
        """
        if not self._matches:
            return None
        return max(self._matches, key=lambda m: m.center.distance_to(location))

    def to_list(self) -> list[Match]:
        """Get list of all matches.

        Returns:
            List of Match objects
        """
        return self._matches.copy()

    def get_matches(self) -> list[Match]:
        """Get list of all matches (alias for to_list).

        Returns:
            List of Match objects
        """
        return self._matches.copy()

    def clear(self) -> "Matches":
        """Clear all matches.

        Returns:
            Self for chaining
        """
        self._matches.clear()
        return self

    def __iter__(self) -> Iterator[Match]:
        """Iterator over matches."""
        return iter(self._matches)

    def __len__(self) -> int:
        """Number of matches."""
        return len(self._matches)

    def __getitem__(self, index: int) -> Match:
        """Get match by index."""
        return self._matches[index]

    def __bool__(self) -> bool:
        """Boolean evaluation - True if has matches."""
        return len(self._matches) > 0

    def __str__(self) -> str:
        """String representation."""
        if not self._matches:
            return "Matches(empty)"
        best = self.best
        if best is None:
            return f"Matches({len(self._matches)} matches)"
        return f"Matches({len(self._matches)} matches, best={best.similarity:.3f})"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"Matches({self._matches!r})"
