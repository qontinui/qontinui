"""Match filtering operations.

Provides filtering functionality for match collections.
"""

from collections.abc import Callable

from ...model.element import Location, Region
from ..match import Match


class MatchFilters:
    """Filter operations for match collections.

    Provides various filtering methods for matches based on
    similarity, region, distance, and custom predicates.
    """

    @staticmethod
    def by_similarity(matches: list[Match], min_similarity: float) -> list[Match]:
        """Filter matches by minimum similarity.

        Args:
            matches: List of matches to filter
            min_similarity: Minimum similarity threshold

        Returns:
            Filtered list of matches
        """
        return [m for m in matches if m.similarity >= min_similarity]

    @staticmethod
    def by_region(matches: list[Match], region: Region) -> list[Match]:
        """Filter matches within a region.

        Args:
            matches: List of matches to filter
            region: Region to filter by

        Returns:
            Filtered list of matches
        """
        return [m for m in matches if region.contains(m.center)]

    @staticmethod
    def by_distance(matches: list[Match], location: Location, max_distance: float) -> list[Match]:
        """Filter matches by distance from location.

        Args:
            matches: List of matches to filter
            location: Reference location
            max_distance: Maximum distance

        Returns:
            Filtered list of matches
        """
        return [m for m in matches if m.center.distance_to(location) <= max_distance]

    @staticmethod
    def by_predicate(matches: list[Match], predicate: Callable[[Match], bool]) -> list[Match]:
        """Filter matches using custom predicate.

        Args:
            matches: List of matches to filter
            predicate: Function that returns True to keep match

        Returns:
            Filtered list of matches
        """
        return [m for m in matches if predicate(m)]

    @staticmethod
    def remove_overlapping(matches: list[Match], overlap_threshold: float = 0.5) -> list[Match]:
        """Remove overlapping matches, keeping best ones.

        Args:
            matches: List of matches to process
            overlap_threshold: Minimum overlap ratio to consider overlapping

        Returns:
            List of matches without overlaps
        """
        if len(matches) <= 1:
            return matches.copy()

        # Sort by similarity (best first)
        sorted_matches = sorted(matches, key=lambda m: m.similarity, reverse=True)
        kept_matches: list[Match] = []

        for match in sorted_matches:
            # Skip matches without regions
            if match.region is None:
                continue

            # Check if overlaps with any kept match
            overlaps = False
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

        return kept_matches
