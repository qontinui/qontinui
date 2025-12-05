"""Similarity threshold filter for match filtering.

Filters matches based on their similarity score, keeping only those
that meet or exceed the minimum threshold.
"""


from ..match import Match
from .match_filter import MatchFilter


class SimilarityFilter(MatchFilter):
    """Filter matches by minimum similarity threshold.

    Removes matches that have a similarity score below the specified
    minimum threshold. This is typically used to ensure only high-quality
    matches are retained.

    The similarity score is expected to be in range [0.0, 1.0] where:
    - 0.0 = no similarity
    - 1.0 = perfect match

    Example:
        >>> filter = SimilarityFilter(min_similarity=0.8)
        >>> filtered = filter.filter(matches)  # Keep only matches >= 0.8
    """

    def __init__(self, min_similarity: float) -> None:
        """Initialize similarity filter.

        Args:
            min_similarity: Minimum similarity threshold.
                           Range: 0.0 to 1.0
                           Higher values = more strict filtering

        Raises:
            ValueError: If min_similarity is not in range [0.0, 1.0]
        """
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError(f"min_similarity must be in [0.0, 1.0], got {min_similarity}")
        self.min_similarity = min_similarity

    def filter(self, matches: list[Match]) -> list[Match]:
        """Filter matches by similarity threshold.

        Args:
            matches: List of matches to filter

        Returns:
            Filtered list containing only matches with similarity >= min_similarity

        Raises:
            ValueError: If any match has invalid similarity score
        """
        if not matches:
            return matches

        filtered_matches: list[Match] = []
        for match in matches:
            # Validate similarity score
            if not isinstance(match.similarity, int | float):
                raise ValueError(f"Match has invalid similarity score: {match.similarity}")

            if not 0.0 <= match.similarity <= 1.0:
                raise ValueError(f"Match similarity must be in [0.0, 1.0], got {match.similarity}")

            # Keep matches that meet or exceed threshold
            if match.similarity >= self.min_similarity:
                filtered_matches.append(match)

        return filtered_matches
