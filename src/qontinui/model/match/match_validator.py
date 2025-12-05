"""Match validation utilities.

Provides validation and comparison logic for Match objects.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .match import Match


class MatchValidator:
    """Validates and compares Match objects."""

    @staticmethod
    def compare_by_score(match1: Match, match2: Match) -> float:
        """Compare two matches by score.

        Args:
            match1: First match
            match2: Second match

        Returns:
            Score difference (match1.score - match2.score)
        """
        return match1.score - match2.score

    @staticmethod
    def is_valid_match(match: Match) -> bool:
        """Check if match has valid data.

        Args:
            match: Match to validate

        Returns:
            True if match has valid region and score
        """
        return match.get_region() is not None and 0.0 <= match.score <= 1.0

    @staticmethod
    def has_image_data(match: Match) -> bool:
        """Check if match has image data.

        Args:
            match: Match to check

        Returns:
            True if match has image
        """
        return match.image is not None

    @staticmethod
    def has_search_image(match: Match) -> bool:
        """Check if match has search image.

        Args:
            match: Match to check

        Returns:
            True if match has search image
        """
        return match.search_image is not None
