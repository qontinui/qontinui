"""Fuzzy text matching implementation."""

from difflib import SequenceMatcher

from .base_matcher import BaseMatcher


class FuzzyMatcher(BaseMatcher):
    """Fuzzy/approximate text matching.

    Uses sequence matching to calculate similarity between strings.
    Handles OCR errors and slight variations in text.
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        ignore_whitespace: bool = False,
        threshold: float = 0.8,
    ) -> None:
        """Initialize fuzzy matcher.

        Args:
            case_sensitive: Whether matching should be case-sensitive
            ignore_whitespace: Whether to ignore whitespace in matching
            threshold: Minimum similarity threshold (0.0-1.0)
        """
        super().__init__(case_sensitive, ignore_whitespace)
        self.threshold = threshold

    def match(self, search_text: str, found_text: str) -> float:
        """Calculate fuzzy similarity between texts.

        Args:
            search_text: Text to search for
            found_text: Text found by OCR

        Returns:
            Similarity score (0.0-1.0)
        """
        search = self.preprocess_text(search_text)
        found = self.preprocess_text(found_text)

        # Use SequenceMatcher for fuzzy comparison
        similarity = SequenceMatcher(None, search, found).ratio()

        return similarity
