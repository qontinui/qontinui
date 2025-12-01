"""Base text matcher interface.

Abstract base class for text matching strategies.
"""

from abc import ABC, abstractmethod


class BaseMatcher(ABC):
    """Abstract base class for text matchers.

    Implements the Strategy pattern for different text matching algorithms.
    """

    def __init__(
        self, case_sensitive: bool = False, ignore_whitespace: bool = False
    ) -> None:
        """Initialize text matcher.

        Args:
            case_sensitive: Whether matching should be case-sensitive
            ignore_whitespace: Whether to ignore whitespace in matching
        """
        self.case_sensitive = case_sensitive
        self.ignore_whitespace = ignore_whitespace

    @abstractmethod
    def match(self, search_text: str, found_text: str) -> float:
        """Check if text matches and return similarity score.

        Args:
            search_text: Text to search for
            found_text: Text found by OCR

        Returns:
            Similarity score (0.0-1.0), where 1.0 is perfect match
        """
        pass

    def preprocess_text(self, text: str) -> str:
        """Preprocess text according to matcher settings.

        Args:
            text: Text to preprocess

        Returns:
            Preprocessed text
        """
        result = text

        if not self.case_sensitive:
            result = result.lower()

        if self.ignore_whitespace:
            result = "".join(result.split())

        return result
