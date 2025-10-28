"""Exact text matching implementation."""

from .base_matcher import BaseMatcher


class ExactMatcher(BaseMatcher):
    """Exact string matching.

    Returns 1.0 for perfect match, 0.0 otherwise.
    """

    def match(self, search_text: str, found_text: str) -> float:
        """Check for exact match.

        Args:
            search_text: Text to search for
            found_text: Text found by OCR

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        search = self.preprocess_text(search_text)
        found = self.preprocess_text(found_text)

        return 1.0 if search == found else 0.0


class ContainsMatcher(BaseMatcher):
    """Contains matching.

    Returns 1.0 if found text contains search text, 0.0 otherwise.
    """

    def match(self, search_text: str, found_text: str) -> float:
        """Check if found text contains search text.

        Args:
            search_text: Text to search for
            found_text: Text found by OCR

        Returns:
            1.0 if contains, 0.0 otherwise
        """
        search = self.preprocess_text(search_text)
        found = self.preprocess_text(found_text)

        return 1.0 if search in found else 0.0


class StartsWithMatcher(BaseMatcher):
    """Starts-with matching.

    Returns 1.0 if found text starts with search text, 0.0 otherwise.
    """

    def match(self, search_text: str, found_text: str) -> float:
        """Check if found text starts with search text.

        Args:
            search_text: Text to search for
            found_text: Text found by OCR

        Returns:
            1.0 if starts with, 0.0 otherwise
        """
        search = self.preprocess_text(search_text)
        found = self.preprocess_text(found_text)

        return 1.0 if found.startswith(search) else 0.0


class EndsWithMatcher(BaseMatcher):
    """Ends-with matching.

    Returns 1.0 if found text ends with search text, 0.0 otherwise.
    """

    def match(self, search_text: str, found_text: str) -> float:
        """Check if found text ends with search text.

        Args:
            search_text: Text to search for
            found_text: Text found by OCR

        Returns:
            1.0 if ends with, 0.0 otherwise
        """
        search = self.preprocess_text(search_text)
        found = self.preprocess_text(found_text)

        return 1.0 if found.endswith(search) else 0.0
