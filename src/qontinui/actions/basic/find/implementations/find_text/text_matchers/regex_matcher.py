"""Regular expression text matching implementation."""

import logging
import re

from .base_matcher import BaseMatcher

logger = logging.getLogger(__name__)


class RegexMatcher(BaseMatcher):
    """Regular expression matching.

    Matches text using Python regular expressions.
    """

    def __init__(
        self, case_sensitive: bool = False, ignore_whitespace: bool = False
    ) -> None:
        """Initialize regex matcher.

        Args:
            case_sensitive: Whether matching should be case-sensitive
            ignore_whitespace: Whether to ignore whitespace in matching
        """
        super().__init__(case_sensitive, ignore_whitespace)
        self._pattern_cache: dict[str, re.Pattern] = {}

    def match(self, search_text: str, found_text: str) -> float:
        """Match text using regular expression.

        Args:
            search_text: Regular expression pattern
            found_text: Text found by OCR

        Returns:
            1.0 if pattern matches, 0.0 otherwise
        """
        found = self.preprocess_text(found_text)

        # Compile pattern with caching
        pattern = self._get_compiled_pattern(search_text)
        if pattern is None:
            return 0.0

        # Check if pattern matches
        return 1.0 if pattern.search(found) else 0.0

    def _get_compiled_pattern(self, pattern_str: str) -> re.Pattern | None:
        """Get compiled regex pattern with caching.

        Args:
            pattern_str: Regular expression pattern string

        Returns:
            Compiled pattern or None if invalid
        """
        # Check cache
        if pattern_str in self._pattern_cache:
            return self._pattern_cache[pattern_str]

        # Compile pattern
        try:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            pattern = re.compile(pattern_str, flags)
            self._pattern_cache[pattern_str] = pattern
            return pattern
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pattern_str}': {e}")
            return None
