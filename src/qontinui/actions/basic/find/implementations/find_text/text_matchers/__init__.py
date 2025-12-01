"""Text matcher implementations.

Strategy pattern implementations for different text matching algorithms.
"""

from .base_matcher import BaseMatcher
from .exact_matcher import (
    ContainsMatcher,
    EndsWithMatcher,
    ExactMatcher,
    StartsWithMatcher,
)
from .fuzzy_matcher import FuzzyMatcher
from .regex_matcher import RegexMatcher

__all__ = [
    "BaseMatcher",
    "ExactMatcher",
    "ContainsMatcher",
    "StartsWithMatcher",
    "EndsWithMatcher",
    "FuzzyMatcher",
    "RegexMatcher",
]
