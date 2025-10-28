"""
Fix suggestion strategies for test migration issues.

This package contains specialized strategy modules for different types of fix suggestions.
"""

from .assertion_suggestions import AssertionSuggestionStrategy
from .dependency_suggestions import DependencySuggestionStrategy
from .import_suggestions import ImportSuggestionStrategy
from .setup_suggestions import SetupSuggestionStrategy
from .suggestion_formatter import SuggestionFormatter
from .suggestion_scorer import SuggestionScorer
from .syntax_suggestions import SyntaxSuggestionStrategy

__all__ = [
    "AssertionSuggestionStrategy",
    "DependencySuggestionStrategy",
    "ImportSuggestionStrategy",
    "SetupSuggestionStrategy",
    "SuggestionFormatter",
    "SuggestionScorer",
    "SyntaxSuggestionStrategy",
]
