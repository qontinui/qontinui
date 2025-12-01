"""
Fix suggestion engine for providing automated repair recommendations for test migration issues.

This module serves as a facade that delegates to specialized strategy modules.
"""

from pathlib import Path

from ..core.models import FailureAnalysis, TestFile
from .fix_models import FixSuggestion
from .strategies.assertion_suggestions import AssertionSuggestionStrategy
from .strategies.dependency_suggestions import DependencySuggestionStrategy
from .strategies.import_suggestions import ImportSuggestionStrategy
from .strategies.setup_suggestions import SetupSuggestionStrategy
from .strategies.suggestion_formatter import SuggestionFormatter
from .strategies.suggestion_scorer import SuggestionScorer
from .strategies.syntax_suggestions import SyntaxSuggestionStrategy


class FixSuggestionEngine:
    """
    Facade for providing automated repair recommendations for test migration issues.

    Delegates to specialized strategy modules for different fix types.
    """

    def __init__(self) -> None:
        """Initialize the fix suggestion engine with all strategies."""
        self._assertion_strategy = AssertionSuggestionStrategy()
        self._import_strategy = ImportSuggestionStrategy()
        self._setup_strategy = SetupSuggestionStrategy()
        self._dependency_strategy = DependencySuggestionStrategy()
        self._syntax_strategy = SyntaxSuggestionStrategy()
        self._scorer = SuggestionScorer()
        self._formatter = SuggestionFormatter(
            self._assertion_strategy, self._import_strategy, self._syntax_strategy
        )

    def suggest_fixes(
        self,
        failure_analysis: FailureAnalysis,
        test_file: TestFile | None = None,
        python_file_path: Path | None = None,
    ) -> list[FixSuggestion]:
        """
        Generate fix suggestions based on failure analysis.

        Args:
            failure_analysis: Analysis of the test failure
            test_file: Original Java test file (optional)
            python_file_path: Path to migrated Python test file (optional)

        Returns:
            List of fix suggestions
        """
        suggestions = []

        # Extract error information from diagnostic info
        diagnostic_info = failure_analysis.diagnostic_info
        error_message = diagnostic_info.get("error_message", "")
        stack_trace = diagnostic_info.get("stack_trace", "")

        # Try to match against known patterns
        for pattern in self._scorer.get_migration_patterns():
            if self._scorer.matches_pattern(pattern, error_message, stack_trace):
                fix_method = self._get_fix_method(pattern.fix_generator)
                if fix_method:
                    pattern_suggestions = fix_method(
                        error_message, stack_trace, test_file, python_file_path
                    )
                    suggestions.extend(pattern_suggestions)

        # Add general suggestions based on failure analysis
        if failure_analysis.is_migration_issue:
            suggestions.extend(
                self._formatter.generate_migration_suggestions(
                    failure_analysis, test_file, python_file_path
                )
            )

        # Sort by confidence and complexity
        return self._scorer.sort_suggestions(suggestions)

    def apply_simple_fixes(
        self, suggestions: list[FixSuggestion], python_file_path: Path
    ) -> list[FixSuggestion]:
        """
        Apply simple fixes automatically to the Python test file.

        Args:
            suggestions: List of fix suggestions
            python_file_path: Path to the Python test file

        Returns:
            List of successfully applied fixes
        """
        return self._formatter.apply_simple_fixes(suggestions, python_file_path)

    def recognize_common_patterns(self, error_message: str, stack_trace: str) -> list[str]:
        """
        Recognize common migration issue patterns from error messages and stack traces.

        Args:
            error_message: The error message from test failure
            stack_trace: The stack trace from test failure

        Returns:
            List of recognized pattern names
        """
        return self._scorer.recognize_common_patterns(error_message, stack_trace)

    def _get_fix_method(self, method_name: str):
        """Get the fix method from the appropriate strategy."""
        method_mapping = {
            "_generate_brobot_import_fix": self._import_strategy.generate_brobot_import_fix,
            "_generate_java_import_fix": self._import_strategy.generate_java_import_fix,
            "_generate_junit_annotation_fix": self._setup_strategy.generate_junit_annotation_fix,
            "_generate_junit_assertion_fix": self._assertion_strategy.generate_junit_assertion_fix,
            "_generate_spring_annotation_fix": self._dependency_strategy.generate_spring_annotation_fix,
            "_generate_mockito_fix": self._dependency_strategy.generate_mockito_fix,
            "_generate_java_syntax_fix": self._syntax_strategy.generate_java_syntax_fix,
            "_generate_indentation_fix": self._syntax_strategy.generate_indentation_fix,
        }
        return method_mapping.get(method_name)
