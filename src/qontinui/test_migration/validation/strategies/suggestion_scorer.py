"""
Confidence scoring logic for fix suggestions.
"""

import re

from ..fix_models import FixSuggestion, MigrationIssuePattern


class SuggestionScorer:
    """Handles confidence scoring and pattern matching for fix suggestions."""

    def __init__(self) -> None:
        """Initialize the suggestion scorer."""
        self._migration_patterns = self._initialize_migration_patterns()

    def _initialize_migration_patterns(self) -> list[MigrationIssuePattern]:
        """Initialize common migration issue patterns."""
        from ...core.models import FailureType

        return [
            MigrationIssuePattern(
                pattern_name="brobot_import_error",
                pattern_regex=r"modulenotfounderror.*brobot|no module named.*brobot",
                failure_types=[FailureType.DEPENDENCY_ERROR],
                fix_generator="_generate_brobot_import_fix",
                confidence_threshold=0.9,
                description="Brobot library import error",
            ),
            MigrationIssuePattern(
                pattern_name="java_import_error",
                pattern_regex=r"importerror.*java\.|no module named.*java\.",
                failure_types=[FailureType.DEPENDENCY_ERROR],
                fix_generator="_generate_java_import_fix",
                confidence_threshold=0.9,
                description="Java-specific import error",
            ),
            MigrationIssuePattern(
                pattern_name="junit_annotation_error",
                pattern_regex=r"nameerror.*@test|@test.*not defined|@beforeeach|@aftereach",
                failure_types=[FailureType.SYNTAX_ERROR],
                fix_generator="_generate_junit_annotation_fix",
                confidence_threshold=0.8,
                description="JUnit annotation error",
            ),
            MigrationIssuePattern(
                pattern_name="junit_assertion_error",
                pattern_regex=r"assertequals|asserttrue|assertfalse|assertnull|assertthrows",
                failure_types=[FailureType.ASSERTION_ERROR],
                fix_generator="_generate_junit_assertion_fix",
                confidence_threshold=0.8,
                description="JUnit assertion method error",
            ),
            MigrationIssuePattern(
                pattern_name="spring_annotation_error",
                pattern_regex=r"@springboottest|@autowired|@component|@service",
                failure_types=[FailureType.DEPENDENCY_ERROR, FailureType.SYNTAX_ERROR],
                fix_generator="_generate_spring_annotation_fix",
                confidence_threshold=0.7,
                description="Spring Boot annotation error",
            ),
            MigrationIssuePattern(
                pattern_name="mockito_error",
                pattern_regex=r"mockito|@mock|when\(.*\)\.then",
                failure_types=[FailureType.MOCK_ERROR, FailureType.DEPENDENCY_ERROR],
                fix_generator="_generate_mockito_fix",
                confidence_threshold=0.7,
                description="Mockito mocking framework error",
            ),
            MigrationIssuePattern(
                pattern_name="java_syntax_error",
                pattern_regex=r"syntaxerror.*{|}|invalid syntax.*{|}",
                failure_types=[FailureType.SYNTAX_ERROR],
                fix_generator="_generate_java_syntax_fix",
                confidence_threshold=0.8,
                description="Java syntax in Python code",
            ),
            MigrationIssuePattern(
                pattern_name="indentation_error",
                pattern_regex=r"indentationerror|expected an indented block",
                failure_types=[FailureType.SYNTAX_ERROR],
                fix_generator="_generate_indentation_fix",
                confidence_threshold=0.9,
                description="Python indentation error",
            ),
        ]

    def get_migration_patterns(self) -> list[MigrationIssuePattern]:
        """Get the list of migration patterns."""
        return self._migration_patterns

    def matches_pattern(
        self, pattern: MigrationIssuePattern, error_message: str, stack_trace: str
    ) -> bool:
        """Check if error matches a migration pattern."""
        combined_text = f"{error_message}\n{stack_trace}"
        return bool(re.search(pattern.pattern_regex, combined_text, re.IGNORECASE))

    def recognize_common_patterns(
        self, error_message: str, stack_trace: str
    ) -> list[str]:
        """
        Recognize common migration issue patterns from error messages and stack traces.

        Args:
            error_message: The error message from test failure
            stack_trace: The stack trace from test failure

        Returns:
            List of recognized pattern names
        """
        recognized_patterns = []

        combined_text = f"{error_message}\n{stack_trace}".lower()

        for pattern in self._migration_patterns:
            if re.search(pattern.pattern_regex, combined_text, re.IGNORECASE):
                recognized_patterns.append(pattern.pattern_name)

        return recognized_patterns

    def sort_suggestions(self, suggestions: list[FixSuggestion]) -> list[FixSuggestion]:
        """Sort suggestions by confidence and complexity."""
        return sorted(
            suggestions, key=lambda x: (x.confidence, x.complexity.value), reverse=True
        )
