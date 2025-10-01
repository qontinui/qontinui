"""
Test failure analysis engine for distinguishing migration errors from code errors.
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.interfaces import FailureAnalyzer
    from ..core.models import FailureAnalysis, FailureType, SuspectedCause, TestFailure
else:
    try:
        from ..core.interfaces import FailureAnalyzer
        from ..core.models import FailureAnalysis, FailureType, SuspectedCause, TestFailure
    except ImportError:
        # For standalone execution
        import sys
        from pathlib import Path

        sys.path.append(str(Path(__file__).parent.parent))
        from core.interfaces import FailureAnalyzer
        from core.models import FailureAnalysis, FailureType, SuspectedCause, TestFailure


@dataclass
class FailurePattern:
    """Pattern for identifying specific types of failures."""

    pattern: str
    failure_type: FailureType
    suspected_cause: SuspectedCause
    confidence: float
    description: str


class TestFailureAnalyzer(FailureAnalyzer):
    """
    Analyzes test failures to categorize them and distinguish between
    migration errors and code errors.
    """

    def __init__(self):
        """Initialize the failure analyzer with predefined patterns."""
        self._migration_patterns = self._initialize_migration_patterns()
        self._code_patterns = self._initialize_code_patterns()
        self._syntax_patterns = self._initialize_syntax_patterns()

    def analyze_failure(self, failure: TestFailure) -> FailureAnalysis:
        """
        Analyze a test failure to determine its cause and provide diagnostic information.

        Args:
            failure: The test failure to analyze

        Returns:
            FailureAnalysis with categorization and suggestions
        """
        # Combine error message and stack trace for analysis
        full_error_text = f"{failure.error_message}\n{failure.stack_trace}"

        # Check for migration-specific patterns
        migration_confidence = self._check_migration_patterns(full_error_text)

        # Check for code-specific patterns
        code_confidence = self._check_code_patterns(full_error_text)

        # Determine primary cause based on confidence scores
        is_migration_issue = migration_confidence > code_confidence
        is_code_issue = code_confidence > migration_confidence

        # If confidence scores are close, it might be both or unknown
        confidence_diff = abs(migration_confidence - code_confidence)
        if confidence_diff < 0.2:
            # Low confidence difference - could be either
            overall_confidence = max(migration_confidence, code_confidence) * 0.7
        else:
            overall_confidence = max(migration_confidence, code_confidence)

        # Generate diagnostic information
        diagnostic_info = self._generate_diagnostic_info(failure, full_error_text)

        # Generate fix suggestions
        suggested_fixes = self._generate_fix_suggestions(
            failure, is_migration_issue, is_code_issue, diagnostic_info
        )

        return FailureAnalysis(
            is_migration_issue=is_migration_issue,
            is_code_issue=is_code_issue,
            confidence=overall_confidence,
            suggested_fixes=suggested_fixes,
            diagnostic_info=diagnostic_info,
        )

    def is_migration_issue(self, failure: TestFailure) -> bool:
        """
        Determine if failure is due to migration issues.

        Args:
            failure: The test failure to analyze

        Returns:
            True if likely a migration issue
        """
        full_error_text = f"{failure.error_message}\n{failure.stack_trace}"
        migration_confidence = self._check_migration_patterns(full_error_text)
        code_confidence = self._check_code_patterns(full_error_text)

        return migration_confidence > code_confidence

    def is_code_issue(self, failure: TestFailure) -> bool:
        """
        Determine if failure is due to code issues.

        Args:
            failure: The test failure to analyze

        Returns:
            True if likely a code issue
        """
        full_error_text = f"{failure.error_message}\n{failure.stack_trace}"
        migration_confidence = self._check_migration_patterns(full_error_text)
        code_confidence = self._check_code_patterns(full_error_text)

        return code_confidence > migration_confidence

    def suggest_fixes(self, analysis: FailureAnalysis) -> list[str]:
        """
        Suggest fixes for the analyzed failure.

        Args:
            analysis: The failure analysis

        Returns:
            List of suggested fixes
        """
        return analysis.suggested_fixes

    def _initialize_migration_patterns(self) -> list[FailurePattern]:
        """Initialize patterns that indicate migration-related issues."""
        return [
            FailurePattern(
                pattern=r"ModuleNotFoundError.*brobot",
                failure_type=FailureType.DEPENDENCY_ERROR,
                suspected_cause=SuspectedCause.MIGRATION_ISSUE,
                confidence=0.9,
                description="Brobot module not properly migrated to Qontinui equivalent",
            ),
            FailurePattern(
                pattern=r"ImportError.*java\.",
                failure_type=FailureType.DEPENDENCY_ERROR,
                suspected_cause=SuspectedCause.MIGRATION_ISSUE,
                confidence=0.95,
                description="Java-specific import not converted to Python equivalent",
            ),
            FailurePattern(
                pattern=r"NameError.*@Test|@Before|@After",
                failure_type=FailureType.SYNTAX_ERROR,
                suspected_cause=SuspectedCause.MIGRATION_ISSUE,
                confidence=0.85,
                description="JUnit annotations not properly converted to pytest equivalents",
            ),
            FailurePattern(
                pattern=r"AttributeError.*assertEquals|assertTrue|assertFalse",
                failure_type=FailureType.ASSERTION_ERROR,
                suspected_cause=SuspectedCause.MIGRATION_ISSUE,
                confidence=0.8,
                description="JUnit assertions not converted to pytest assertions",
            ),
            FailurePattern(
                pattern=r"TypeError.*mock.*Brobot",
                failure_type=FailureType.MOCK_ERROR,
                suspected_cause=SuspectedCause.MIGRATION_ISSUE,
                confidence=0.85,
                description="Brobot mock not properly migrated to Qontinui mock",
            ),
            FailurePattern(
                pattern=r"SyntaxError.*@.*\(",
                failure_type=FailureType.SYNTAX_ERROR,
                suspected_cause=SuspectedCause.MIGRATION_ISSUE,
                confidence=0.7,
                description="Java annotation syntax not converted to Python decorator syntax",
            ),
            FailurePattern(
                pattern=r"AttributeError.*SpringBootTest|@Autowired",
                failure_type=FailureType.DEPENDENCY_ERROR,
                suspected_cause=SuspectedCause.MIGRATION_ISSUE,
                confidence=0.9,
                description="Spring Boot annotations not migrated to Python dependency injection",
            ),
            FailurePattern(
                pattern=r"NameError.*setUp|tearDown",
                failure_type=FailureType.SYNTAX_ERROR,
                suspected_cause=SuspectedCause.MIGRATION_ISSUE,
                confidence=0.75,
                description="JUnit setup/teardown methods not converted to pytest fixtures",
            ),
        ]

    def _initialize_code_patterns(self) -> list[FailurePattern]:
        """Initialize patterns that indicate code-related issues."""
        return [
            FailurePattern(
                pattern=r"AssertionError.*expected.*but was",
                failure_type=FailureType.ASSERTION_ERROR,
                suspected_cause=SuspectedCause.CODE_ISSUE,
                confidence=0.8,
                description="Test assertion failed - actual behavior differs from expected",
            ),
            FailurePattern(
                pattern=r"AttributeError.*'NoneType'",
                failure_type=FailureType.RUNTIME_ERROR,
                suspected_cause=SuspectedCause.CODE_ISSUE,
                confidence=0.7,
                description="Null pointer equivalent - object not properly initialized",
            ),
            FailurePattern(
                pattern=r"IndexError|KeyError",
                failure_type=FailureType.RUNTIME_ERROR,
                suspected_cause=SuspectedCause.CODE_ISSUE,
                confidence=0.75,
                description="Data structure access error - likely logic issue",
            ),
            FailurePattern(
                pattern=r"ValueError.*invalid literal",
                failure_type=FailureType.RUNTIME_ERROR,
                suspected_cause=SuspectedCause.CODE_ISSUE,
                confidence=0.8,
                description="Data conversion error - input validation issue",
            ),
            FailurePattern(
                pattern=r"TimeoutError|timeout",
                failure_type=FailureType.RUNTIME_ERROR,
                suspected_cause=SuspectedCause.CODE_ISSUE,
                confidence=0.6,
                description="Operation timeout - performance or logic issue",
            ),
            FailurePattern(
                pattern=r"FileNotFoundError",
                failure_type=FailureType.RUNTIME_ERROR,
                suspected_cause=SuspectedCause.ENVIRONMENT_ISSUE,
                confidence=0.7,
                description="Missing file - configuration or environment issue",
            ),
        ]

    def _initialize_syntax_patterns(self) -> list[FailurePattern]:
        """Initialize patterns for syntax-related issues."""
        return [
            FailurePattern(
                pattern=r"SyntaxError.*invalid syntax",
                failure_type=FailureType.SYNTAX_ERROR,
                suspected_cause=SuspectedCause.MIGRATION_ISSUE,
                confidence=0.9,
                description=(
                    "Python syntax error - likely from incomplete Java-to-Python conversion"
                ),
            ),
            FailurePattern(
                pattern=r"IndentationError",
                failure_type=FailureType.SYNTAX_ERROR,
                suspected_cause=SuspectedCause.MIGRATION_ISSUE,
                confidence=0.85,
                description="Python indentation error - Java braces not properly converted",
            ),
            FailurePattern(
                pattern=r"NameError.*name.*is not defined",
                failure_type=FailureType.SYNTAX_ERROR,
                suspected_cause=SuspectedCause.MIGRATION_ISSUE,
                confidence=0.8,
                description="Undefined name - Java variable or method not properly converted",
            ),
        ]

    def _check_migration_patterns(self, error_text: str) -> float:
        """Check error text against migration-specific patterns."""
        max_confidence = 0.0

        for pattern in self._migration_patterns:
            if re.search(pattern.pattern, error_text, re.IGNORECASE):
                max_confidence = max(max_confidence, pattern.confidence)

        # Also check syntax patterns as they're usually migration-related
        for pattern in self._syntax_patterns:
            if re.search(pattern.pattern, error_text, re.IGNORECASE):
                max_confidence = max(max_confidence, pattern.confidence)

        return max_confidence

    def _check_code_patterns(self, error_text: str) -> float:
        """Check error text against code-specific patterns."""
        max_confidence = 0.0

        for pattern in self._code_patterns:
            if re.search(pattern.pattern, error_text, re.IGNORECASE):
                max_confidence = max(max_confidence, pattern.confidence)

        return max_confidence

    def _generate_diagnostic_info(self, failure: TestFailure, error_text: str) -> dict[str, Any]:
        """Generate detailed diagnostic information."""
        diagnostic_info: dict[str, Any] = {
            "failure_type": failure.failure_type.value,
            "test_file": failure.test_file,
            "test_name": failure.test_name,
            "matched_patterns": [],
            "error_categories": set(),
            "migration_indicators": [],
            "code_indicators": [],
        }

        # Check all patterns and collect matches
        all_patterns = self._migration_patterns + self._code_patterns + self._syntax_patterns

        for pattern in all_patterns:
            if re.search(pattern.pattern, error_text, re.IGNORECASE):
                diagnostic_info["matched_patterns"].append(
                    {
                        "pattern": pattern.pattern,
                        "description": pattern.description,
                        "confidence": pattern.confidence,
                        "suspected_cause": pattern.suspected_cause.value,
                    }
                )

                diagnostic_info["error_categories"].add(pattern.failure_type.value)

                if pattern.suspected_cause == SuspectedCause.MIGRATION_ISSUE:
                    diagnostic_info["migration_indicators"].append(pattern.description)
                elif pattern.suspected_cause == SuspectedCause.CODE_ISSUE:
                    diagnostic_info["code_indicators"].append(pattern.description)

        # Convert set to list for JSON serialization
        diagnostic_info["error_categories"] = list(diagnostic_info["error_categories"])

        return diagnostic_info

    def _generate_fix_suggestions(
        self,
        failure: TestFailure,
        is_migration_issue: bool,
        is_code_issue: bool,
        diagnostic_info: dict[str, Any],
    ) -> list[str]:
        """Generate specific fix suggestions based on the analysis."""
        suggestions = []

        if is_migration_issue:
            suggestions.extend(self._get_migration_fix_suggestions(failure, diagnostic_info))

        if is_code_issue:
            suggestions.extend(self._get_code_fix_suggestions(failure, diagnostic_info))

        # Add general suggestions if no specific ones found
        if not suggestions:
            suggestions.extend(self._get_general_fix_suggestions(failure))

        return suggestions

    def _get_migration_fix_suggestions(
        self, failure: TestFailure, diagnostic_info: dict[str, Any]
    ) -> list[str]:
        """Get fix suggestions for migration-related issues."""
        suggestions = []

        error_text = f"{failure.error_message}\n{failure.stack_trace}"

        if "brobot" in error_text.lower():
            suggestions.append("Replace Brobot imports with equivalent Qontinui imports")
            suggestions.append("Update Brobot mock usage to use Qontinui mocks")

        if re.search(r"@Test|@Before|@After", error_text):
            suggestions.append("Convert JUnit annotations to pytest fixtures and decorators")
            suggestions.append("Replace @Test with pytest test function naming convention (test_*)")

        if re.search(r"assertEquals|assertTrue|assertFalse", error_text):
            suggestions.append("Convert JUnit assertions to pytest assertions (assert statements)")
            suggestions.append(
                "Update assertion syntax from assertEquals(expected, actual) "
                "to assert actual == expected"
            )

        if "SpringBootTest" in error_text or "Autowired" in error_text:
            suggestions.append(
                "Replace Spring Boot annotations with Python dependency injection patterns"
            )
            suggestions.append("Set up Python application context equivalent for integration tests")

        if "SyntaxError" in error_text:
            suggestions.append("Review Java-to-Python syntax conversion for completeness")
            suggestions.append("Check for proper Python indentation and syntax")

        return suggestions

    def _get_code_fix_suggestions(
        self, failure: TestFailure, diagnostic_info: dict[str, Any]
    ) -> list[str]:
        """Get fix suggestions for code-related issues."""
        suggestions = []

        error_text = f"{failure.error_message}\n{failure.stack_trace}"

        if "AssertionError" in error_text:
            suggestions.append("Review test expectations - the actual behavior may have changed")
            suggestions.append(
                "Verify that the migrated code produces equivalent results to the Java version"
            )
            suggestions.append("Check if test data or setup conditions are correctly migrated")

        if "NoneType" in error_text:
            suggestions.append("Check for proper object initialization in the migrated code")
            suggestions.append("Verify that null handling is correctly implemented in Python")
            suggestions.append("Review dependency injection and object creation patterns")

        if "IndexError" in error_text or "KeyError" in error_text:
            suggestions.append("Verify data structure access patterns in migrated code")
            suggestions.append("Check for off-by-one errors in array/list indexing")
            suggestions.append("Ensure dictionary keys exist before access")

        if "TimeoutError" in error_text or "timeout" in error_text.lower():
            suggestions.append("Review performance characteristics of migrated code")
            suggestions.append("Check for infinite loops or blocking operations")
            suggestions.append("Adjust timeout values for Python execution characteristics")

        return suggestions

    def _get_general_fix_suggestions(self, failure: TestFailure) -> list[str]:
        """Get general fix suggestions when specific patterns don't match."""
        return [
            "Review the original Java test to understand expected behavior",
            "Compare the migrated Python code with the original Java implementation",
            "Check test setup and teardown procedures for completeness",
            "Verify that all dependencies are properly configured",
            "Run the test in isolation to check for environmental issues",
            "Review error logs for additional context clues",
        ]
