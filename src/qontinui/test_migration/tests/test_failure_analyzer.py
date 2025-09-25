"""
Unit tests for the TestFailureAnalyzer class.
"""

from qontinui.test_migration.core.models import FailureType, SuspectedCause, TestFailure
from qontinui.test_migration.validation.test_failure_analyzer import (
    FailurePattern,
    TestFailureAnalyzer,
)


class TestTestFailureAnalyzer:
    """Test cases for TestFailureAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TestFailureAnalyzer()

    def test_initialization(self):
        """Test that analyzer initializes with proper patterns."""
        assert len(self.analyzer._migration_patterns) > 0
        assert len(self.analyzer._code_patterns) > 0
        assert len(self.analyzer._syntax_patterns) > 0

    def test_analyze_brobot_import_error(self):
        """Test analysis of Brobot import error (migration issue)."""
        failure = TestFailure(
            test_name="test_gui_automation",
            test_file="test_automation.py",
            error_message="ModuleNotFoundError: No module named 'brobot.library'",
            stack_trace="  File 'test_automation.py', line 5, in <module>\n    from brobot.library import Action",
            failure_type=FailureType.DEPENDENCY_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        analysis = self.analyzer.analyze_failure(failure)

        assert analysis.is_migration_issue is True
        assert analysis.is_code_issue is False
        assert analysis.confidence > 0.8
        assert any("Brobot" in fix for fix in analysis.suggested_fixes)
        assert "brobot" in str(analysis.diagnostic_info).lower()

    def test_analyze_java_import_error(self):
        """Test analysis of Java import error (migration issue)."""
        failure = TestFailure(
            test_name="test_spring_integration",
            test_file="test_integration.py",
            error_message="ImportError: cannot import name 'java.util.List'",
            stack_trace="  File 'test_integration.py', line 3\n    from java.util import List",
            failure_type=FailureType.DEPENDENCY_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        analysis = self.analyzer.analyze_failure(failure)

        assert analysis.is_migration_issue is True
        assert analysis.confidence > 0.9
        assert "Java-specific import" in str(analysis.diagnostic_info)

    def test_analyze_junit_annotation_error(self):
        """Test analysis of JUnit annotation error (migration issue)."""
        failure = TestFailure(
            test_name="test_user_login",
            test_file="test_user.py",
            error_message="NameError: name '@Test' is not defined",
            stack_trace="  File 'test_user.py', line 10\n    @Test\n    ^",
            failure_type=FailureType.SYNTAX_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        analysis = self.analyzer.analyze_failure(failure)

        assert analysis.is_migration_issue is True
        assert analysis.confidence > 0.8
        assert any("JUnit" in fix for fix in analysis.suggested_fixes)

    def test_analyze_junit_assertion_error(self):
        """Test analysis of JUnit assertion error (migration issue)."""
        failure = TestFailure(
            test_name="test_calculation",
            test_file="test_math.py",
            error_message="AttributeError: module 'unittest' has no attribute 'assertEquals'",
            stack_trace="  File 'test_math.py', line 15\n    assertEquals(expected, actual)",
            failure_type=FailureType.ASSERTION_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        analysis = self.analyzer.analyze_failure(failure)

        assert analysis.is_migration_issue is True
        assert analysis.confidence > 0.7
        assert any("assertion" in fix.lower() for fix in analysis.suggested_fixes)

    def test_analyze_spring_boot_error(self):
        """Test analysis of Spring Boot annotation error (migration issue)."""
        failure = TestFailure(
            test_name="test_service_integration",
            test_file="test_service.py",
            error_message="AttributeError: 'TestClass' has no attribute 'SpringBootTest'",
            stack_trace="  File 'test_service.py', line 8\n    @SpringBootTest",
            failure_type=FailureType.DEPENDENCY_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        analysis = self.analyzer.analyze_failure(failure)

        assert analysis.is_migration_issue is True
        assert analysis.confidence > 0.8
        assert any("Spring Boot" in fix for fix in analysis.suggested_fixes)

    def test_analyze_assertion_failure_code_issue(self):
        """Test analysis of assertion failure (code issue)."""
        failure = TestFailure(
            test_name="test_business_logic",
            test_file="test_logic.py",
            error_message="AssertionError: expected 42 but was 24",
            stack_trace="  File 'test_logic.py', line 20\n    assert result == 42",
            failure_type=FailureType.ASSERTION_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        analysis = self.analyzer.analyze_failure(failure)

        assert analysis.is_code_issue is True
        assert analysis.confidence > 0.7
        assert any("behavior" in fix.lower() for fix in analysis.suggested_fixes)

    def test_analyze_none_type_error_code_issue(self):
        """Test analysis of NoneType error (code issue)."""
        failure = TestFailure(
            test_name="test_object_creation",
            test_file="test_objects.py",
            error_message="AttributeError: 'NoneType' object has no attribute 'process'",
            stack_trace="  File 'test_objects.py', line 25\n    result.process()",
            failure_type=FailureType.RUNTIME_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        analysis = self.analyzer.analyze_failure(failure)

        assert analysis.is_code_issue is True
        assert analysis.confidence > 0.6
        assert any("initialization" in fix.lower() for fix in analysis.suggested_fixes)

    def test_analyze_syntax_error_migration_issue(self):
        """Test analysis of Python syntax error (migration issue)."""
        failure = TestFailure(
            test_name="test_converted_method",
            test_file="test_converted.py",
            error_message="SyntaxError: invalid syntax",
            stack_trace="  File 'test_converted.py', line 12\n    if (condition) {\n                   ^",
            failure_type=FailureType.SYNTAX_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        analysis = self.analyzer.analyze_failure(failure)

        assert analysis.is_migration_issue is True
        assert analysis.confidence > 0.8
        assert any("syntax" in fix.lower() for fix in analysis.suggested_fixes)

    def test_analyze_indentation_error_migration_issue(self):
        """Test analysis of indentation error (migration issue)."""
        failure = TestFailure(
            test_name="test_indentation",
            test_file="test_indent.py",
            error_message="IndentationError: expected an indented block",
            stack_trace="  File 'test_indent.py', line 18\n    return result\n    ^",
            failure_type=FailureType.SYNTAX_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        analysis = self.analyzer.analyze_failure(failure)

        assert analysis.is_migration_issue is True
        assert analysis.confidence > 0.8
        assert any("indentation" in fix.lower() for fix in analysis.suggested_fixes)

    def test_analyze_timeout_error_code_issue(self):
        """Test analysis of timeout error (code issue)."""
        failure = TestFailure(
            test_name="test_long_operation",
            test_file="test_performance.py",
            error_message="TimeoutError: Operation timed out after 30 seconds",
            stack_trace="  File 'test_performance.py', line 30\n    result = long_running_operation()",
            failure_type=FailureType.RUNTIME_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        analysis = self.analyzer.analyze_failure(failure)

        assert analysis.is_code_issue is True
        assert analysis.confidence > 0.5
        assert any(
            "timeout" in fix.lower() or "performance" in fix.lower()
            for fix in analysis.suggested_fixes
        )

    def test_is_migration_issue_method(self):
        """Test the is_migration_issue method directly."""
        migration_failure = TestFailure(
            test_name="test_migration",
            test_file="test.py",
            error_message="ModuleNotFoundError: No module named 'brobot'",
            stack_trace="",
            failure_type=FailureType.DEPENDENCY_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        code_failure = TestFailure(
            test_name="test_code",
            test_file="test.py",
            error_message="AssertionError: expected 5 but was 3",
            stack_trace="",
            failure_type=FailureType.ASSERTION_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        assert self.analyzer.is_migration_issue(migration_failure) is True
        assert self.analyzer.is_migration_issue(code_failure) is False

    def test_is_code_issue_method(self):
        """Test the is_code_issue method directly."""
        migration_failure = TestFailure(
            test_name="test_migration",
            test_file="test.py",
            error_message="ImportError: cannot import name 'java.util.List'",
            stack_trace="",
            failure_type=FailureType.DEPENDENCY_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        code_failure = TestFailure(
            test_name="test_code",
            test_file="test.py",
            error_message="ValueError: invalid literal for int() with base 10: 'abc'",
            stack_trace="",
            failure_type=FailureType.RUNTIME_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        assert self.analyzer.is_code_issue(migration_failure) is False
        assert self.analyzer.is_code_issue(code_failure) is True

    def test_suggest_fixes_method(self):
        """Test the suggest_fixes method."""
        failure = TestFailure(
            test_name="test_example",
            test_file="test.py",
            error_message="ModuleNotFoundError: No module named 'brobot'",
            stack_trace="",
            failure_type=FailureType.DEPENDENCY_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        analysis = self.analyzer.analyze_failure(failure)
        fixes = self.analyzer.suggest_fixes(analysis)

        assert len(fixes) > 0
        assert fixes == analysis.suggested_fixes

    def test_confidence_scoring_close_scores(self):
        """Test confidence scoring when migration and code scores are close."""
        # Create a failure that could be either migration or code issue
        failure = TestFailure(
            test_name="test_ambiguous",
            test_file="test.py",
            error_message="Some generic error message",
            stack_trace="Generic stack trace",
            failure_type=FailureType.RUNTIME_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        analysis = self.analyzer.analyze_failure(failure)

        # When no clear patterns match, confidence should be lower
        assert analysis.confidence < 0.8

    def test_diagnostic_info_generation(self):
        """Test that diagnostic info is properly generated."""
        failure = TestFailure(
            test_name="test_diagnostic",
            test_file="test_file.py",
            error_message="ModuleNotFoundError: No module named 'brobot'",
            stack_trace="Stack trace here",
            failure_type=FailureType.DEPENDENCY_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        analysis = self.analyzer.analyze_failure(failure)

        assert "failure_type" in analysis.diagnostic_info
        assert "test_file" in analysis.diagnostic_info
        assert "test_name" in analysis.diagnostic_info
        assert "matched_patterns" in analysis.diagnostic_info
        assert "migration_indicators" in analysis.diagnostic_info
        assert "code_indicators" in analysis.diagnostic_info

        assert analysis.diagnostic_info["test_file"] == "test_file.py"
        assert analysis.diagnostic_info["test_name"] == "test_diagnostic"

    def test_multiple_pattern_matches(self):
        """Test behavior when multiple patterns match."""
        failure = TestFailure(
            test_name="test_multiple",
            test_file="test.py",
            error_message="ModuleNotFoundError: No module named 'brobot'. SyntaxError: invalid syntax",
            stack_trace="Multiple errors in stack trace",
            failure_type=FailureType.DEPENDENCY_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        analysis = self.analyzer.analyze_failure(failure)

        # Should still identify as migration issue with high confidence
        assert analysis.is_migration_issue is True
        assert analysis.confidence > 0.8
        assert len(analysis.diagnostic_info["matched_patterns"]) > 1

    def test_failure_pattern_dataclass(self):
        """Test the FailurePattern dataclass."""
        pattern = FailurePattern(
            pattern=r"test_pattern",
            failure_type=FailureType.SYNTAX_ERROR,
            suspected_cause=SuspectedCause.MIGRATION_ISSUE,
            confidence=0.85,
            description="Test pattern description",
        )

        assert pattern.pattern == r"test_pattern"
        assert pattern.failure_type == FailureType.SYNTAX_ERROR
        assert pattern.suspected_cause == SuspectedCause.MIGRATION_ISSUE
        assert pattern.confidence == 0.85
        assert pattern.description == "Test pattern description"

    def test_empty_error_message(self):
        """Test handling of empty error messages."""
        failure = TestFailure(
            test_name="test_empty",
            test_file="test.py",
            error_message="",
            stack_trace="",
            failure_type=FailureType.RUNTIME_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        analysis = self.analyzer.analyze_failure(failure)

        # Should still provide some analysis even with empty messages
        assert isinstance(analysis.is_migration_issue, bool)
        assert isinstance(analysis.is_code_issue, bool)
        assert isinstance(analysis.confidence, float)
        assert isinstance(analysis.suggested_fixes, list)

    def test_case_insensitive_pattern_matching(self):
        """Test that pattern matching is case insensitive."""
        failure = TestFailure(
            test_name="test_case",
            test_file="test.py",
            error_message="MODULENOTFOUNDERROR: No module named 'BROBOT'",
            stack_trace="",
            failure_type=FailureType.DEPENDENCY_ERROR,
            suspected_cause=SuspectedCause.UNKNOWN,
        )

        analysis = self.analyzer.analyze_failure(failure)

        # Should still match despite different case
        assert analysis.is_migration_issue is True
        assert analysis.confidence > 0.8
