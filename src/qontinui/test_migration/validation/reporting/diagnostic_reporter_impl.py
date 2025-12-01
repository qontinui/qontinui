"""
Diagnostic reporter facade that coordinates all reporting components.
"""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.interfaces import DiagnosticReporter
    from ...core.models import FailureAnalysis, TestFile, TestResults
else:
    try:
        from ...core.interfaces import DiagnosticReporter
        from ...core.models import FailureAnalysis, TestFile, TestResults
    except ImportError:
        from core.interfaces import DiagnosticReporter
        from core.models import FailureAnalysis, TestFile, TestResults

from .comprehensive_analyzer import ComprehensiveAnalyzer
from .error_analyzer import ErrorAnalyzer
from .report_data_collector import ReportDataCollector
from .report_formatter import ReportFormatterFactory
from .report_models import (
    AssertionDifference,
    DependencyDifference,
    DiagnosticReport,
    SetupDifference,
)


class DiagnosticReporterImpl(DiagnosticReporter):
    """
    Facade coordinator for diagnostic reporting.

    Coordinates:
    - ReportDataCollector: Data extraction from test files
    - ErrorAnalyzer: Assertion and error analysis
    - ComprehensiveAnalyzer: Difference detection and metrics
    - ReportFormatter: Output formatting (text, JSON, HTML)
    """

    def __init__(self, format_type: str = "text") -> None:
        """
        Initialize the diagnostic reporter.

        Args:
            format_type: Output format type (default: "text")
        """
        self._data_collector = ReportDataCollector()
        self._error_analyzer = ErrorAnalyzer()
        self._comprehensive_analyzer = ComprehensiveAnalyzer()
        self._formatter = ReportFormatterFactory.create_formatter(format_type)

    def generate_failure_report(self, analysis: FailureAnalysis) -> str:
        """
        Generate a detailed failure analysis report.

        Args:
            analysis: The failure analysis to report on

        Returns:
            Formatted report string
        """
        return self._formatter.format_failure_report(analysis)

    def generate_migration_summary(self, results: TestResults) -> str:
        """
        Generate a summary of migration results.

        Args:
            results: Test execution results

        Returns:
            Formatted summary string
        """
        return self._formatter.format_migration_summary(results)

    def detect_dependency_differences(
        self, java_test: TestFile, python_test_path: Path
    ) -> list[DependencyDifference]:
        """
        Detect differences in dependencies between Java and Python tests.

        Args:
            java_test: Original Java test file
            python_test_path: Path to migrated Python test

        Returns:
            List of dependency differences
        """
        return self._comprehensive_analyzer.detect_dependency_differences(
            java_test, python_test_path
        )

    def detect_setup_differences(
        self, java_test: TestFile, python_test_path: Path
    ) -> list[SetupDifference]:
        """
        Detect differences in test setup between Java and Python tests.

        Args:
            java_test: Original Java test file
            python_test_path: Path to migrated Python test

        Returns:
            List of setup differences
        """
        return self._comprehensive_analyzer.detect_setup_differences(
            java_test, python_test_path
        )

    def compare_assertion_logic(
        self, java_test: TestFile, python_test_path: Path
    ) -> list[AssertionDifference]:
        """
        Compare assertion logic between original and migrated tests.

        Args:
            java_test: Original Java test file
            python_test_path: Path to migrated Python test

        Returns:
            List of assertion differences
        """
        return self._error_analyzer.compare_assertion_logic(java_test, python_test_path)

    def generate_comprehensive_report(
        self,
        java_test: TestFile,
        python_test_path: Path,
        failure_analysis: FailureAnalysis | None = None,
    ) -> DiagnosticReport:
        """
        Generate a comprehensive diagnostic report for a test migration.

        Args:
            java_test: Original Java test file
            python_test_path: Path to migrated Python test
            failure_analysis: Optional failure analysis if test failed

        Returns:
            Comprehensive diagnostic report
        """
        return self._comprehensive_analyzer.generate_comprehensive_report(
            java_test, python_test_path, failure_analysis
        )
