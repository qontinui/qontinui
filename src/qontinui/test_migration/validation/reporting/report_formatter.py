"""
Report formatting with Strategy pattern for multiple output formats.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.models import FailureAnalysis, TestResults
else:
    try:
        from ...core.models import FailureAnalysis, TestResults
    except ImportError:
        from core.models import FailureAnalysis, TestResults


class ReportFormatter(ABC):
    """Abstract base class for report formatters."""

    @abstractmethod
    def format_failure_report(self, analysis: FailureAnalysis) -> str:
        """Format a failure analysis report."""
        pass

    @abstractmethod
    def format_migration_summary(self, results: TestResults) -> str:
        """Format a migration summary report."""
        pass


class TextReportFormatter(ReportFormatter):
    """
    Formats reports as plain text with structured sections.

    Responsibilities:
    - Generate text-based failure reports
    - Generate text-based migration summaries
    - Format with clear headers and sections
    """

    def format_failure_report(self, analysis: FailureAnalysis) -> str:
        """
        Generate a detailed failure analysis report in text format.

        Args:
            analysis: The failure analysis to report on

        Returns:
            Formatted text report
        """
        report_lines = []

        # Header
        report_lines.append("=" * 80)
        report_lines.append("TEST FAILURE DIAGNOSTIC REPORT")
        report_lines.append("=" * 80)
        report_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append("")

        # Summary
        report_lines.append("FAILURE SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(
            f"Migration Issue: {'YES' if analysis.is_migration_issue else 'NO'}"
        )
        report_lines.append(f"Code Issue: {'YES' if analysis.is_code_issue else 'NO'}")
        report_lines.append(f"Confidence: {analysis.confidence:.2%}")
        report_lines.append("")

        # Diagnostic Information
        if analysis.diagnostic_info:
            report_lines.append("DIAGNOSTIC DETAILS")
            report_lines.append("-" * 40)

            # Test information
            test_file = analysis.diagnostic_info.get("test_file", "Unknown")
            test_name = analysis.diagnostic_info.get("test_name", "Unknown")
            report_lines.append(f"Test File: {test_file}")
            report_lines.append(f"Test Name: {test_name}")
            report_lines.append("")

            # Matched patterns
            matched_patterns = analysis.diagnostic_info.get("matched_patterns", [])
            if matched_patterns:
                report_lines.append("MATCHED ERROR PATTERNS")
                report_lines.append("-" * 30)
                for i, pattern in enumerate(matched_patterns, 1):
                    report_lines.append(
                        f"{i}. {pattern.get('description', 'Unknown pattern')}"
                    )
                    report_lines.append(f"   Pattern: {pattern.get('pattern', 'N/A')}")
                    report_lines.append(
                        f"   Confidence: {pattern.get('confidence', 0):.2%}"
                    )
                report_lines.append("")

        # Suggested fixes
        if analysis.suggested_fixes:
            report_lines.append("SUGGESTED FIXES")
            report_lines.append("-" * 40)
            for i, fix in enumerate(analysis.suggested_fixes, 1):
                report_lines.append(f"{i}. {fix}")
            report_lines.append("")

        return "\n".join(report_lines)

    def format_migration_summary(self, results: TestResults) -> str:
        """
        Generate a summary of migration results in text format.

        Args:
            results: Test execution results

        Returns:
            Formatted text summary
        """
        summary_lines = []

        # Header
        summary_lines.append("=" * 80)
        summary_lines.append("TEST MIGRATION SUMMARY")
        summary_lines.append("=" * 80)
        summary_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        summary_lines.append("")

        # Overall statistics
        summary_lines.append("OVERALL STATISTICS")
        summary_lines.append("-" * 40)
        summary_lines.append(f"Total Tests: {results.total_tests}")
        summary_lines.append(f"Passed: {results.passed_tests}")
        summary_lines.append(f"Failed: {results.failed_tests}")
        summary_lines.append(f"Skipped: {results.skipped_tests}")

        if results.total_tests > 0:
            summary_lines.append(
                f"Success Rate: {(results.passed_tests / results.total_tests * 100):.1f}%"
            )

        summary_lines.append(f"Total Execution Time: {results.execution_time:.2f}s")
        summary_lines.append("")

        # Failed tests details
        if results.failed_tests > 0:
            failed_results = [r for r in results.individual_results if not r.passed]
            summary_lines.append("FAILED TESTS")
            summary_lines.append("-" * 40)
            for result in failed_results:
                summary_lines.append(f"• {result.test_name} ({result.test_file})")
                if result.error_message:
                    summary_lines.append(f"  Error: {result.error_message}")
            summary_lines.append("")

        return "\n".join(summary_lines)


class ReportFormatterFactory:
    """Factory for creating report formatters."""

    _formatters: dict[str, type[ReportFormatter]] = {
        "text": TextReportFormatter,
    }

    @classmethod
    def create_formatter(cls, format_type: str = "text") -> ReportFormatter:
        """
        Create a report formatter of the specified type.

        Args:
            format_type: Type of formatter to create

        Returns:
            Report formatter instance

        Raises:
            ValueError: If format type is not supported
        """
        formatter_class = cls._formatters.get(format_type.lower())
        if not formatter_class:
            raise ValueError(f"Unsupported format type: {format_type}")
        return formatter_class()

    @classmethod
    def register_formatter(
        cls, format_type: str, formatter_class: type[ReportFormatter]
    ) -> None:
        """
        Register a new formatter type.

        Args:
            format_type: Name of the format type
            formatter_class: Formatter class to register
        """
        cls._formatters[format_type.lower()] = formatter_class
