"""Vision verification result builders and formatters.

Provides utilities for building and formatting assertion results
with rich error information for debugging and AI analysis.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qontinui_schemas.testing.assertions import (
    AssertionResult,
    AssertionStatus,
    AssertionSuiteResult,
    VisionLocatorMatch,
)

logger = logging.getLogger(__name__)


class ResultBuilder:
    """Builder for creating assertion results."""

    def __init__(
        self,
        assertion_id: str,
        assertion_method: str,
    ) -> None:
        """Initialize result builder.

        Args:
            assertion_id: ID of the assertion.
            assertion_method: Method name being executed.
        """
        self._assertion_id = assertion_id
        self._assertion_method = assertion_method
        self._started_at = datetime.now(UTC)
        self._completed_at: datetime | None = None
        self._status = AssertionStatus.PENDING
        self._retry_count = 0
        self._matches: list[VisionLocatorMatch] = []
        self._expected_value: Any = None
        self._actual_value: Any = None
        self._error_message: str | None = None
        self._error_details: dict[str, Any] = {}
        self._suggestion: str | None = None
        self._screenshot_path: str | None = None
        self._annotated_screenshot_path: str | None = None
        self._diff_screenshot_path: str | None = None

    def set_status(self, status: AssertionStatus) -> "ResultBuilder":
        """Set the result status.

        Args:
            status: Result status.

        Returns:
            Self for chaining.
        """
        self._status = status
        return self

    def set_passed(self) -> "ResultBuilder":
        """Mark result as passed.

        Returns:
            Self for chaining.
        """
        self._status = AssertionStatus.PASSED
        return self

    def set_failed(
        self,
        message: str | None = None,
        suggestion: str | None = None,
    ) -> "ResultBuilder":
        """Mark result as failed.

        Args:
            message: Error message.
            suggestion: Suggested fix.

        Returns:
            Self for chaining.
        """
        self._status = AssertionStatus.FAILED
        if message:
            self._error_message = message
        if suggestion:
            self._suggestion = suggestion
        return self

    def set_error(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> "ResultBuilder":
        """Mark result as error.

        Args:
            message: Error message.
            details: Error details.

        Returns:
            Self for chaining.
        """
        self._status = AssertionStatus.ERROR
        self._error_message = message
        if details:
            self._error_details.update(details)
        return self

    def set_skipped(self, reason: str | None = None) -> "ResultBuilder":
        """Mark result as skipped.

        Args:
            reason: Skip reason.

        Returns:
            Self for chaining.
        """
        self._status = AssertionStatus.SKIPPED
        if reason:
            self._error_message = reason
        return self

    def set_expected(self, value: Any) -> "ResultBuilder":
        """Set expected value.

        Args:
            value: Expected value.

        Returns:
            Self for chaining.
        """
        self._expected_value = value
        return self

    def set_actual(self, value: Any) -> "ResultBuilder":
        """Set actual value.

        Args:
            value: Actual value.

        Returns:
            Self for chaining.
        """
        self._actual_value = value
        return self

    def add_match(self, match: VisionLocatorMatch) -> "ResultBuilder":
        """Add a locator match.

        Args:
            match: Match to add.

        Returns:
            Self for chaining.
        """
        self._matches.append(match)
        return self

    def set_matches(self, matches: list[VisionLocatorMatch]) -> "ResultBuilder":
        """Set all matches.

        Args:
            matches: List of matches.

        Returns:
            Self for chaining.
        """
        self._matches = matches
        return self

    def increment_retry(self) -> "ResultBuilder":
        """Increment retry count.

        Returns:
            Self for chaining.
        """
        self._retry_count += 1
        return self

    def set_screenshot(self, path: str | Path) -> "ResultBuilder":
        """Set screenshot path.

        Args:
            path: Screenshot path.

        Returns:
            Self for chaining.
        """
        self._screenshot_path = str(path)
        return self

    def set_annotated_screenshot(self, path: str | Path) -> "ResultBuilder":
        """Set annotated screenshot path.

        Args:
            path: Annotated screenshot path.

        Returns:
            Self for chaining.
        """
        self._annotated_screenshot_path = str(path)
        return self

    def set_diff_screenshot(self, path: str | Path) -> "ResultBuilder":
        """Set diff screenshot path.

        Args:
            path: Diff screenshot path.

        Returns:
            Self for chaining.
        """
        self._diff_screenshot_path = str(path)
        return self

    def add_detail(self, key: str, value: Any) -> "ResultBuilder":
        """Add error detail.

        Args:
            key: Detail key.
            value: Detail value.

        Returns:
            Self for chaining.
        """
        self._error_details[key] = value
        return self

    def set_suggestion(self, suggestion: str) -> "ResultBuilder":
        """Set suggestion.

        Args:
            suggestion: Suggested fix.

        Returns:
            Self for chaining.
        """
        self._suggestion = suggestion
        return self

    def build(self) -> AssertionResult:
        """Build the final assertion result.

        Returns:
            AssertionResult instance.
        """
        self._completed_at = datetime.now(UTC)
        duration_ms = int((self._completed_at - self._started_at).total_seconds() * 1000)

        best_match = self._matches[0] if self._matches else None

        return AssertionResult(
            assertion_id=self._assertion_id,
            assertion_method=self._assertion_method,
            status=self._status,
            started_at=self._started_at,
            completed_at=self._completed_at,
            duration_ms=duration_ms,
            retry_count=self._retry_count,
            matches_found=len(self._matches),
            best_match=best_match,
            all_matches=self._matches,
            expected_value=self._expected_value,
            actual_value=self._actual_value,
            error_message=self._error_message,
            error_details=self._error_details,
            suggestion=self._suggestion,
            screenshot_path=self._screenshot_path,
            annotated_screenshot_path=self._annotated_screenshot_path,
            diff_screenshot_path=self._diff_screenshot_path,
        )


class SuiteResultBuilder:
    """Builder for creating assertion suite results."""

    def __init__(
        self,
        suite_id: str,
        suite_name: str | None = None,
    ) -> None:
        """Initialize suite result builder.

        Args:
            suite_id: Suite execution ID.
            suite_name: Optional suite name.
        """
        self._suite_id = suite_id
        self._suite_name = suite_name
        self._started_at = datetime.now(UTC)
        self._results: list[AssertionResult] = []
        self._environment_id: str | None = None

    def add_result(self, result: AssertionResult) -> "SuiteResultBuilder":
        """Add an assertion result.

        Args:
            result: Result to add.

        Returns:
            Self for chaining.
        """
        self._results.append(result)
        return self

    def set_environment_id(self, env_id: str) -> "SuiteResultBuilder":
        """Set environment ID.

        Args:
            env_id: Environment ID.

        Returns:
            Self for chaining.
        """
        self._environment_id = env_id
        return self

    def build(self) -> AssertionSuiteResult:
        """Build the final suite result.

        Returns:
            AssertionSuiteResult instance.
        """
        completed_at = datetime.now(UTC)
        total_duration_ms = int((completed_at - self._started_at).total_seconds() * 1000)

        passed = sum(1 for r in self._results if r.status == AssertionStatus.PASSED)
        failed = sum(1 for r in self._results if r.status == AssertionStatus.FAILED)
        errors = sum(1 for r in self._results if r.status == AssertionStatus.ERROR)
        skipped = sum(1 for r in self._results if r.status == AssertionStatus.SKIPPED)

        total = len(self._results)
        pass_rate = passed / total if total > 0 else 0.0

        return AssertionSuiteResult(
            suite_id=self._suite_id,
            suite_name=self._suite_name,
            started_at=self._started_at,
            completed_at=completed_at,
            total_duration_ms=total_duration_ms,
            results=self._results,
            total_assertions=total,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            pass_rate=pass_rate,
            environment_id=self._environment_id,
        )


def format_result_for_ai(result: AssertionResult) -> str:
    """Format assertion result for AI analysis context.

    Args:
        result: Assertion result.

    Returns:
        Formatted string for AI context.
    """
    lines = [
        f"Assertion: {result.assertion_method}",
        f"Status: {result.status.value.upper()}",
        f"Duration: {result.duration_ms}ms",
    ]

    if result.expected_value is not None:
        lines.append(f"Expected: {result.expected_value}")
    if result.actual_value is not None:
        lines.append(f"Actual: {result.actual_value}")

    if result.matches_found > 0:
        lines.append(f"Matches found: {result.matches_found}")
        if result.best_match:
            lines.append(f"Best match confidence: {result.best_match.confidence:.2%}")

    if result.error_message:
        lines.append(f"Error: {result.error_message}")

    if result.error_details:
        lines.append("Details:")
        for key, value in result.error_details.items():
            lines.append(f"  {key}: {value}")

    if result.suggestion:
        lines.append(f"Suggestion: {result.suggestion}")

    if result.screenshot_path:
        lines.append(f"Screenshot: {result.screenshot_path}")
    if result.annotated_screenshot_path:
        lines.append(f"Annotated: {result.annotated_screenshot_path}")

    return "\n".join(lines)


def format_suite_for_ai(suite: AssertionSuiteResult) -> str:
    """Format suite result for AI analysis context.

    Args:
        suite: Suite result.

    Returns:
        Formatted string for AI context.
    """
    lines = [
        "=== Vision Verification Results ===",
        "",
        f"Suite: {suite.suite_name or suite.suite_id}",
        f"Total: {suite.total_assertions} | Passed: {suite.passed} | "
        f"Failed: {suite.failed} | Errors: {suite.errors} | Skipped: {suite.skipped}",
        f"Pass Rate: {suite.pass_rate:.1%}",
        f"Duration: {suite.total_duration_ms}ms",
        "",
    ]

    # Group by status
    failed_results = [r for r in suite.results if r.status == AssertionStatus.FAILED]
    error_results = [r for r in suite.results if r.status == AssertionStatus.ERROR]
    passed_results = [r for r in suite.results if r.status == AssertionStatus.PASSED]

    if failed_results:
        lines.append("--- Failed Assertions ---")
        for result in failed_results:
            lines.append("")
            lines.append(format_result_for_ai(result))
        lines.append("")

    if error_results:
        lines.append("--- Error Assertions ---")
        for result in error_results:
            lines.append("")
            lines.append(format_result_for_ai(result))
        lines.append("")

    if passed_results:
        lines.append(f"--- Passed Assertions ({len(passed_results)}) ---")
        for result in passed_results:
            lines.append(f"  [PASS] {result.assertion_method} ({result.duration_ms}ms)")
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "ResultBuilder",
    "SuiteResultBuilder",
    "format_result_for_ai",
    "format_suite_for_ai",
]
