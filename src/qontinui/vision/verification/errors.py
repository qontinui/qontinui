"""Vision verification error types.

Provides custom exceptions for vision assertion failures with rich
error messages and debugging information.
"""

from typing import Any


class VisionError(Exception):
    """Base exception for all vision verification errors."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Initialize vision error.

        Args:
            message: Human-readable error message.
            details: Additional error details for debugging.
            suggestion: Suggested fix for the error.
        """
        self.message = message
        self.details = details or {}
        self.suggestion = suggestion
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the full error message."""
        lines = [self.message]

        if self.details:
            lines.append("")
            lines.append("Details:")
            for key, value in self.details.items():
                lines.append(f"  {key}: {value}")

        if self.suggestion:
            lines.append("")
            lines.append(f"Suggestion: {self.suggestion}")

        return "\n".join(lines)


class AssertionError(VisionError):
    """Raised when a vision assertion fails."""

    def __init__(
        self,
        assertion_method: str,
        expected: Any,
        actual: Any,
        message: str | None = None,
        details: dict[str, Any] | None = None,
        suggestion: str | None = None,
        screenshot_path: str | None = None,
        annotated_screenshot_path: str | None = None,
    ) -> None:
        """Initialize assertion error.

        Args:
            assertion_method: The assertion method that failed.
            expected: Expected value.
            actual: Actual value found.
            message: Custom error message.
            details: Additional error details.
            suggestion: Suggested fix.
            screenshot_path: Path to failure screenshot.
            annotated_screenshot_path: Path to annotated screenshot.
        """
        self.assertion_method = assertion_method
        self.expected = expected
        self.actual = actual
        self.screenshot_path = screenshot_path
        self.annotated_screenshot_path = annotated_screenshot_path

        if message is None:
            message = f"Assertion failed: {assertion_method}"

        full_details = details or {}
        full_details["expected"] = expected
        full_details["actual"] = actual

        if screenshot_path:
            full_details["screenshot"] = screenshot_path
        if annotated_screenshot_path:
            full_details["annotated_screenshot"] = annotated_screenshot_path

        super().__init__(message, full_details, suggestion)


class ElementNotFoundError(VisionError):
    """Raised when a target element cannot be found."""

    def __init__(
        self,
        locator_type: str,
        locator_value: str,
        message: str | None = None,
        details: dict[str, Any] | None = None,
        suggestion: str | None = None,
        best_match_confidence: float | None = None,
    ) -> None:
        """Initialize element not found error.

        Args:
            locator_type: Type of locator used.
            locator_value: Locator value that wasn't found.
            message: Custom error message.
            details: Additional error details.
            suggestion: Suggested fix.
            best_match_confidence: Confidence of best (insufficient) match.
        """
        self.locator_type = locator_type
        self.locator_value = locator_value
        self.best_match_confidence = best_match_confidence

        if message is None:
            message = f"Element not found: {locator_type}={locator_value!r}"

        full_details = details or {}
        full_details["locator_type"] = locator_type
        full_details["locator_value"] = locator_value

        if best_match_confidence is not None:
            full_details["best_match_confidence"] = f"{best_match_confidence:.2f}"

        if suggestion is None and best_match_confidence is not None:
            if best_match_confidence > 0.5:
                suggestion = (
                    f"A similar element was found with confidence {best_match_confidence:.2f}. "
                    "Consider lowering the match threshold or updating the template."
                )
            else:
                suggestion = (
                    "No similar elements found. Verify the element is visible on screen "
                    "or check if the locator is correct."
                )

        super().__init__(message, full_details, suggestion)


class TimeoutError(VisionError):
    """Raised when an operation times out."""

    def __init__(
        self,
        operation: str,
        timeout_ms: int,
        message: str | None = None,
        details: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Initialize timeout error.

        Args:
            operation: Operation that timed out.
            timeout_ms: Timeout value in milliseconds.
            message: Custom error message.
            details: Additional error details.
            suggestion: Suggested fix.
        """
        self.operation = operation
        self.timeout_ms = timeout_ms

        if message is None:
            message = f"Operation timed out after {timeout_ms}ms: {operation}"

        full_details = details or {}
        full_details["operation"] = operation
        full_details["timeout_ms"] = timeout_ms

        if suggestion is None:
            suggestion = (
                "Consider increasing the timeout or checking if the expected "
                "condition can actually be met."
            )

        super().__init__(message, full_details, suggestion)


class ScreenshotComparisonError(VisionError):
    """Raised when screenshot comparison fails."""

    def __init__(
        self,
        baseline_path: str,
        actual_path: str,
        similarity_score: float,
        threshold: float,
        message: str | None = None,
        details: dict[str, Any] | None = None,
        suggestion: str | None = None,
        diff_path: str | None = None,
        diff_regions: list[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize screenshot comparison error.

        Args:
            baseline_path: Path to baseline image.
            actual_path: Path to actual screenshot.
            similarity_score: Computed similarity score.
            threshold: Required threshold.
            message: Custom error message.
            details: Additional error details.
            suggestion: Suggested fix.
            diff_path: Path to diff image.
            diff_regions: List of regions with differences.
        """
        self.baseline_path = baseline_path
        self.actual_path = actual_path
        self.similarity_score = similarity_score
        self.threshold = threshold
        self.diff_path = diff_path
        self.diff_regions = diff_regions or []

        if message is None:
            message = (
                f"Screenshot comparison failed: similarity {similarity_score:.2%} "
                f"< threshold {threshold:.2%}"
            )

        full_details = details or {}
        full_details["baseline"] = baseline_path
        full_details["actual"] = actual_path
        full_details["similarity_score"] = f"{similarity_score:.2%}"
        full_details["threshold"] = f"{threshold:.2%}"

        if diff_path:
            full_details["diff_image"] = diff_path
        if diff_regions:
            full_details["diff_region_count"] = len(diff_regions)

        if suggestion is None:
            if similarity_score > 0.9:
                suggestion = (
                    "Minor visual differences detected. Consider updating the "
                    "baseline or adding ignore regions for dynamic content."
                )
            else:
                suggestion = (
                    "Significant visual differences detected. Review the diff image "
                    "to identify the cause of the change."
                )

        super().__init__(message, full_details, suggestion)


class ConfigurationError(VisionError):
    """Raised when there's a configuration error."""

    def __init__(
        self,
        config_key: str,
        message: str | None = None,
        details: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Initialize configuration error.

        Args:
            config_key: Configuration key with the issue.
            message: Custom error message.
            details: Additional error details.
            suggestion: Suggested fix.
        """
        self.config_key = config_key

        if message is None:
            message = f"Configuration error for key: {config_key}"

        full_details = details or {}
        full_details["config_key"] = config_key

        super().__init__(message, full_details, suggestion)


class EnvironmentNotLoadedError(VisionError):
    """Raised when environment data is required but not loaded."""

    def __init__(
        self,
        feature: str,
        message: str | None = None,
        details: dict[str, Any] | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Initialize environment not loaded error.

        Args:
            feature: Feature that requires environment data.
            message: Custom error message.
            details: Additional error details.
            suggestion: Suggested fix.
        """
        self.feature = feature

        if message is None:
            message = f"GUI environment not loaded, required for: {feature}"

        full_details = details or {}
        full_details["feature"] = feature

        if suggestion is None:
            suggestion = (
                "Load a GUI environment with VisionVerifier.load_environment() "
                "or run environment discovery first."
            )

        super().__init__(message, full_details, suggestion)


__all__ = [
    "VisionError",
    "AssertionError",
    "ElementNotFoundError",
    "TimeoutError",
    "ScreenshotComparisonError",
    "ConfigurationError",
    "EnvironmentNotLoadedError",
]
