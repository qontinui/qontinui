"""
Exception validation strategy.

This module implements validation logic for comparing exception messages
and error types between Java and Python test executions.
"""

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.models import TestResult
else:
    try:
        from ...core.models import TestResult
    except ImportError:
        pass


class ExceptionValidator:
    """
    Validates and compares test exceptions for equivalence.

    This validator handles exception message normalization, comparison,
    and similarity scoring for error cases.
    """

    def __init__(
        self, ignore_whitespace: bool = True, ignore_case: bool = False
    ) -> None:
        """
        Initialize the exception validator.

        Args:
            ignore_whitespace: Whether to ignore whitespace differences
            ignore_case: Whether to ignore case differences
        """
        self.ignore_whitespace = ignore_whitespace
        self.ignore_case = ignore_case

    def normalize_error_message(self, error_msg: str) -> str:
        """
        Normalize error messages for comparison.

        Removes file paths, line numbers, and other environment-specific details
        that may differ between Java and Python executions.

        Args:
            error_msg: Raw error message

        Returns:
            Normalized error message
        """
        # Remove file paths
        error_msg = re.sub(r"[A-Za-z]:[\\\/][^:\s]+", "<path>", error_msg)
        error_msg = re.sub(r"\/[^:\s]+\.py", "<file>.py", error_msg)
        error_msg = re.sub(r"\/[^:\s]+\.java", "<file>.java", error_msg)

        # Remove line numbers
        error_msg = re.sub(r"line \d+", "line <num>", error_msg)
        error_msg = re.sub(r":\d+:", ":<num>:", error_msg)

        # Apply basic normalization
        if self.ignore_whitespace:
            error_msg = " ".join(error_msg.split())

        if self.ignore_case:
            error_msg = error_msg.lower()

        return error_msg.strip()

    def compare_exceptions(
        self, java_result: "TestResult", python_result: "TestResult"
    ) -> tuple[str, list[str], float]:
        """
        Compare exception messages between Java and Python tests.

        Args:
            java_result: Result from Java test execution
            python_result: Result from Python test execution

        Returns:
            Tuple of (validation_result, differences, similarity_score)
            validation_result is one of: "equivalent", "different"
        """
        java_error = java_result.error_message or ""
        python_error = python_result.error_message or ""

        # Normalize error messages for comparison
        java_error_normalized = self.normalize_error_message(java_error)
        python_error_normalized = self.normalize_error_message(python_error)

        if java_error_normalized == python_error_normalized:
            return "equivalent", [], 1.0

        differences = ["Error messages differ"]
        similarity_score = self._calculate_error_similarity(
            java_error_normalized, python_error_normalized
        )

        return "different", differences, similarity_score

    def _calculate_error_similarity(self, error1: str, error2: str) -> float:
        """
        Calculate similarity score between error messages.

        Args:
            error1: First normalized error message
            error2: Second normalized error message

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not error1 and not error2:
            return 1.0

        if not error1 or not error2:
            return 0.0

        max_len = max(len(error1), len(error2))
        if max_len == 0:
            return 1.0

        # Count matching characters at same positions
        matches = sum(
            1 for i in range(min(len(error1), len(error2))) if error1[i] == error2[i]
        )

        return matches / max_len

    def get_normalized_errors(
        self, java_result: "TestResult", python_result: "TestResult"
    ) -> tuple[str, str]:
        """
        Get normalized error messages for both Java and Python results.

        Args:
            java_result: Result from Java test execution
            python_result: Result from Python test execution

        Returns:
            Tuple of (normalized_java_error, normalized_python_error)
        """
        java_error = java_result.error_message or ""
        python_error = python_result.error_message or ""

        return (
            self.normalize_error_message(java_error),
            self.normalize_error_message(python_error),
        )
