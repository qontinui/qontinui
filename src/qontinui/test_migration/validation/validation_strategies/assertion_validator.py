"""
Assertion and behavioral equivalence validation strategy.

This module implements validation logic for comparing test assertions
and behavioral equivalence between Java and Python test executions.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...core.models import TestResult
else:
    try:
        from ...core.models import TestResult
    except ImportError:
        pass


@dataclass
class BehavioralEquivalenceConfig:
    """Configuration for behavioral equivalence verification."""

    ignore_whitespace: bool = True
    ignore_case: bool = False
    ignore_order: bool = False
    tolerance_threshold: float = 0.95
    custom_comparators: dict[str, Any] = field(default_factory=dict)


class AssertionValidator:
    """
    Validates behavioral equivalence between Java and Python test results.

    This validator checks if both tests exhibit equivalent behavior by comparing
    pass/fail status, outputs, and applying tolerance thresholds.
    """

    def __init__(self, config: BehavioralEquivalenceConfig | None = None) -> None:
        """
        Initialize the assertion validator.

        Args:
            config: Configuration for behavioral equivalence checks
        """
        self.config = config or BehavioralEquivalenceConfig()

    def validate_behavior(
        self,
        java_result: "TestResult",
        python_result: "TestResult",
        java_output_normalized: str,
        python_output_normalized: str,
        similarity_score: float,
    ) -> tuple[str, list[str]]:
        """
        Validate behavioral equivalence between test results.

        Args:
            java_result: Result from Java test execution
            python_result: Result from Python test execution
            java_output_normalized: Normalized Java output
            python_output_normalized: Normalized Python output
            similarity_score: Calculated similarity score

        Returns:
            Tuple of (validation_result, differences_list)
            validation_result is one of: "equivalent", "different"
        """
        differences = []

        # Check if both tests have same pass/fail status
        if java_result.passed != python_result.passed:
            differences.append(
                f"Test status differs: Java={'PASS' if java_result.passed else 'FAIL'}, "
                f"Python={'PASS' if python_result.passed else 'FAIL'}"
            )
            return "different", differences

        # If both failed, validation is handled by exception validator
        # Return early with behavior check marker
        if not java_result.passed and not python_result.passed:
            return "check_exceptions", differences

        # If both passed, check output equivalence
        if java_output_normalized == python_output_normalized:
            return "equivalent", differences

        # Apply tolerance threshold
        if similarity_score >= self.config.tolerance_threshold:
            return "equivalent", differences

        differences.append(
            f"Similarity score {similarity_score:.2f} below threshold "
            f"{self.config.tolerance_threshold:.2f}"
        )
        return "different", differences

    def compare_test_status(
        self, java_result: "TestResult", python_result: "TestResult"
    ) -> tuple[bool, str | None]:
        """
        Compare the pass/fail status of Java and Python tests.

        Args:
            java_result: Result from Java test execution
            python_result: Result from Python test execution

        Returns:
            Tuple of (status_matches, difference_message)
        """
        if java_result.passed == python_result.passed:
            return True, None

        message = (
            f"Test status mismatch: Java={'PASS' if java_result.passed else 'FAIL'}, "
            f"Python={'PASS' if python_result.passed else 'FAIL'}"
        )
        return False, message

    def should_compare_exceptions(
        self, java_result: "TestResult", python_result: "TestResult"
    ) -> bool:
        """
        Determine if exception comparison is needed.

        Args:
            java_result: Result from Java test execution
            python_result: Result from Python test execution

        Returns:
            True if both tests failed and exception comparison is needed
        """
        return not java_result.passed and not python_result.passed

    def apply_tolerance_threshold(self, similarity_score: float) -> bool:
        """
        Apply tolerance threshold to similarity score.

        Args:
            similarity_score: Calculated similarity score (0.0 to 1.0)

        Returns:
            True if similarity meets or exceeds threshold
        """
        return similarity_score >= self.config.tolerance_threshold
