"""
Result validation and comparison system for test migration.

This module provides functionality to compare Java and Python test outputs,
verify behavioral equivalence, and collect performance comparison metrics.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.models import TestResult, TestResults
else:
    try:
        from ..core.models import TestResult, TestResults
    except ImportError:
        # For standalone testing, define minimal models
        pass  # dataclass and field are already imported above

        @dataclass
        class TestResult:
            test_name: str
            test_file: str
            passed: bool
            execution_time: float
            error_message: str | None = None
            stack_trace: str | None = None
            output: str = ""

        @dataclass
        class TestResults:
            total_tests: int
            passed_tests: int
            failed_tests: int

        skipped_tests: int
        execution_time: float
        individual_results: list[TestResult] = field(default_factory=list)


class ValidationResult(Enum):
    """Result of validation comparison."""

    EQUIVALENT = "equivalent"
    DIFFERENT = "different"
    INCONCLUSIVE = "inconclusive"
    ERROR = "error"


class ComparisonType(Enum):
    """Type of comparison being performed."""

    OUTPUT = "output"
    BEHAVIOR = "behavior"
    PERFORMANCE = "performance"
    EXCEPTION = "exception"


@dataclass
class PerformanceMetrics:
    """Performance comparison metrics."""

    java_execution_time: float
    python_execution_time: float
    time_difference: float
    time_ratio: float
    memory_usage_java: int | None = None
    memory_usage_python: int | None = None

    @property
    def performance_delta_percent(self) -> float:
        """Calculate performance difference as percentage."""
        if self.java_execution_time == 0:
            return 0.0
        return (
            (self.python_execution_time - self.java_execution_time) / self.java_execution_time
        ) * 100


@dataclass
class ValidationComparison:
    """Result of comparing Java and Python test outputs."""

    test_name: str
    comparison_type: ComparisonType
    validation_result: ValidationResult
    java_output: str
    python_output: str
    differences: list[str] = field(default_factory=list)
    similarity_score: float = 0.0
    performance_metrics: PerformanceMetrics | None = None
    error_details: str | None = None

    @property
    def is_equivalent(self) -> bool:
        """Check if outputs are considered equivalent."""
        return self.validation_result == ValidationResult.EQUIVALENT


@dataclass
class BehavioralEquivalenceCheck:
    """Configuration for behavioral equivalence verification."""

    ignore_whitespace: bool = True
    ignore_case: bool = False
    ignore_order: bool = False
    tolerance_threshold: float = 0.95
    custom_comparators: dict[str, Callable[..., Any]] = field(default_factory=dict)


class ResultValidator:
    """
    Validates and compares Java and Python test outputs for equivalence.

    This class implements behavioral equivalence verification logic and
    performance comparison metrics collection as specified in requirements 5.1, 5.2, 5.3.
    """

    def __init__(self, equivalence_config: BehavioralEquivalenceCheck | None = None):
        """
        Initialize the result validator.

        Args:
            equivalence_config: Configuration for behavioral equivalence checks
        """
        self.equivalence_config = equivalence_config or BehavioralEquivalenceCheck()
        self.validation_history: list[ValidationComparison] = []

    def compare_test_outputs(
        self,
        java_result: TestResult,
        python_result: TestResult,
        comparison_type: ComparisonType = ComparisonType.OUTPUT,
    ) -> ValidationComparison:
        """
        Compare Java and Python test outputs for equivalence.

        Args:
            java_result: Result from Java test execution
            python_result: Result from Python test execution
            comparison_type: Type of comparison to perform

        Returns:
            ValidationComparison with detailed comparison results
        """
        try:
            # Create base comparison object
            comparison = ValidationComparison(
                test_name=java_result.test_name,
                comparison_type=comparison_type,
                validation_result=ValidationResult.INCONCLUSIVE,
                java_output=java_result.output,
                python_output=python_result.output,
            )

            # Add performance metrics
            comparison.performance_metrics = self._calculate_performance_metrics(
                java_result, python_result
            )

            # Perform comparison based on type
            if comparison_type == ComparisonType.OUTPUT:
                self._compare_outputs(comparison, java_result, python_result)
            elif comparison_type == ComparisonType.BEHAVIOR:
                self._compare_behavior(comparison, java_result, python_result)
            elif comparison_type == ComparisonType.PERFORMANCE:
                self._compare_performance(comparison, java_result, python_result)
            elif comparison_type == ComparisonType.EXCEPTION:
                self._compare_exceptions(comparison, java_result, python_result)

            # Store in history
            self.validation_history.append(comparison)

            return comparison

        except Exception as e:
            error_comparison = ValidationComparison(
                test_name=java_result.test_name,
                comparison_type=comparison_type,
                validation_result=ValidationResult.ERROR,
                java_output=java_result.output,
                python_output=python_result.output,
                error_details=str(e),
            )
            self.validation_history.append(error_comparison)
            return error_comparison

    def verify_behavioral_equivalence(
        self, java_results: TestResults, python_results: TestResults
    ) -> list[ValidationComparison]:
        """
        Verify behavioral equivalence between Java and Python test suites.

        Args:
            java_results: Results from Java test suite execution
            python_results: Results from Python test suite execution

        Returns:
            List of ValidationComparison objects for each test pair
        """
        comparisons = []

        # Create mapping of test names to results
        java_map = {result.test_name: result for result in java_results.individual_results}
        python_map = {result.test_name: result for result in python_results.individual_results}

        # Compare each test that exists in both suites
        common_tests = set(java_map.keys()) & set(python_map.keys())

        for test_name in common_tests:
            java_result = java_map[test_name]
            python_result = python_map[test_name]

            # Perform behavioral comparison
            comparison = self.compare_test_outputs(
                java_result, python_result, ComparisonType.BEHAVIOR
            )
            comparisons.append(comparison)

        # Flag tests that exist in only one suite
        java_only = set(java_map.keys()) - set(python_map.keys())
        python_only = set(python_map.keys()) - set(java_map.keys())

        for test_name in java_only:
            comparison = ValidationComparison(
                test_name=test_name,
                comparison_type=ComparisonType.BEHAVIOR,
                validation_result=ValidationResult.DIFFERENT,
                java_output=java_map[test_name].output,
                python_output="",
                differences=["Test exists only in Java suite"],
            )
            comparisons.append(comparison)

        for test_name in python_only:
            comparison = ValidationComparison(
                test_name=test_name,
                comparison_type=ComparisonType.BEHAVIOR,
                validation_result=ValidationResult.DIFFERENT,
                java_output="",
                python_output=python_map[test_name].output,
                differences=["Test exists only in Python suite"],
            )
            comparisons.append(comparison)

        return comparisons

    def collect_performance_metrics(
        self, java_results: TestResults, python_results: TestResults
    ) -> dict[str, PerformanceMetrics]:
        """
        Collect performance comparison metrics between Java and Python tests.

        Args:
            java_results: Results from Java test suite execution
            python_results: Results from Python test suite execution

        Returns:
            Dictionary mapping test names to performance metrics
        """
        metrics = {}

        # Create mapping of test names to results
        java_map = {result.test_name: result for result in java_results.individual_results}
        python_map = {result.test_name: result for result in python_results.individual_results}

        # Calculate metrics for common tests
        common_tests = set(java_map.keys()) & set(python_map.keys())

        for test_name in common_tests:
            java_result = java_map[test_name]
            python_result = python_map[test_name]

            metrics[test_name] = self._calculate_performance_metrics(java_result, python_result)

        return metrics

    def _compare_outputs(
        self, comparison: ValidationComparison, java_result: TestResult, python_result: TestResult
    ) -> None:
        """Compare raw test outputs."""
        java_output = self._normalize_output(java_result.output)
        python_output = self._normalize_output(python_result.output)

        if java_output == python_output:
            comparison.validation_result = ValidationResult.EQUIVALENT
            comparison.similarity_score = 1.0
        else:
            comparison.validation_result = ValidationResult.DIFFERENT
            comparison.differences = self._find_output_differences(java_output, python_output)
            comparison.similarity_score = self._calculate_similarity_score(
                java_output, python_output
            )

    def _compare_behavior(
        self, comparison: ValidationComparison, java_result: TestResult, python_result: TestResult
    ) -> None:
        """Compare test behavior for equivalence."""
        # Check if both tests have same pass/fail status
        if java_result.passed != python_result.passed:
            comparison.validation_result = ValidationResult.DIFFERENT
            comparison.differences.append(
                f"Test status differs: Java={'PASS' if java_result.passed else 'FAIL'}, "
                f"Python={'PASS' if python_result.passed else 'FAIL'}"
            )
            return

        # If both failed, compare error messages
        if not java_result.passed and not python_result.passed:
            self._compare_exceptions(comparison, java_result, python_result)
            return

        # If both passed, compare outputs
        self._compare_outputs(comparison, java_result, python_result)

        # Apply tolerance threshold
        if comparison.similarity_score >= self.equivalence_config.tolerance_threshold:
            comparison.validation_result = ValidationResult.EQUIVALENT

    def _compare_performance(
        self, comparison: ValidationComparison, java_result: TestResult, python_result: TestResult
    ) -> None:
        """Compare performance characteristics."""
        metrics = comparison.performance_metrics
        if not metrics:
            comparison.validation_result = ValidationResult.ERROR
            comparison.error_details = "Performance metrics not available"
            return

        # Consider performance equivalent if within 50% difference
        if abs(metrics.performance_delta_percent) <= 50.0:
            comparison.validation_result = ValidationResult.EQUIVALENT
        else:
            comparison.validation_result = ValidationResult.DIFFERENT
            comparison.differences.append(
                f"Performance difference: {metrics.performance_delta_percent:.1f}%"
            )

    def _compare_exceptions(
        self, comparison: ValidationComparison, java_result: TestResult, python_result: TestResult
    ) -> None:
        """Compare exception messages and types."""
        java_error = java_result.error_message or ""
        python_error = python_result.error_message or ""

        # Normalize error messages for comparison
        java_error_normalized = self._normalize_error_message(java_error)
        python_error_normalized = self._normalize_error_message(python_error)

        if java_error_normalized == python_error_normalized:
            comparison.validation_result = ValidationResult.EQUIVALENT
            comparison.similarity_score = 1.0
        else:
            comparison.validation_result = ValidationResult.DIFFERENT
            comparison.differences.append("Error messages differ")
            comparison.similarity_score = self._calculate_similarity_score(
                java_error_normalized, python_error_normalized
            )

    def _calculate_performance_metrics(
        self, java_result: TestResult, python_result: TestResult
    ) -> PerformanceMetrics:
        """Calculate performance comparison metrics."""
        time_diff = python_result.execution_time - java_result.execution_time
        time_ratio = (
            python_result.execution_time / java_result.execution_time
            if java_result.execution_time > 0
            else 0.0
        )

        return PerformanceMetrics(
            java_execution_time=java_result.execution_time,
            python_execution_time=python_result.execution_time,
            time_difference=time_diff,
            time_ratio=time_ratio,
        )

    def _normalize_output(self, output: str) -> str:
        """Normalize output for comparison."""
        if self.equivalence_config.ignore_whitespace:
            output = " ".join(output.split())

        if self.equivalence_config.ignore_case:
            output = output.lower()

        return output.strip()

    def _normalize_error_message(self, error_msg: str) -> str:
        """Normalize error messages for comparison."""
        # Remove file paths and line numbers that may differ
        import re

        # Remove file paths
        error_msg = re.sub(r"[A-Za-z]:[\\\/][^:\s]+", "<path>", error_msg)
        error_msg = re.sub(r"\/[^:\s]+\.py", "<file>.py", error_msg)
        error_msg = re.sub(r"\/[^:\s]+\.java", "<file>.java", error_msg)

        # Remove line numbers
        error_msg = re.sub(r"line \d+", "line <num>", error_msg)
        error_msg = re.sub(r":\d+:", ":<num>:", error_msg)

        return self._normalize_output(error_msg)

    def _find_output_differences(self, output1: str, output2: str) -> list[str]:
        """Find specific differences between outputs."""
        differences = []

        lines1 = output1.split("\n")
        lines2 = output2.split("\n")

        if len(lines1) != len(lines2):
            differences.append(f"Line count differs: {len(lines1)} vs {len(lines2)}")

        max_lines = max(len(lines1), len(lines2))
        for i in range(max_lines):
            line1 = lines1[i] if i < len(lines1) else ""
            line2 = lines2[i] if i < len(lines2) else ""

            if line1 != line2:
                differences.append(f"Line {i+1}: '{line1}' vs '{line2}'")

        return differences[:10]  # Limit to first 10 differences

    def _calculate_similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity score between two texts."""
        if not text1 and not text2:
            return 1.0

        if not text1 or not text2:
            return 0.0

        # Simple character-based similarity
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0

        # Count matching characters at same positions
        matches = sum(1 for i in range(min(len(text1), len(text2))) if text1[i] == text2[i])

        return matches / max_len

    def get_validation_summary(self) -> dict[str, Any]:
        """Get summary of all validation results."""
        if not self.validation_history:
            return {"total_comparisons": 0}

        total = len(self.validation_history)
        equivalent = sum(1 for c in self.validation_history if c.is_equivalent)
        different = sum(
            1 for c in self.validation_history if c.validation_result == ValidationResult.DIFFERENT
        )
        errors = sum(
            1 for c in self.validation_history if c.validation_result == ValidationResult.ERROR
        )

        avg_similarity = sum(c.similarity_score for c in self.validation_history) / total

        return {
            "total_comparisons": total,
            "equivalent": equivalent,
            "different": different,
            "errors": errors,
            "equivalent_percentage": (equivalent / total) * 100,
            "average_similarity_score": avg_similarity,
            "comparison_types": {
                comp_type.value: sum(
                    1 for c in self.validation_history if c.comparison_type == comp_type
                )
                for comp_type in ComparisonType
            },
        }

    def export_validation_results(self, output_path: Path) -> None:
        """Export validation results to JSON file."""
        results_data = {
            "summary": self.get_validation_summary(),
            "comparisons": [
                {
                    "test_name": c.test_name,
                    "comparison_type": c.comparison_type.value,
                    "validation_result": c.validation_result.value,
                    "similarity_score": c.similarity_score,
                    "differences": c.differences,
                    "performance_metrics": (
                        {
                            "java_time": c.performance_metrics.java_execution_time,
                            "python_time": c.performance_metrics.python_execution_time,
                            "time_difference": c.performance_metrics.time_difference,
                            "performance_delta_percent": c.performance_metrics.performance_delta_percent,
                        }
                        if c.performance_metrics
                        else None
                    ),
                    "error_details": c.error_details,
                }
                for c in self.validation_history
            ],
        }

        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2)

    def validate_test_results(self, execution_results: Any) -> None:
        """
        Validate test execution results.

        Args:
            execution_results: Results from test execution to validate
        """
        # TODO: Implement validation logic
        pass
