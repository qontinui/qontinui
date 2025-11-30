"""
Result validation and comparison system for test migration.

This module provides a facade for comparing Java and Python test outputs,
verify behavioral equivalence, and collect performance comparison metrics.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .validation_strategies import (
    AssertionValidator,
    ExceptionValidator,
    OutputValidator,
    PerformanceValidator,
    ValidationReporter,
)
from .validation_strategies.assertion_validator import BehavioralEquivalenceConfig
from .validation_strategies.performance_validator import PerformanceMetrics
from .validation_strategies.validation_reporter import ComparisonType, ValidationResult

if TYPE_CHECKING:
    from ..core.models import TestResult, TestResults
else:
    try:
        from ..core.models import TestResult, TestResults
    except ImportError:
        # For standalone testing, define minimal models
        pass

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


class ResultValidator:
    """
    Validates and compares Java and Python test outputs for equivalence.

    This facade class coordinates multiple validation strategies to implement
    behavioral equivalence verification and performance comparison metrics.
    """

    def __init__(self, equivalence_config: BehavioralEquivalenceConfig | None = None) -> None:
        """
        Initialize the result validator.

        Args:
            equivalence_config: Configuration for behavioral equivalence checks
        """
        self.equivalence_config = equivalence_config or BehavioralEquivalenceConfig()

        # Initialize validation strategies
        self.output_validator = OutputValidator(
            ignore_whitespace=self.equivalence_config.ignore_whitespace,
            ignore_case=self.equivalence_config.ignore_case,
        )
        self.exception_validator = ExceptionValidator(
            ignore_whitespace=self.equivalence_config.ignore_whitespace,
            ignore_case=self.equivalence_config.ignore_case,
        )
        self.assertion_validator = AssertionValidator(config=self.equivalence_config)
        self.performance_validator = PerformanceValidator()
        self.reporter = ValidationReporter()

    @property
    def validation_history(self) -> list[dict[str, Any]]:
        """Access validation history from reporter."""
        return self.reporter.validation_history

    def compare_test_outputs(
        self,
        java_result: "TestResult",
        python_result: "TestResult",
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
            # Calculate performance metrics
            performance_metrics = self.performance_validator.calculate_metrics(
                java_result, python_result
            )

            # Perform comparison based on type
            if comparison_type == ComparisonType.OUTPUT:
                result, differences, similarity = self._compare_outputs(java_result, python_result)
            elif comparison_type == ComparisonType.BEHAVIOR:
                result, differences, similarity = self._compare_behavior(
                    java_result, python_result, performance_metrics
                )
            elif comparison_type == ComparisonType.PERFORMANCE:
                result, differences = self._compare_performance(java_result, python_result)
                similarity = 1.0 if result == "equivalent" else 0.0
            elif comparison_type == ComparisonType.EXCEPTION:
                result, differences, similarity = self._compare_exceptions(
                    java_result, python_result
                )
            else:
                result = "inconclusive"
                differences = []
                similarity = 0.0

            # Create comparison object
            comparison = ValidationComparison(
                test_name=java_result.test_name,
                comparison_type=comparison_type,
                validation_result=ValidationResult(result),
                java_output=java_result.output,
                python_output=python_result.output,
                differences=differences,
                similarity_score=similarity,
                performance_metrics=performance_metrics,
            )

            # Record in reporter
            self.reporter.record_comparison(
                test_name=comparison.test_name,
                comparison_type=comparison.comparison_type,
                validation_result=comparison.validation_result,
                java_output=comparison.java_output,
                python_output=comparison.python_output,
                differences=comparison.differences,
                similarity_score=comparison.similarity_score,
                performance_metrics=comparison.performance_metrics,
            )

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

            # Record error in reporter
            self.reporter.record_comparison(
                test_name=error_comparison.test_name,
                comparison_type=error_comparison.comparison_type,
                validation_result=error_comparison.validation_result,
                java_output=error_comparison.java_output,
                python_output=error_comparison.python_output,
                differences=[],
                similarity_score=0.0,
                error_details=error_comparison.error_details,
            )

            return error_comparison

    def verify_behavioral_equivalence(
        self, java_results: "TestResults", python_results: "TestResults"
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
            comparison = self.compare_test_outputs(
                java_map[test_name], python_map[test_name], ComparisonType.BEHAVIOR
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
        self, java_results: "TestResults", python_results: "TestResults"
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
            metrics[test_name] = self.performance_validator.calculate_metrics(
                java_map[test_name], python_map[test_name]
            )

        return metrics

    def get_validation_summary(self) -> dict[str, Any]:
        """Get summary of all validation results."""
        return self.reporter.get_validation_summary()

    def export_validation_results(self, output_path: Path) -> None:
        """Export validation results to JSON file."""
        self.reporter.export_validation_results(output_path)

    def validate_test_results(self, execution_results: Any) -> None:
        """
        Validate test execution results.

        Args:
            execution_results: Results from test execution to validate

        Raises:
            ValueError: If validation fails
        """
        if execution_results is None:
            raise ValueError("Execution results cannot be None")

        # Handle TestResults object
        if hasattr(execution_results, "total_tests"):
            self._validate_test_results_object(execution_results)
        # Handle TestResult object
        elif hasattr(execution_results, "test_name"):
            self._validate_test_result_object(execution_results)
        # Handle dict format
        elif isinstance(execution_results, dict):
            self._validate_results_dict(execution_results)
        # Handle list of results
        elif isinstance(execution_results, list):
            for result in execution_results:
                self.validate_test_results(result)
        else:
            raise ValueError(f"Unsupported execution results type: {type(execution_results)}")

    def _compare_outputs(
        self, java_result: "TestResult", python_result: "TestResult"
    ) -> tuple[str, list[str], float]:
        """Compare raw test outputs using OutputValidator."""
        return self.output_validator.compare_outputs(java_result, python_result)

    def _compare_behavior(
        self,
        java_result: "TestResult",
        python_result: "TestResult",
        performance_metrics: PerformanceMetrics,
    ) -> tuple[str, list[str], float]:
        """Compare test behavior using AssertionValidator and OutputValidator."""
        java_output, python_output = self.output_validator.get_normalized_outputs(
            java_result, python_result
        )

        # Calculate similarity first
        similarity = self.output_validator.calculate_similarity_score(java_output, python_output)

        # Use assertion validator for behavior check
        result, differences = self.assertion_validator.validate_behavior(
            java_result, python_result, java_output, python_output, similarity
        )

        # If behavior validator says to check exceptions, delegate to exception validator
        if result == "check_exceptions":
            result, differences, similarity = self._compare_exceptions(java_result, python_result)

        return result, differences, similarity

    def _compare_performance(
        self, java_result: "TestResult", python_result: "TestResult"
    ) -> tuple[str, list[str]]:
        """Compare performance using PerformanceValidator."""
        result, differences, _ = self.performance_validator.compare_performance(
            java_result, python_result
        )
        return result, differences

    def _compare_exceptions(
        self, java_result: "TestResult", python_result: "TestResult"
    ) -> tuple[str, list[str], float]:
        """Compare exceptions using ExceptionValidator."""
        return self.exception_validator.compare_exceptions(java_result, python_result)

    def _normalize_output(self, output: str) -> str:
        """Normalize output for comparison (delegates to OutputValidator)."""
        return self.output_validator.normalize_output(output)

    def _normalize_error_message(self, error_msg: str) -> str:
        """Normalize error messages for comparison (delegates to ExceptionValidator)."""
        return self.exception_validator.normalize_error_message(error_msg)

    def _calculate_similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity score between two texts (delegates to OutputValidator)."""
        return self.output_validator.calculate_similarity_score(text1, text2)

    def _validate_test_results_object(self, results: "TestResults") -> None:
        """Validate a TestResults object."""
        if results.total_tests < 0:
            raise ValueError("Total tests count cannot be negative")
        if results.passed_tests < 0:
            raise ValueError("Passed tests count cannot be negative")
        if results.failed_tests < 0:
            raise ValueError("Failed tests count cannot be negative")
        if results.skipped_tests < 0:
            raise ValueError("Skipped tests count cannot be negative")

        calculated_total = results.passed_tests + results.failed_tests + results.skipped_tests
        if calculated_total != results.total_tests:
            raise ValueError(
                f"Test counts inconsistent: {results.passed_tests} + {results.failed_tests} "
                f"+ {results.skipped_tests} != {results.total_tests}"
            )

        if results.execution_time < 0:
            raise ValueError("Execution time cannot be negative")

        if results.individual_results:
            for result in results.individual_results:
                self._validate_test_result_object(result)

    def _validate_test_result_object(self, result: "TestResult") -> None:
        """Validate a single TestResult object."""
        if not result.test_name:
            raise ValueError("Test name is required")
        if not result.test_file:
            raise ValueError("Test file is required")
        if result.execution_time < 0:
            raise ValueError(f"Execution time cannot be negative for test {result.test_name}")

        if not result.passed and not result.error_message:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Failed test {result.test_name} has no error message")

    def _validate_results_dict(self, results: dict) -> None:
        """Validate results in dictionary format."""
        if "total_tests" in results:
            if not isinstance(results["total_tests"], int) or results["total_tests"] < 0:
                raise ValueError("Invalid total_tests value")

        if "passed_tests" in results:
            if not isinstance(results["passed_tests"], int) or results["passed_tests"] < 0:
                raise ValueError("Invalid passed_tests value")

        if "failed_tests" in results:
            if not isinstance(results["failed_tests"], int) or results["failed_tests"] < 0:
                raise ValueError("Invalid failed_tests value")

        if "execution_time" in results:
            if (
                not isinstance(results["execution_time"], int | float)
                or results["execution_time"] < 0
            ):
                raise ValueError("Invalid execution_time value")

        if "individual_results" in results and isinstance(results["individual_results"], list):
            for result in results["individual_results"]:
                if isinstance(result, dict):
                    self._validate_results_dict(result)
                else:
                    self.validate_test_results(result)
