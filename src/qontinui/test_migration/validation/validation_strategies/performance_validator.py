"""
Performance validation strategy.

This module implements validation logic for comparing performance metrics
and execution times between Java and Python test executions.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...core.models import TestResult
else:
    try:
        from ...core.models import TestResult
    except ImportError:
        pass


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
            (self.python_execution_time - self.java_execution_time)
            / self.java_execution_time
        ) * 100


class PerformanceValidator:
    """
    Validates and compares performance metrics between test executions.

    This validator handles performance metric calculation, comparison,
    and threshold-based validation.
    """

    def __init__(self, performance_tolerance_percent: float = 50.0) -> None:
        """
        Initialize the performance validator.

        Args:
            performance_tolerance_percent: Maximum acceptable performance difference
                as percentage (default: 50%)
        """
        self.performance_tolerance_percent = performance_tolerance_percent

    def calculate_metrics(
        self, java_result: "TestResult", python_result: "TestResult"
    ) -> PerformanceMetrics:
        """
        Calculate performance comparison metrics.

        Args:
            java_result: Result from Java test execution
            python_result: Result from Python test execution

        Returns:
            PerformanceMetrics object with calculated values
        """
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

    def compare_performance(
        self, java_result: "TestResult", python_result: "TestResult"
    ) -> tuple[str, list[str], PerformanceMetrics]:
        """
        Compare performance characteristics between test results.

        Args:
            java_result: Result from Java test execution
            python_result: Result from Python test execution

        Returns:
            Tuple of (validation_result, differences, metrics)
            validation_result is one of: "equivalent", "different", "error"
        """
        metrics = self.calculate_metrics(java_result, python_result)
        differences: list[Any] = []

        # Consider performance equivalent if within tolerance
        if abs(metrics.performance_delta_percent) <= self.performance_tolerance_percent:
            return "equivalent", differences, metrics

        differences.append(
            f"Performance difference: {metrics.performance_delta_percent:.1f}%"
        )
        return "different", differences, metrics

    def is_within_tolerance(self, metrics: PerformanceMetrics) -> bool:
        """
        Check if performance difference is within acceptable tolerance.

        Args:
            metrics: Performance metrics to check

        Returns:
            True if performance difference is within tolerance
        """
        return (
            abs(metrics.performance_delta_percent) <= self.performance_tolerance_percent
        )

    def get_performance_summary(self, metrics: PerformanceMetrics) -> dict[str, float]:
        """
        Get a summary of performance metrics as a dictionary.

        Args:
            metrics: Performance metrics to summarize

        Returns:
            Dictionary with performance metric values
        """
        return {
            "java_time": metrics.java_execution_time,
            "python_time": metrics.python_execution_time,
            "time_difference": metrics.time_difference,
            "time_ratio": metrics.time_ratio,
            "performance_delta_percent": metrics.performance_delta_percent,
        }
