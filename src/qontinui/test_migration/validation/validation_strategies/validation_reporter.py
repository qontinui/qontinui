"""
Validation reporting and export strategy.

This module implements reporting and export functionality for validation results,
including summaries, JSON export, and validation result aggregation.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any

from .performance_validator import PerformanceMetrics


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


class ValidationReporter:
    """
    Generates reports and exports validation results.

    This reporter handles validation history tracking, summary generation,
    and JSON export of validation results.
    """

    def __init__(self) -> None:
        """Initialize the validation reporter."""
        self.validation_history: list[dict[str, Any]] = []

    def record_comparison(
        self,
        test_name: str,
        comparison_type: ComparisonType,
        validation_result: ValidationResult,
        java_output: str,
        python_output: str,
        differences: list[str],
        similarity_score: float,
        performance_metrics: PerformanceMetrics | None = None,
        error_details: str | None = None,
    ) -> None:
        """
        Record a validation comparison in history.

        Args:
            test_name: Name of the test being compared
            comparison_type: Type of comparison performed
            validation_result: Result of the validation
            java_output: Java test output
            python_output: Python test output
            differences: List of identified differences
            similarity_score: Calculated similarity score
            performance_metrics: Optional performance metrics
            error_details: Optional error details
        """
        comparison_record = {
            "test_name": test_name,
            "comparison_type": comparison_type,
            "validation_result": validation_result,
            "java_output": java_output,
            "python_output": python_output,
            "differences": differences,
            "similarity_score": similarity_score,
            "performance_metrics": performance_metrics,
            "error_details": error_details,
        }
        self.validation_history.append(comparison_record)

    def get_validation_summary(self) -> dict[str, Any]:
        """
        Get summary of all validation results.

        Returns:
            Dictionary containing validation summary statistics
        """
        if not self.validation_history:
            return {"total_comparisons": 0}

        total = len(self.validation_history)
        equivalent = sum(
            1
            for c in self.validation_history
            if c["validation_result"] == ValidationResult.EQUIVALENT
        )
        different = sum(
            1
            for c in self.validation_history
            if c["validation_result"] == ValidationResult.DIFFERENT
        )
        errors = sum(
            1 for c in self.validation_history if c["validation_result"] == ValidationResult.ERROR
        )

        avg_similarity = sum(c["similarity_score"] for c in self.validation_history) / total

        return {
            "total_comparisons": total,
            "equivalent": equivalent,
            "different": different,
            "errors": errors,
            "equivalent_percentage": (equivalent / total) * 100,
            "average_similarity_score": avg_similarity,
            "comparison_types": {
                comp_type.value: sum(
                    1 for c in self.validation_history if c["comparison_type"] == comp_type
                )
                for comp_type in ComparisonType
            },
        }

    def export_validation_results(self, output_path: Path) -> None:
        """
        Export validation results to JSON file.

        Args:
            output_path: Path where JSON file should be written
        """
        results_data = {
            "summary": self.get_validation_summary(),
            "comparisons": [
                {
                    "test_name": c["test_name"],
                    "comparison_type": c["comparison_type"].value,
                    "validation_result": c["validation_result"].value,
                    "similarity_score": c["similarity_score"],
                    "differences": c["differences"],
                    "performance_metrics": (
                        {
                            "java_time": c["performance_metrics"].java_execution_time,
                            "python_time": c["performance_metrics"].python_execution_time,
                            "time_difference": c["performance_metrics"].time_difference,
                            "performance_delta_percent": c[
                                "performance_metrics"
                            ].performance_delta_percent,
                        }
                        if c["performance_metrics"]
                        else None
                    ),
                    "error_details": c["error_details"],
                }
                for c in self.validation_history
            ],
        }

        with open(output_path, "w") as f:
            json.dump(results_data, f, indent=2)

    def clear_history(self) -> None:
        """Clear validation history."""
        self.validation_history.clear()

    def get_history_count(self) -> int:
        """
        Get the number of validation comparisons in history.

        Returns:
            Count of validation comparisons
        """
        return len(self.validation_history)

    def get_comparison_types_summary(self) -> dict[str, int]:
        """
        Get summary of comparison types performed.

        Returns:
            Dictionary mapping comparison type names to counts
        """
        return {
            comp_type.value: sum(
                1 for c in self.validation_history if c["comparison_type"] == comp_type
            )
            for comp_type in ComparisonType
        }
