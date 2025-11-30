"""
Coverage analysis module.

Performs issue detection, generates recommendations, and creates
migration summaries based on coverage data.
"""

from datetime import datetime

from .coverage_models import (
    MigrationStatus,
    MigrationSummary,
    TestMapping,
)
from .coverage_stats import CoverageStats


class CoverageAnalyzer:
    """Analyzes coverage data and generates insights."""

    def __init__(self, test_mappings: dict[str, TestMapping], stats: CoverageStats) -> None:
        """
        Initialize the analyzer.

        Args:
            test_mappings: Dictionary of test mappings
            stats: Statistics calculator instance
        """
        self.test_mappings = test_mappings
        self.stats = stats

    def identify_issues(self) -> dict[str, int]:
        """
        Identify common migration issues.

        Returns:
            Dictionary mapping issue types to counts
        """
        issues = {
            "failed_migrations": 0,
            "missing_python_tests": 0,
            "orphaned_python_tests": 0,
            "incomplete_method_mapping": 0,
            "long_running_migrations": 0,
        }

        for mapping in self.test_mappings.values():
            if mapping.migration_status == MigrationStatus.FAILED:
                issues["failed_migrations"] += 1

            if (
                mapping.migration_status == MigrationStatus.COMPLETED
                and mapping.python_test_path is None
            ):
                issues["missing_python_tests"] += 1

            if mapping.python_test_path and not mapping.java_test_path.exists():
                issues["orphaned_python_tests"] += 1

            if mapping.test_methods and mapping.migration_success_rate < 1.0:
                issues["incomplete_method_mapping"] += 1

            if (
                mapping.migration_status == MigrationStatus.IN_PROGRESS
                and mapping.migration_date
                and (datetime.now() - mapping.migration_date).days > 1
            ):
                issues["long_running_migrations"] += 1

        return issues

    def generate_recommendations(self) -> list[str]:
        """
        Generate recommendations based on current migration status.

        Returns:
            List of recommendation strings
        """
        recommendations = []
        progress = self.stats.calculate_progress()
        issues = self.identify_issues()

        # Progress-based recommendations
        if progress.completion_percentage < 25:
            recommendations.append("Consider prioritizing simple unit tests for initial migration")
        elif progress.completion_percentage < 75:
            recommendations.append("Focus on integration tests and complex mock scenarios")
        else:
            recommendations.append("Review failed migrations and complete remaining edge cases")

        # Issue-based recommendations
        if issues["failed_migrations"] > 5:
            recommendations.append("Investigate common failure patterns in migration process")

        if issues["incomplete_method_mapping"] > 0:
            recommendations.append("Complete method-level mapping for better traceability")

        if issues["orphaned_python_tests"] > 0:
            recommendations.append("Review orphaned Python tests for proper Java test association")

        if progress.success_rate < 80:
            recommendations.append("Consider improving migration tooling or process")

        return recommendations

    def generate_migration_summary(self, recent_migrations: list[TestMapping]) -> MigrationSummary:
        """
        Generate comprehensive migration summary.

        Args:
            recent_migrations: List of recent migrations

        Returns:
            MigrationSummary object with complete status information
        """
        return MigrationSummary(
            timestamp=datetime.now(),
            progress=self.stats.calculate_progress(),
            coverage_metrics=self.stats.calculate_coverage_metrics(),
            category_breakdown=self.stats.get_category_breakdown(),
            status_breakdown=self.stats.get_status_breakdown(),
            recent_migrations=recent_migrations,
            issues_summary=self.identify_issues(),
            recommendations=self.generate_recommendations(),
        )
