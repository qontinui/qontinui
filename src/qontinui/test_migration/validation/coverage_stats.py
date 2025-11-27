"""
Coverage statistics computation module.

Calculates migration progress metrics, coverage statistics,
and breakdowns by category and status.
"""

from datetime import datetime
from typing import Any

from .coverage_models import (
    CoverageMetrics,
    MigrationProgress,
    MigrationStatus,
    TestCategory,
    TestMapping,
)


class CoverageStats:
    """Computes coverage statistics and metrics."""

    def __init__(
        self, test_mappings: dict[str, TestMapping], tracking_start_time: datetime
    ) -> None:
        """
        Initialize the stats calculator.

        Args:
            test_mappings: Dictionary of test mappings
            tracking_start_time: When tracking started
        """
        self.test_mappings = test_mappings
        self.tracking_start_time = tracking_start_time

    def calculate_progress(self) -> MigrationProgress:
        """
        Calculate current migration progress.

        Returns:
            MigrationProgress object with current statistics
        """
        total_tests = len(self.test_mappings)
        migrated = sum(
            1
            for m in self.test_mappings.values()
            if m.migration_status == MigrationStatus.COMPLETED
        )
        failed = sum(
            1 for m in self.test_mappings.values() if m.migration_status == MigrationStatus.FAILED
        )
        skipped = sum(
            1 for m in self.test_mappings.values() if m.migration_status == MigrationStatus.SKIPPED
        )
        in_progress = sum(
            1
            for m in self.test_mappings.values()
            if m.migration_status == MigrationStatus.IN_PROGRESS
        )

        return MigrationProgress(
            total_java_tests=total_tests,
            migrated_tests=migrated,
            failed_migrations=failed,
            skipped_tests=skipped,
            in_progress_tests=in_progress,
        )

    def calculate_coverage_metrics(self) -> CoverageMetrics:
        """
        Calculate test coverage metrics.

        Returns:
            CoverageMetrics object with coverage statistics
        """
        java_tests = len(self.test_mappings)
        python_tests = sum(1 for m in self.test_mappings.values() if m.python_test_path is not None)
        mapped_tests = sum(1 for m in self.test_mappings.values() if m.is_migrated)
        unmapped_java = java_tests - mapped_tests

        # Calculate orphaned Python tests (Python tests without Java counterpart)
        orphaned_python = python_tests - mapped_tests

        # Calculate method coverage
        total_methods = sum(len(m.test_methods) for m in self.test_mappings.values())
        mapped_methods = sum(
            len([method for method in m.test_methods.values() if method])
            for m in self.test_mappings.values()
        )
        method_coverage = (mapped_methods / total_methods * 100) if total_methods > 0 else 0.0

        return CoverageMetrics(
            java_test_count=java_tests,
            python_test_count=python_tests,
            mapped_tests=mapped_tests,
            unmapped_java_tests=unmapped_java,
            orphaned_python_tests=orphaned_python,
            test_method_coverage=method_coverage,
        )

    def get_category_breakdown(self) -> dict[TestCategory, int]:
        """Get breakdown of tests by category."""
        breakdown = dict.fromkeys(TestCategory, 0)

        for mapping in self.test_mappings.values():
            breakdown[mapping.test_category] += 1

        return breakdown

    def get_status_breakdown(self) -> dict[MigrationStatus, int]:
        """Get breakdown of tests by migration status."""
        breakdown = dict.fromkeys(MigrationStatus, 0)

        for mapping in self.test_mappings.values():
            breakdown[mapping.migration_status] += 1

        return breakdown

    def get_migration_statistics(self) -> dict[str, Any]:
        """Get detailed migration statistics."""
        progress = self.calculate_progress()
        coverage = self.calculate_coverage_metrics()

        return {
            "completion_rate": progress.completion_percentage,
            "success_rate": progress.success_rate,
            "mapping_coverage": coverage.mapping_coverage,
            "method_coverage": coverage.test_method_coverage,
            "total_tests": progress.total_java_tests,
            "migrated_count": progress.migrated_tests,
            "failed_count": progress.failed_migrations,
            "average_methods_per_test": (
                sum(len(m.test_methods) for m in self.test_mappings.values())
                / len(self.test_mappings)
                if self.test_mappings
                else 0
            ),
            "migration_velocity": self._calculate_recent_migration_count(7),
            "time_since_start": (datetime.now() - self.tracking_start_time).days,
        }

    def _calculate_recent_migration_count(self, days: int) -> int:
        """Calculate number of migrations in last N days."""
        cutoff = datetime.now().timestamp() - (days * 86400)
        return sum(
            1
            for m in self.test_mappings.values()
            if m.migration_date and m.migration_date.timestamp() >= cutoff
        )
