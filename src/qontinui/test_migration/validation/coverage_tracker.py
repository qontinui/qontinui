"""
Coverage and progress tracking system for test migration.

This module provides a facade for monitoring migration progress,
tracking test mapping between Java and Python tests, and generating
migration status reports and summaries.
"""

from pathlib import Path
from typing import Any

from ..core.models import TestFile
from .coverage_analyzer import CoverageAnalyzer
from .coverage_collector import CoverageCollector
from .coverage_comparator import CoverageComparator
from .coverage_models import (
    CoverageMetrics,
    MigrationProgress,
    MigrationStatus,
    MigrationSummary,
    TestCategory,
    TestMapping,
)
from .coverage_reporter import CoverageReporter
from .coverage_stats import CoverageStats


class CoverageTracker:
    """
    Tracks migration progress and test coverage between Java and Python tests.

    This facade coordinates coverage collection, analysis, reporting, and comparison
    to monitor migration progress and generate status reports as specified in
    requirements 7.2, 7.5.
    """

    def __init__(self, java_source_dir: Path, python_target_dir: Path) -> None:
        """
        Initialize the coverage tracker.

        Args:
            java_source_dir: Directory containing Java test files
            python_target_dir: Directory containing Python test files
        """
        self.collector = CoverageCollector(java_source_dir, python_target_dir)
        self.stats = CoverageStats(self.collector.test_mappings, self.collector.tracking_start_time)
        self.analyzer = CoverageAnalyzer(self.collector.test_mappings, self.stats)
        self.reporter = CoverageReporter(
            java_source_dir,
            python_target_dir,
            self.collector.test_mappings,
            self.collector.tracking_start_time,
        )
        self.comparator = CoverageComparator(self.collector.test_mappings)

    def register_java_test(
        self, test_file: TestFile, test_category: TestCategory = TestCategory.UNIT_SIMPLE
    ) -> None:
        """Register a Java test file for tracking."""
        self.collector.register_java_test(test_file, test_category)

    def register_python_test(
        self, python_test_path: Path, java_test_path: Path, python_class_name: str
    ) -> None:
        """Register a Python test file as migrated from Java."""
        self.collector.register_python_test(python_test_path, java_test_path, python_class_name)

    def update_migration_status(
        self, java_test_path: Path, status: MigrationStatus, notes: str = ""
    ) -> None:
        """Update the migration status of a test."""
        self.collector.update_migration_status(java_test_path, status, notes)

    def add_method_mapping(
        self, java_test_path: Path, java_method: str, python_method: str
    ) -> None:
        """Add a mapping between Java and Python test methods."""
        self.collector.add_method_mapping(java_test_path, java_method, python_method)

    def calculate_progress(self) -> MigrationProgress:
        """Calculate current migration progress."""
        return self.stats.calculate_progress()

    def calculate_coverage_metrics(self) -> CoverageMetrics:
        """Calculate test coverage metrics."""
        return self.stats.calculate_coverage_metrics()

    def get_migration_statistics(self) -> dict[str, Any]:
        """Get detailed migration statistics."""
        return self.stats.get_migration_statistics()

    def generate_migration_summary(self) -> MigrationSummary:
        """Generate comprehensive migration summary."""
        recent_migrations = self.collector.get_recent_migrations()
        return self.analyzer.generate_migration_summary(recent_migrations)

    def export_mapping_documentation(self, output_path: Path) -> None:
        """Export test mapping documentation to JSON file."""
        self.reporter.export_mapping_documentation(output_path)

    def export_progress_report(self, output_path: Path) -> None:
        """Export migration progress report to JSON file."""
        summary = self.generate_migration_summary()
        self.reporter.export_progress_report(output_path, summary)

    def load_mapping_documentation(self, input_path: Path) -> None:
        """Load test mapping documentation from JSON file."""
        tracking_time_holder = [self.collector.tracking_start_time]
        migration_history = self.comparator.load_mapping_documentation(  # type: ignore[func-returns-value]
            input_path, tracking_time_holder
        )
        self.collector.tracking_start_time = tracking_time_holder[0]
        self.collector.migration_history = migration_history

    def update_coverage(self, test_name: str, coverage_data: Any) -> None:
        """Update coverage information for a test."""
        self.comparator.update_coverage(test_name, coverage_data)

    def get_unmigrated_tests(self) -> list[TestMapping]:
        """Get list of tests that haven't been migrated yet."""
        return self.collector.get_unmigrated_tests()
