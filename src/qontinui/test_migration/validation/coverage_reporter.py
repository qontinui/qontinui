"""
Coverage reporting module.

Handles exporting migration summaries, progress reports, and test mapping
documentation to JSON files.
"""

import json
from datetime import datetime
from pathlib import Path

from .coverage_models import MigrationSummary, TestMapping


class CoverageReporter:
    """Generates and exports coverage reports."""

    def __init__(
        self,
        java_source_dir: Path,
        python_target_dir: Path,
        test_mappings: dict[str, TestMapping],
        tracking_start_time: datetime,
    ) -> None:
        """
        Initialize the reporter.

        Args:
            java_source_dir: Directory containing Java test files
            python_target_dir: Directory containing Python test files
            test_mappings: Dictionary of test mappings
            tracking_start_time: When tracking started
        """
        self.java_source_dir = java_source_dir
        self.python_target_dir = python_target_dir
        self.test_mappings = test_mappings
        self.tracking_start_time = tracking_start_time

    def export_mapping_documentation(self, output_path: Path) -> None:
        """
        Export test mapping documentation to JSON file.

        Args:
            output_path: Path where to save the mapping documentation
        """
        mapping_data = {
            "metadata": {
                "java_source_dir": str(self.java_source_dir),
                "python_target_dir": str(self.python_target_dir),
                "tracking_start_time": self.tracking_start_time.isoformat(),
                "export_time": datetime.now().isoformat(),
                "total_mappings": len(self.test_mappings),
            },
            "mappings": [
                {
                    "java_test_path": str(mapping.java_test_path),
                    "python_test_path": (
                        str(mapping.python_test_path) if mapping.python_test_path else None
                    ),
                    "java_class_name": mapping.java_class_name,
                    "python_class_name": mapping.python_class_name,
                    "test_type": mapping.test_type.value,
                    "test_category": mapping.test_category.value,
                    "migration_status": mapping.migration_status.value,
                    "migration_date": (
                        mapping.migration_date.isoformat() if mapping.migration_date else None
                    ),
                    "migration_notes": mapping.migration_notes,
                    "test_methods": mapping.test_methods,
                    "migration_success_rate": mapping.migration_success_rate,
                }
                for mapping in self.test_mappings.values()
            ],
        }

        with open(output_path, "w") as f:
            json.dump(mapping_data, f, indent=2)

    def export_progress_report(self, output_path: Path, summary: MigrationSummary) -> None:
        """
        Export migration progress report to JSON file.

        Args:
            output_path: Path where to save the progress report
            summary: Migration summary to export
        """
        report_data = {
            "summary": {
                "timestamp": summary.timestamp.isoformat(),
                "completion_percentage": summary.progress.completion_percentage,
                "success_rate": summary.progress.success_rate,
                "mapping_coverage": summary.coverage_metrics.mapping_coverage,
                "method_coverage": summary.coverage_metrics.test_method_coverage,
            },
            "progress": {
                "total_java_tests": summary.progress.total_java_tests,
                "migrated_tests": summary.progress.migrated_tests,
                "failed_migrations": summary.progress.failed_migrations,
                "skipped_tests": summary.progress.skipped_tests,
                "in_progress_tests": summary.progress.in_progress_tests,
            },
            "coverage_metrics": {
                "java_test_count": summary.coverage_metrics.java_test_count,
                "python_test_count": summary.coverage_metrics.python_test_count,
                "mapped_tests": summary.coverage_metrics.mapped_tests,
                "unmapped_java_tests": summary.coverage_metrics.unmapped_java_tests,
                "orphaned_python_tests": summary.coverage_metrics.orphaned_python_tests,
            },
            "category_breakdown": {
                category.value: count for category, count in summary.category_breakdown.items()
            },
            "status_breakdown": {
                status.value: count for status, count in summary.status_breakdown.items()
            },
            "recent_migrations": [
                {
                    "java_class": mapping.java_class_name,
                    "python_class": mapping.python_class_name,
                    "migration_date": (
                        mapping.migration_date.isoformat() if mapping.migration_date else None
                    ),
                    "test_category": mapping.test_category.value,
                }
                for mapping in summary.recent_migrations
            ],
            "issues": summary.issues_summary,
            "recommendations": summary.recommendations,
        }

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)
