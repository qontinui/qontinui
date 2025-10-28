"""
Coverage data collection module.

Handles registration of Java and Python tests, tracking test mappings,
and maintaining migration history.
"""

from datetime import datetime
from pathlib import Path

from ..core.models import TestFile
from .coverage_models import MigrationStatus, TestCategory, TestMapping, TestType


class CoverageCollector:
    """Collects and maintains test mapping data."""

    def __init__(self, java_source_dir: Path, python_target_dir: Path) -> None:
        """
        Initialize the coverage collector.

        Args:
            java_source_dir: Directory containing Java test files
            python_target_dir: Directory containing Python test files
        """
        self.java_source_dir = java_source_dir
        self.python_target_dir = python_target_dir
        self.test_mappings: dict[str, TestMapping] = {}
        self.migration_history: list[TestMapping] = []
        self.tracking_start_time = datetime.now()

    def register_java_test(
        self, test_file: TestFile, test_category: TestCategory = TestCategory.UNIT_SIMPLE
    ) -> None:
        """
        Register a Java test file for tracking.

        Args:
            test_file: Java test file information
            test_category: Category of the test for tracking purposes
        """
        mapping_key = self._get_mapping_key(test_file.path)

        if mapping_key not in self.test_mappings:
            self.test_mappings[mapping_key] = TestMapping(
                java_test_path=test_file.path,
                python_test_path=None,
                java_class_name=test_file.class_name,
                python_class_name=None,
                test_type=test_file.test_type,
                test_category=test_category,
                migration_status=MigrationStatus.NOT_STARTED,
            )

    def register_python_test(
        self, python_test_path: Path, java_test_path: Path, python_class_name: str
    ) -> None:
        """
        Register a Python test file as migrated from Java.

        Args:
            python_test_path: Path to the Python test file
            java_test_path: Path to the original Java test file
            python_class_name: Name of the Python test class
        """
        mapping_key = self._get_mapping_key(java_test_path)

        if mapping_key in self.test_mappings:
            mapping = self.test_mappings[mapping_key]
            mapping.python_test_path = python_test_path
            mapping.python_class_name = python_class_name
            mapping.migration_status = MigrationStatus.COMPLETED
            mapping.migration_date = datetime.now()

            # Add to history
            self.migration_history.append(mapping)
        else:
            # Create new mapping for orphaned Python test
            self.test_mappings[mapping_key] = TestMapping(
                java_test_path=java_test_path,
                python_test_path=python_test_path,
                java_class_name="Unknown",
                python_class_name=python_class_name,
                test_type=TestType.UNKNOWN,
                test_category=TestCategory.UNIT_SIMPLE,
                migration_status=MigrationStatus.COMPLETED,
                migration_date=datetime.now(),
            )

    def update_migration_status(
        self, java_test_path: Path, status: MigrationStatus, notes: str = ""
    ) -> None:
        """
        Update the migration status of a test.

        Args:
            java_test_path: Path to the Java test file
            status: New migration status
            notes: Optional notes about the status change
        """
        mapping_key = self._get_mapping_key(java_test_path)

        if mapping_key in self.test_mappings:
            mapping = self.test_mappings[mapping_key]
            mapping.migration_status = status
            mapping.migration_notes = notes

            if status == MigrationStatus.COMPLETED:
                mapping.migration_date = datetime.now()

    def add_method_mapping(
        self, java_test_path: Path, java_method: str, python_method: str
    ) -> None:
        """
        Add a mapping between Java and Python test methods.

        Args:
            java_test_path: Path to the Java test file
            java_method: Name of the Java test method
            python_method: Name of the corresponding Python test method
        """
        mapping_key = self._get_mapping_key(java_test_path)

        if mapping_key in self.test_mappings:
            self.test_mappings[mapping_key].test_methods[java_method] = python_method

    def get_recent_migrations(self, limit: int = 10) -> list[TestMapping]:
        """
        Get recently migrated tests.

        Args:
            limit: Maximum number of recent migrations to return

        Returns:
            List of recently migrated test mappings
        """
        recent = [m for m in self.migration_history if m.migration_date is not None]
        recent.sort(key=lambda x: x.migration_date or datetime.min, reverse=True)
        return recent[:limit]

    def get_unmigrated_tests(self) -> list[TestMapping]:
        """Get list of tests that haven't been migrated yet."""
        return [
            mapping
            for mapping in self.test_mappings.values()
            if mapping.migration_status in [MigrationStatus.NOT_STARTED, MigrationStatus.FAILED]
        ]

    def _get_mapping_key(self, java_test_path: Path) -> str:
        """Generate a unique key for test mapping."""
        return str(java_test_path.resolve())
