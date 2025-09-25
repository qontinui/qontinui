"""
Unit tests for the CoverageTracker system.

Tests the accuracy of coverage tracking and progress monitoring functionality
as required by task 9.2.
"""

import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from ..core.models import TestFile, TestType
from ..validation.coverage_tracker import (
    CoverageMetrics,
    CoverageTracker,
    MigrationProgress,
    MigrationStatus,
    MigrationSummary,
    TestCategory,
    TestMapping,
)


class TestCoverageTracker:
    """Test suite for CoverageTracker functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.java_dir = Path("/test/java")
        self.python_dir = Path("/test/python")
        self.tracker = CoverageTracker(self.java_dir, self.python_dir)

        # Sample test files
        self.java_test_file = TestFile(
            path=Path("/test/java/CalculatorTest.java"),
            test_type=TestType.UNIT,
            class_name="CalculatorTest",
            package="com.example.calculator",
        )

        self.integration_test_file = TestFile(
            path=Path("/test/java/IntegrationTest.java"),
            test_type=TestType.INTEGRATION,
            class_name="IntegrationTest",
            package="com.example.integration",
        )

    def test_register_java_test(self):
        """Test registering Java test files for tracking."""
        self.tracker.register_java_test(self.java_test_file, TestCategory.UNIT_SIMPLE)

        mapping_key = str(self.java_test_file.path.resolve())
        assert mapping_key in self.tracker.test_mappings

        mapping = self.tracker.test_mappings[mapping_key]
        assert mapping.java_test_path == self.java_test_file.path
        assert mapping.java_class_name == "CalculatorTest"
        assert mapping.test_type == TestType.UNIT
        assert mapping.test_category == TestCategory.UNIT_SIMPLE
        assert mapping.migration_status == MigrationStatus.NOT_STARTED
        assert mapping.python_test_path is None

    def test_register_python_test(self):
        """Test registering Python test files as migrated."""
        # First register the Java test
        self.tracker.register_java_test(self.java_test_file, TestCategory.UNIT_SIMPLE)

        # Then register the Python test
        python_path = Path("/test/python/test_calculator.py")
        self.tracker.register_python_test(python_path, self.java_test_file.path, "TestCalculator")

        mapping_key = str(self.java_test_file.path.resolve())
        mapping = self.tracker.test_mappings[mapping_key]

        assert mapping.python_test_path == python_path
        assert mapping.python_class_name == "TestCalculator"
        assert mapping.migration_status == MigrationStatus.COMPLETED
        assert mapping.migration_date is not None
        assert mapping in self.tracker.migration_history

    def test_register_orphaned_python_test(self):
        """Test registering Python test without existing Java mapping."""
        python_path = Path("/test/python/test_orphan.py")
        java_path = Path("/test/java/OrphanTest.java")

        self.tracker.register_python_test(python_path, java_path, "TestOrphan")

        mapping_key = str(java_path.resolve())
        assert mapping_key in self.tracker.test_mappings

        mapping = self.tracker.test_mappings[mapping_key]
        assert mapping.python_test_path == python_path
        assert mapping.java_class_name == "Unknown"
        assert mapping.test_type == TestType.UNKNOWN

    def test_update_migration_status(self):
        """Test updating migration status of tests."""
        self.tracker.register_java_test(self.java_test_file, TestCategory.UNIT_SIMPLE)

        # Update to in progress
        self.tracker.update_migration_status(
            self.java_test_file.path, MigrationStatus.IN_PROGRESS, "Starting migration"
        )

        mapping_key = str(self.java_test_file.path.resolve())
        mapping = self.tracker.test_mappings[mapping_key]

        assert mapping.migration_status == MigrationStatus.IN_PROGRESS
        assert mapping.migration_notes == "Starting migration"

        # Update to completed
        self.tracker.update_migration_status(self.java_test_file.path, MigrationStatus.COMPLETED)

        assert mapping.migration_status == MigrationStatus.COMPLETED
        assert mapping.migration_date is not None

    def test_add_method_mapping(self):
        """Test adding method-level mappings."""
        self.tracker.register_java_test(self.java_test_file, TestCategory.UNIT_SIMPLE)

        # Add method mappings
        self.tracker.add_method_mapping(self.java_test_file.path, "testAddition", "test_addition")
        self.tracker.add_method_mapping(
            self.java_test_file.path, "testSubtraction", "test_subtraction"
        )

        mapping_key = str(self.java_test_file.path.resolve())
        mapping = self.tracker.test_mappings[mapping_key]

        assert len(mapping.test_methods) == 2
        assert mapping.test_methods["testAddition"] == "test_addition"
        assert mapping.test_methods["testSubtraction"] == "test_subtraction"
        assert mapping.migration_success_rate == 1.0

    def test_calculate_progress(self):
        """Test migration progress calculation."""
        # Register multiple tests with different statuses
        tests = [
            (self.java_test_file, TestCategory.UNIT_SIMPLE, MigrationStatus.COMPLETED),
            (self.integration_test_file, TestCategory.INTEGRATION_BASIC, MigrationStatus.FAILED),
        ]

        for test_file, category, status in tests:
            self.tracker.register_java_test(test_file, category)
            self.tracker.update_migration_status(test_file.path, status)

        # Add one more test that's not started
        another_test = TestFile(
            path=Path("/test/java/AnotherTest.java"),
            test_type=TestType.UNIT,
            class_name="AnotherTest",
        )
        self.tracker.register_java_test(another_test, TestCategory.UNIT_WITH_MOCKS)

        progress = self.tracker.calculate_progress()

        assert progress.total_java_tests == 3
        assert progress.migrated_tests == 1
        assert progress.failed_migrations == 1
        assert progress.skipped_tests == 0
        assert progress.in_progress_tests == 0
        assert progress.completion_percentage == (1 / 3) * 100
        assert progress.success_rate == (1 / 2) * 100  # 1 success out of 2 attempts

    def test_calculate_coverage_metrics(self):
        """Test coverage metrics calculation."""
        # Register Java tests
        self.tracker.register_java_test(self.java_test_file, TestCategory.UNIT_SIMPLE)
        self.tracker.register_java_test(self.integration_test_file, TestCategory.INTEGRATION_BASIC)

        # Migrate one test
        python_path = Path("/test/python/test_calculator.py")
        self.tracker.register_python_test(python_path, self.java_test_file.path, "TestCalculator")

        # Add method mappings
        self.tracker.add_method_mapping(self.java_test_file.path, "testAdd", "test_add")
        self.tracker.add_method_mapping(self.java_test_file.path, "testSub", "test_sub")

        coverage = self.tracker.calculate_coverage_metrics()

        assert coverage.java_test_count == 2
        assert coverage.python_test_count == 1
        assert coverage.mapped_tests == 1
        assert coverage.unmapped_java_tests == 1
        assert coverage.orphaned_python_tests == 0
        assert coverage.mapping_coverage == 50.0
        assert coverage.test_method_coverage == 100.0  # All methods mapped

    def test_category_breakdown(self):
        """Test category breakdown calculation."""
        # Register tests in different categories
        self.tracker.register_java_test(self.java_test_file, TestCategory.UNIT_SIMPLE)
        self.tracker.register_java_test(self.integration_test_file, TestCategory.INTEGRATION_SPRING)

        breakdown = self.tracker.get_category_breakdown()

        assert breakdown[TestCategory.UNIT_SIMPLE] == 1
        assert breakdown[TestCategory.INTEGRATION_SPRING] == 1
        assert breakdown[TestCategory.UNIT_WITH_MOCKS] == 0

    def test_status_breakdown(self):
        """Test status breakdown calculation."""
        # Register tests with different statuses
        self.tracker.register_java_test(self.java_test_file, TestCategory.UNIT_SIMPLE)
        self.tracker.update_migration_status(self.java_test_file.path, MigrationStatus.COMPLETED)

        self.tracker.register_java_test(self.integration_test_file, TestCategory.INTEGRATION_BASIC)
        self.tracker.update_migration_status(
            self.integration_test_file.path, MigrationStatus.FAILED
        )

        breakdown = self.tracker.get_status_breakdown()

        assert breakdown[MigrationStatus.COMPLETED] == 1
        assert breakdown[MigrationStatus.FAILED] == 1
        assert breakdown[MigrationStatus.NOT_STARTED] == 0

    def test_recent_migrations(self):
        """Test recent migrations tracking."""
        # Register and migrate a test
        self.tracker.register_java_test(self.java_test_file, TestCategory.UNIT_SIMPLE)
        python_path = Path("/test/python/test_calculator.py")
        self.tracker.register_python_test(python_path, self.java_test_file.path, "TestCalculator")

        recent = self.tracker.get_recent_migrations(limit=5)

        assert len(recent) == 1
        assert recent[0].java_class_name == "CalculatorTest"
        assert recent[0].python_class_name == "TestCalculator"
        assert recent[0].migration_date is not None

    def test_identify_issues(self):
        """Test issue identification."""
        # Create various issue scenarios

        # Failed migration
        self.tracker.register_java_test(self.java_test_file, TestCategory.UNIT_SIMPLE)
        self.tracker.update_migration_status(self.java_test_file.path, MigrationStatus.FAILED)

        # Incomplete method mapping
        self.tracker.register_java_test(self.integration_test_file, TestCategory.INTEGRATION_BASIC)
        self.tracker.update_migration_status(
            self.integration_test_file.path, MigrationStatus.COMPLETED
        )
        self.tracker.add_method_mapping(
            self.integration_test_file.path, "testMethod1", "test_method1"
        )
        self.tracker.add_method_mapping(
            self.integration_test_file.path, "testMethod2", ""
        )  # Incomplete

        issues = self.tracker.identify_issues()

        assert issues["failed_migrations"] == 1
        assert issues["incomplete_method_mapping"] == 1

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        # Create scenario with low completion rate
        for i in range(10):
            test_file = TestFile(
                path=Path(f"/test/java/Test{i}.java"),
                test_type=TestType.UNIT,
                class_name=f"Test{i}",
            )
            self.tracker.register_java_test(test_file, TestCategory.UNIT_SIMPLE)

        # Only migrate 2 out of 10 (20% completion)
        for i in range(2):
            test_file = TestFile(
                path=Path(f"/test/java/Test{i}.java"),
                test_type=TestType.UNIT,
                class_name=f"Test{i}",
            )
            self.tracker.update_migration_status(test_file.path, MigrationStatus.COMPLETED)

        recommendations = self.tracker.generate_recommendations()

        assert len(recommendations) > 0
        assert any("simple unit tests" in rec for rec in recommendations)

    def test_generate_migration_summary(self):
        """Test comprehensive migration summary generation."""
        # Set up test scenario
        self.tracker.register_java_test(self.java_test_file, TestCategory.UNIT_SIMPLE)
        python_path = Path("/test/python/test_calculator.py")
        self.tracker.register_python_test(python_path, self.java_test_file.path, "TestCalculator")

        summary = self.tracker.generate_migration_summary()

        assert isinstance(summary, MigrationSummary)
        assert isinstance(summary.timestamp, datetime)
        assert isinstance(summary.progress, MigrationProgress)
        assert isinstance(summary.coverage_metrics, CoverageMetrics)
        assert isinstance(summary.category_breakdown, dict)
        assert isinstance(summary.status_breakdown, dict)
        assert isinstance(summary.recent_migrations, list)
        assert isinstance(summary.issues_summary, dict)
        assert isinstance(summary.recommendations, list)

    def test_export_mapping_documentation(self):
        """Test exporting mapping documentation to JSON."""
        # Set up test data
        self.tracker.register_java_test(self.java_test_file, TestCategory.UNIT_SIMPLE)
        python_path = Path("/test/python/test_calculator.py")
        self.tracker.register_python_test(python_path, self.java_test_file.path, "TestCalculator")
        self.tracker.add_method_mapping(self.java_test_file.path, "testAdd", "test_add")

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "mapping.json"
            self.tracker.export_mapping_documentation(output_path)

            assert output_path.exists()

            with open(output_path) as f:
                data = json.load(f)

            assert "metadata" in data
            assert "mappings" in data
            assert len(data["mappings"]) == 1

            mapping_data = data["mappings"][0]
            assert mapping_data["java_class_name"] == "CalculatorTest"
            assert mapping_data["python_class_name"] == "TestCalculator"
            assert mapping_data["test_methods"]["testAdd"] == "test_add"

    def test_export_progress_report(self):
        """Test exporting progress report to JSON."""
        # Set up test data
        self.tracker.register_java_test(self.java_test_file, TestCategory.UNIT_SIMPLE)
        self.tracker.update_migration_status(self.java_test_file.path, MigrationStatus.COMPLETED)

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "progress.json"
            self.tracker.export_progress_report(output_path)

            assert output_path.exists()

            with open(output_path) as f:
                data = json.load(f)

            assert "summary" in data
            assert "progress" in data
            assert "coverage_metrics" in data
            assert "category_breakdown" in data
            assert "status_breakdown" in data
            assert "recent_migrations" in data
            assert "issues" in data
            assert "recommendations" in data

            # Verify specific values
            assert data["progress"]["total_java_tests"] == 1
            assert data["progress"]["migrated_tests"] == 1
            assert data["summary"]["completion_percentage"] == 100.0

    def test_load_mapping_documentation(self):
        """Test loading mapping documentation from JSON."""
        # Create test data
        test_data = {
            "metadata": {
                "java_source_dir": str(self.java_dir),
                "python_target_dir": str(self.python_dir),
                "tracking_start_time": datetime.now().isoformat(),
                "total_mappings": 1,
            },
            "mappings": [
                {
                    "java_test_path": str(self.java_test_file.path),
                    "python_test_path": "/test/python/test_calculator.py",
                    "java_class_name": "CalculatorTest",
                    "python_class_name": "TestCalculator",
                    "test_type": "unit",
                    "test_category": "unit_simple",
                    "migration_status": "completed",
                    "migration_date": datetime.now().isoformat(),
                    "migration_notes": "Test migration",
                    "test_methods": {"testAdd": "test_add"},
                    "migration_success_rate": 1.0,
                }
            ],
        }

        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "mapping.json"
            with open(input_path, "w") as f:
                json.dump(test_data, f)

            # Load the data
            new_tracker = CoverageTracker(self.java_dir, self.python_dir)
            new_tracker.load_mapping_documentation(input_path)

            # Verify loaded data
            mapping_key = str(self.java_test_file.path.resolve())
            assert mapping_key in new_tracker.test_mappings

            mapping = new_tracker.test_mappings[mapping_key]
            assert mapping.java_class_name == "CalculatorTest"
            assert mapping.python_class_name == "TestCalculator"
            assert mapping.migration_status == MigrationStatus.COMPLETED
            assert "testAdd" in mapping.test_methods
            assert len(new_tracker.migration_history) == 1

    def test_get_unmigrated_tests(self):
        """Test getting list of unmigrated tests."""
        # Register tests with different statuses
        self.tracker.register_java_test(self.java_test_file, TestCategory.UNIT_SIMPLE)
        self.tracker.update_migration_status(self.java_test_file.path, MigrationStatus.COMPLETED)

        self.tracker.register_java_test(self.integration_test_file, TestCategory.INTEGRATION_BASIC)
        # Leave as NOT_STARTED

        unmigrated = self.tracker.get_unmigrated_tests()

        assert len(unmigrated) == 1
        assert unmigrated[0].java_class_name == "IntegrationTest"
        assert unmigrated[0].migration_status == MigrationStatus.NOT_STARTED

    def test_get_migration_statistics(self):
        """Test getting detailed migration statistics."""
        # Set up test scenario
        self.tracker.register_java_test(self.java_test_file, TestCategory.UNIT_SIMPLE)
        self.tracker.update_migration_status(self.java_test_file.path, MigrationStatus.COMPLETED)
        self.tracker.add_method_mapping(self.java_test_file.path, "testAdd", "test_add")

        stats = self.tracker.get_migration_statistics()

        assert "completion_rate" in stats
        assert "success_rate" in stats
        assert "mapping_coverage" in stats
        assert "method_coverage" in stats
        assert "total_tests" in stats
        assert "migrated_count" in stats
        assert "average_methods_per_test" in stats
        assert "migration_velocity" in stats
        assert "time_since_start" in stats

        assert stats["completion_rate"] == 100.0
        assert stats["success_rate"] == 100.0
        assert stats["total_tests"] == 1
        assert stats["migrated_count"] == 1


class TestTestMapping:
    """Test suite for TestMapping functionality."""

    def test_is_migrated_property(self):
        """Test the is_migrated property."""
        mapping = TestMapping(
            java_test_path=Path("/test/Test.java"),
            python_test_path=None,
            java_class_name="Test",
            python_class_name=None,
            test_type=TestType.UNIT,
            test_category=TestCategory.UNIT_SIMPLE,
            migration_status=MigrationStatus.COMPLETED,
        )

        assert mapping.is_migrated is True

        mapping.migration_status = MigrationStatus.FAILED
        assert mapping.is_migrated is False

    def test_migration_success_rate(self):
        """Test migration success rate calculation."""
        mapping = TestMapping(
            java_test_path=Path("/test/Test.java"),
            python_test_path=None,
            java_class_name="Test",
            python_class_name=None,
            test_type=TestType.UNIT,
            test_category=TestCategory.UNIT_SIMPLE,
            migration_status=MigrationStatus.NOT_STARTED,
        )

        # No methods mapped
        assert mapping.migration_success_rate == 0.0

        # Add some method mappings
        mapping.test_methods = {
            "testMethod1": "test_method1",
            "testMethod2": "",  # Not mapped
            "testMethod3": "test_method3",
        }

        # 2 out of 3 methods mapped
        assert mapping.migration_success_rate == 2 / 3


class TestMigrationProgress:
    """Test suite for MigrationProgress functionality."""

    def test_completion_percentage(self):
        """Test completion percentage calculation."""
        progress = MigrationProgress(
            total_java_tests=10,
            migrated_tests=3,
            failed_migrations=2,
            skipped_tests=1,
            in_progress_tests=4,
        )

        assert progress.completion_percentage == 30.0

        # Test zero total
        progress.total_java_tests = 0
        assert progress.completion_percentage == 0.0

    def test_success_rate(self):
        """Test success rate calculation."""
        progress = MigrationProgress(
            total_java_tests=10,
            migrated_tests=6,
            failed_migrations=2,
            skipped_tests=1,
            in_progress_tests=1,
        )

        # 6 successful out of 8 attempted (6 + 2)
        assert progress.success_rate == 75.0

        # Test no attempts
        progress.migrated_tests = 0
        progress.failed_migrations = 0
        assert progress.success_rate == 0.0


class TestCoverageMetrics:
    """Test suite for CoverageMetrics functionality."""

    def test_mapping_coverage(self):
        """Test mapping coverage calculation."""
        metrics = CoverageMetrics(
            java_test_count=10,
            python_test_count=7,
            mapped_tests=7,
            unmapped_java_tests=3,
            orphaned_python_tests=0,
            test_method_coverage=85.0,
        )

        assert metrics.mapping_coverage == 70.0

        # Test zero Java tests
        metrics.java_test_count = 0
        assert metrics.mapping_coverage == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
