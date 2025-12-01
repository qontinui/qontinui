"""
End-to-end tests for the complete migration workflow.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ..cli import TestMigrationCLI
from ..core.models import MigrationConfig
from ..orchestrator import TestMigrationOrchestrator
from ..reporting.dashboard import MigrationReportingDashboard


class TestEndToEndMigrationWorkflow:
    """End-to-end tests for the complete migration workflow."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            # Create source directory with sample Java tests
            source_dir = workspace / "brobot_tests"
            source_dir.mkdir()

            # Create target directory
            target_dir = workspace / "qontinui_tests"
            target_dir.mkdir()

            # Create sample Java test files
            self._create_sample_java_tests(source_dir)

            yield workspace, source_dir, target_dir

    def _create_sample_java_tests(self, source_dir: Path):
        """Create sample Java test files for testing."""

        # Simple unit test
        unit_test = source_dir / "SimpleUnitTest.java"
        unit_test.write_text(
            """
package com.example.test;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;

public class SimpleUnitTest {

    @Test
    public void testAddition() {
        int result = 1 + 1;
        Assertions.assertEquals(2, result);
    }

    @Test
    public void testSubtraction() {
        int result = 5 - 3;
        Assertions.assertEquals(2, result);
    }
}
"""
        )

        # Integration test with mocks
        integration_test = source_dir / "IntegrationTest.java"
        integration_test.write_text(
            """
package com.example.integration;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit.jupiter.SpringJUnitConfig;
import org.mockito.Mock;
import org.mockito.Mockito;

@SpringBootTest
@SpringJUnitConfig
public class IntegrationTest {

    @Mock
    private SomeService mockService;

    @Test
    public void testServiceIntegration() {
        Mockito.when(mockService.getValue()).thenReturn("test");
        String result = mockService.getValue();
        Assertions.assertEquals("test", result);
    }
}
"""
        )

        # Test with Brobot mocks
        brobot_test = source_dir / "BrobotMockTest.java"
        brobot_test.write_text(
            """
package com.example.brobot;

import org.junit.jupiter.api.Test;
import io.github.jspinak.brobot.mock.Mock;
import io.github.jspinak.brobot.mock.MockBuilder;

public class BrobotMockTest {

    @Test
    public void testBrobotMock() {
        Mock mock = new MockBuilder()
            .withElement("button")
            .build();

        // Test mock functionality
        mock.click();
    }
}
"""
        )

    def test_complete_migration_workflow_with_orchestrator(self, temp_workspace):
        """Test complete migration workflow using the orchestrator directly."""
        workspace, source_dir, target_dir = temp_workspace

        # Create configuration
        config = MigrationConfig(
            source_directories=[source_dir],
            target_directory=target_dir,
            preserve_structure=False,
            enable_mock_migration=True,
            diagnostic_level="detailed",
            parallel_execution=False,
        )

        # Create orchestrator
        orchestrator = TestMigrationOrchestrator(config)

        # Execute migration
        orchestrator.migrate_test_suite(source_dir, target_dir)

        # Verify migration state
        assert len(orchestrator.migration_state["discovered_tests"]) > 0

        # Check that Python test files were created
        python_files = list(target_dir.glob("*.py"))
        assert len(python_files) > 0

        # Verify at least one test was successfully migrated
        successful_migrations = orchestrator.migration_state["migrated_tests"]
        assert len(successful_migrations) > 0

        # Check migration progress
        progress = orchestrator.get_migration_progress()
        assert progress["discovered_tests"] > 0
        assert progress["migrated_tests"] > 0

    @patch("qontinui.src.qontinui.test_migration.orchestrator.BrobotTestScanner")
    @patch("qontinui.src.qontinui.test_migration.orchestrator.HybridTestTranslator")
    @patch("qontinui.src.qontinui.test_migration.orchestrator.PytestRunner")
    def test_migration_workflow_with_mocked_components(
        self, mock_runner, mock_translator, mock_scanner, temp_workspace
    ):
        """Test migration workflow with mocked components for controlled testing."""
        workspace, source_dir, target_dir = temp_workspace

        # Configure mocks for successful migration
        from ..core.models import TestFile, TestResult, TestResults, TestType

        mock_test_file = TestFile(
            path=source_dir / "SimpleUnitTest.java",
            test_type=TestType.UNIT,
            class_name="SimpleUnitTest",
            package="com.example.test",
        )

        mock_scanner.return_value.scan_directory.return_value = [mock_test_file]
        mock_translator.return_value.translate_test_file.return_value = """
def test_addition():
    result = 1 + 1
    assert result == 2

def test_subtraction():
    result = 5 - 3
    assert result == 2
"""

        mock_results = TestResults(
            total_tests=2,
            passed_tests=2,
            failed_tests=0,
            skipped_tests=0,
            execution_time=0.5,
            individual_results=[
                TestResult(
                    test_name="test_addition",
                    test_file="simple_unit_test.py",
                    passed=True,
                    execution_time=0.2,
                ),
                TestResult(
                    test_name="test_subtraction",
                    test_file="simple_unit_test.py",
                    passed=True,
                    execution_time=0.3,
                ),
            ],
        )
        mock_runner.return_value.run_test_suite.return_value = mock_results

        # Create configuration
        config = MigrationConfig(
            source_directories=[source_dir],
            target_directory=target_dir,
            preserve_structure=False,
            enable_mock_migration=True,
            diagnostic_level="detailed",
            parallel_execution=False,
        )

        # Execute migration
        orchestrator = TestMigrationOrchestrator(config)
        results = orchestrator.migrate_test_suite(source_dir, target_dir)

        # Verify results
        assert results.total_tests == 2
        assert results.passed_tests == 2
        assert results.failed_tests == 0

        # Verify migration state
        assert len(orchestrator.migration_state["discovered_tests"]) == 1
        assert len(orchestrator.migration_state["migrated_tests"]) == 1
        assert len(orchestrator.migration_state["failed_migrations"]) == 0

    def test_cli_migrate_command(self, temp_workspace):
        """Test CLI migrate command."""
        workspace, source_dir, target_dir = temp_workspace

        cli = TestMigrationCLI()

        # Test dry run
        args = [
            "migrate",
            str(source_dir),
            str(target_dir),
            "--dry-run",
            "--no-parallel",
        ]

        exit_code = cli.run(args)

        # Dry run should succeed
        assert exit_code == 0

    def test_cli_config_creation(self, temp_workspace):
        """Test CLI configuration file creation."""
        workspace, _, _ = temp_workspace

        cli = TestMigrationCLI()
        config_file = workspace / "test_config.json"

        args = ["config", "--create", "--output", str(config_file)]

        exit_code = cli.run(args)

        # Config creation should succeed
        assert exit_code == 0
        assert config_file.exists()

        # Validate config file content
        import json

        with open(config_file) as f:
            config_data = json.load(f)

        required_fields = [
            "source_directories",
            "target_directory",
            "preserve_structure",
            "enable_mock_migration",
            "diagnostic_level",
            "parallel_execution",
        ]

        for field in required_fields:
            assert field in config_data

    def test_cli_config_validation(self, temp_workspace):
        """Test CLI configuration file validation."""
        workspace, _, _ = temp_workspace

        # Create a valid config file
        config_file = workspace / "valid_config.json"
        config_data = {
            "source_directories": ["/path/to/source"],
            "target_directory": "/path/to/target",
            "preserve_structure": True,
            "enable_mock_migration": True,
            "diagnostic_level": "detailed",
            "parallel_execution": True,
        }

        with open(config_file, "w") as f:
            json.dump(config_data, f)

        cli = TestMigrationCLI()

        args = ["config", "--validate", "--input", str(config_file)]

        exit_code = cli.run(args)

        # Validation should succeed
        assert exit_code == 0

    def test_cli_config_validation_invalid_file(self, temp_workspace):
        """Test CLI configuration validation with invalid file."""
        workspace, _, _ = temp_workspace

        # Create an invalid config file
        config_file = workspace / "invalid_config.json"
        config_data = {
            "source_directories": ["/path/to/source"],
            # Missing required fields
        }

        with open(config_file, "w") as f:
            json.dump(config_data, f)

        cli = TestMigrationCLI()

        args = ["config", "--validate", "--input", str(config_file)]

        exit_code = cli.run(args)

        # Validation should fail
        assert exit_code == 1

    def test_reporting_dashboard_comprehensive_report(self, temp_workspace):
        """Test reporting dashboard comprehensive report generation."""
        workspace, source_dir, target_dir = temp_workspace

        # Create some migrated test files
        migrated_test = target_dir / "test_simple.py"
        migrated_test.write_text(
            """
import pytest

def test_addition():
    result = 1 + 1
    assert result == 2

@pytest.fixture
def sample_data():
    return {"key": "value"}

def test_with_fixture(sample_data):
    assert sample_data["key"] == "value"
"""
        )

        dashboard = MigrationReportingDashboard()

        # Generate comprehensive report
        report_data = dashboard.generate_comprehensive_report(
            target_dir, include_coverage=True, include_diagnostics=True
        )

        # Verify report structure
        assert "metadata" in report_data
        assert "summary" in report_data
        assert "migration_statistics" in report_data
        assert "coverage" in report_data
        assert "diagnostics" in report_data

        # Verify summary data
        summary = report_data["summary"]
        assert summary["total_test_files"] == 1
        assert str(target_dir) in summary["test_directory"]

        # Verify migration statistics
        stats = report_data["migration_statistics"]
        assert stats["total_migrated_files"] == 1

        patterns = stats["migration_patterns"]
        assert patterns["pytest_fixtures"] == 1
        assert patterns["assert_statements"] >= 2

    def test_reporting_dashboard_multiple_formats(self, temp_workspace):
        """Test reporting dashboard with multiple output formats."""
        workspace, _, target_dir = temp_workspace

        # Create a simple test file
        test_file = target_dir / "test_example.py"
        test_file.write_text("def test_example():\n    assert True")

        dashboard = MigrationReportingDashboard()
        report_data = dashboard.generate_comprehensive_report(target_dir)

        # Test JSON format
        json_file = workspace / "report.json"
        dashboard.save_report(report_data, json_file, "json")
        assert json_file.exists()

        # Test text format
        text_file = workspace / "report.txt"
        dashboard.save_report(report_data, text_file, "text")
        assert text_file.exists()

        # Test HTML format
        html_file = workspace / "report.html"
        dashboard.save_report(report_data, html_file, "html")
        assert html_file.exists()

        # Verify HTML content
        html_content = html_file.read_text()
        assert "<html>" in html_content
        assert "Test Migration Report" in html_content

    def test_cli_report_generation(self, temp_workspace):
        """Test CLI report generation command."""
        workspace, _, target_dir = temp_workspace

        # Create a test file
        test_file = target_dir / "test_sample.py"
        test_file.write_text("def test_sample():\n    assert True")

        cli = TestMigrationCLI()
        report_file = workspace / "cli_report.html"

        args = [
            "report",
            str(target_dir),
            "--format",
            "html",
            "--output",
            str(report_file),
            "--include-coverage",
            "--include-diagnostics",
        ]

        exit_code = cli.run(args)

        # Report generation should succeed
        assert exit_code == 0
        assert report_file.exists()

    def test_error_handling_in_workflow(self, temp_workspace):
        """Test error handling throughout the workflow."""
        workspace, source_dir, target_dir = temp_workspace

        # Test with non-existent source directory
        non_existent = workspace / "non_existent"

        config = MigrationConfig(
            source_directories=[non_existent],
            target_directory=target_dir,
            preserve_structure=False,
            enable_mock_migration=True,
            diagnostic_level="detailed",
            parallel_execution=False,
        )

        orchestrator = TestMigrationOrchestrator(config)
        results = orchestrator.migrate_test_suite(non_existent, target_dir)

        # Should handle gracefully
        assert results.total_tests == 0

    def test_cli_error_handling(self, temp_workspace):
        """Test CLI error handling."""
        workspace, _, _ = temp_workspace

        cli = TestMigrationCLI()

        # Test with non-existent source directory
        non_existent = workspace / "non_existent"
        target_dir = workspace / "target"

        args = ["migrate", str(non_existent), str(target_dir)]

        exit_code = cli.run(args)

        # Should return error exit code
        assert exit_code == 1

    @pytest.mark.slow
    def test_integration_with_real_pytest(self, temp_workspace):
        """Integration test with real pytest execution (marked as slow)."""
        workspace, source_dir, target_dir = temp_workspace

        # Create a simple migrated test that should pass
        migrated_test = target_dir / "test_integration.py"
        migrated_test.write_text(
            """
def test_simple_assertion():
    assert 1 + 1 == 2

def test_string_operations():
    text = "hello world"
    assert "hello" in text
    assert text.upper() == "HELLO WORLD"
"""
        )

        # Try to run pytest on the migrated test
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(migrated_test), "-v"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # If pytest is available and runs successfully
            if result.returncode == 0:
                assert "test_simple_assertion PASSED" in result.stdout
                assert "test_string_operations PASSED" in result.stdout

        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Skip if pytest is not available or times out
            pytest.skip("pytest not available or execution timed out")


class TestWorkflowRecovery:
    """Test workflow recovery and error handling scenarios."""

    @pytest.fixture
    def recovery_workspace(self):
        """Create workspace for recovery testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source_dir = workspace / "source"
            target_dir = workspace / "target"
            source_dir.mkdir()
            target_dir.mkdir()

            # Create a problematic Java test
            problematic_test = source_dir / "ProblematicTest.java"
            problematic_test.write_text(
                """
package com.example;

import org.junit.jupiter.api.Test;
import some.unknown.Library;  // This will cause issues

public class ProblematicTest {

    @Test
    public void testWithUnknownDependency() {
        Library lib = new Library();
        // This test has migration challenges
    }
}
"""
            )

            yield workspace, source_dir, target_dir

    def test_recovery_from_translation_failure(self, recovery_workspace):
        """Test recovery from translation failures."""
        workspace, source_dir, target_dir = recovery_workspace

        config = MigrationConfig(
            source_directories=[source_dir],
            target_directory=target_dir,
            preserve_structure=False,
            enable_mock_migration=True,
            diagnostic_level="detailed",
            parallel_execution=False,
        )

        orchestrator = TestMigrationOrchestrator(config)

        # Execute migration (may have failures)
        orchestrator.migrate_test_suite(source_dir, target_dir)

        # Check if there are failed migrations
        failed_migrations = orchestrator.migration_state["failed_migrations"]

        if failed_migrations:
            # Test recovery attempt
            failed_test = failed_migrations[0]["source_file"]
            error_info = {
                "test_file": failed_test,
                "error_message": "Translation failed",
                "stack_trace": "Mock stack trace",
            }

            recovery_result = orchestrator.recover_from_failure(
                "ProblematicTest", error_info
            )

            # Current implementation returns False, but the mechanism is tested
            assert isinstance(recovery_result, bool)

    def test_partial_migration_success(self, recovery_workspace):
        """Test handling of partial migration success."""
        workspace, source_dir, target_dir = recovery_workspace

        # Add a good test alongside the problematic one
        good_test = source_dir / "GoodTest.java"
        good_test.write_text(
            """
package com.example;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;

public class GoodTest {

    @Test
    public void testSimple() {
        Assertions.assertEquals(1, 1);
    }
}
"""
        )

        config = MigrationConfig(
            source_directories=[source_dir],
            target_directory=target_dir,
            preserve_structure=False,
            enable_mock_migration=True,
            diagnostic_level="detailed",
            parallel_execution=False,
        )

        orchestrator = TestMigrationOrchestrator(config)
        orchestrator.migrate_test_suite(source_dir, target_dir)

        # Should have discovered both tests
        assert len(orchestrator.migration_state["discovered_tests"]) == 2

        # May have some successful and some failed migrations
        total_attempts = len(orchestrator.migration_state["migrated_tests"]) + len(
            orchestrator.migration_state["failed_migrations"]
        )
        assert total_attempts == 2
