"""
Demonstration of the complete end-to-end migration workflow.
"""

import sys
import tempfile
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))


def create_sample_java_tests(source_dir: Path):
    """Create sample Java test files for demonstration."""

    # Simple unit test
    unit_test = source_dir / "CalculatorTest.java"
    unit_test.write_text(
        """
package com.example.calculator;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;

public class CalculatorTest {

    private Calculator calculator;

    @BeforeEach
    public void setUp() {
        calculator = new Calculator();
    }

    @Test
    public void testAddition() {
        int result = calculator.add(2, 3);
        Assertions.assertEquals(5, result);
    }

    @Test
    public void testSubtraction() {
        int result = calculator.subtract(5, 3);
        Assertions.assertEquals(2, result);
    }

    @Test
    public void testDivisionByZero() {
        Assertions.assertThrows(ArithmeticException.class, () -> {
            calculator.divide(1, 0);
        });
    }
}
"""
    )

    # Integration test
    integration_test = source_dir / "DatabaseIntegrationTest.java"
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
public class DatabaseIntegrationTest {

    @Mock
    private DatabaseService databaseService;

    @Test
    public void testDatabaseConnection() {
        Mockito.when(databaseService.isConnected()).thenReturn(true);
        boolean connected = databaseService.isConnected();
        Assertions.assertTrue(connected);
    }

    @Test
    public void testDataRetrieval() {
        String testData = "test data";
        Mockito.when(databaseService.getData("key")).thenReturn(testData);
        String result = databaseService.getData("key");
        Assertions.assertEquals(testData, result);
    }
}
"""
    )


def demonstrate_discovery_phase(orchestrator, source_dir: Path):
    """Demonstrate the test discovery phase."""
    print("Phase 1: Test Discovery")
    print("-" * 30)

    discovered_tests = orchestrator.discover_tests(source_dir)

    for i, test_file in enumerate(discovered_tests, 1):
        print(f"{i}. {test_file.path.name}")
        print(f"   Type: {test_file.test_type.value}")
        print(f"   Package: {test_file.package}")
        print(f"   Dependencies: {len(test_file.dependencies)}")
        if test_file.dependencies:
            for dep in test_file.dependencies[:3]:  # Show first 3
                print(f"     - {dep.java_import}")
            if len(test_file.dependencies) > 3:
                print(f"     ... and {len(test_file.dependencies) - 3} more")
        print()

    return discovered_tests


def demonstrate_validation_phase(orchestrator, target_dir: Path):
    """Demonstrate the validation phase."""
    print("Phase 2: Test Validation")
    print("-" * 30)

    # Create some sample migrated tests
    migrated_test1 = target_dir / "test_calculator.py"
    migrated_test1.write_text(
        """
import pytest

class TestCalculator:

    def setup_method(self):
        # This would be the migrated Calculator class
        self.calculator = MockCalculator()

    def test_addition(self):
        result = self.calculator.add(2, 3)
        assert result == 5

    def test_subtraction(self):
        result = self.calculator.subtract(5, 3)
        assert result == 2

    def test_division_by_zero(self):
        with pytest.raises(ZeroDivisionError):
            self.calculator.divide(1, 0)

class MockCalculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def divide(self, a, b):
        if b == 0:
            raise ZeroDivisionError("Division by zero")
        return a / b
"""
    )

    migrated_test2 = target_dir / "test_database_integration.py"
    migrated_test2.write_text(
        """
import pytest
from unittest.mock import Mock

@pytest.mark.integration
class TestDatabaseIntegration:

    def setup_method(self):
        self.database_service = Mock()

    def test_database_connection(self):
        self.database_service.is_connected.return_value = True
        connected = self.database_service.is_connected()
        assert connected is True

    def test_data_retrieval(self):
        test_data = "test data"
        self.database_service.get_data.return_value = test_data
        result = self.database_service.get_data("key")
        assert result == test_data
"""
    )

    print("Created migrated test files:")
    print(f"  - {migrated_test1.name}")
    print(f"  - {migrated_test2.name}")
    print()

    # Validate the migrated tests
    results = orchestrator.validate_migration(target_dir)

    print("Validation Results:")
    print(f"  Total tests: {results.total_tests}")
    print(f"  Passed: {results.passed_tests}")
    print(f"  Failed: {results.failed_tests}")
    print(f"  Execution time: {results.execution_time:.2f}s")

    return results


def demonstrate_reporting_phase(target_dir: Path):
    """Demonstrate the reporting phase."""
    print("\nPhase 3: Report Generation")
    print("-" * 30)

    try:
        from reporting.dashboard import MigrationReportingDashboard

        dashboard = MigrationReportingDashboard()

        # Generate comprehensive report
        report_data = dashboard.generate_comprehensive_report(
            target_dir, include_coverage=False, include_diagnostics=False
        )

        print("Generated comprehensive report with sections:")
        for section in report_data.keys():
            print(f"  - {section}")

        # Show summary information
        summary = report_data.get("summary", {})
        print("\nSummary:")
        print(f"  Test directory: {summary.get('test_directory', 'N/A')}")
        print(f"  Total test files: {summary.get('total_test_files', 0)}")

        # Show migration statistics
        stats = report_data.get("migration_statistics", {})
        print(f"  Migrated files: {stats.get('total_migrated_files', 0)}")

        patterns = stats.get("migration_patterns", {})
        if patterns:
            print("  Migration patterns detected:")
            for pattern, count in patterns.items():
                if count > 0:
                    print(f"    {pattern}: {count}")

        return report_data

    except Exception as e:
        print(f"Report generation failed: {e}")
        return None


def demonstrate_cli_usage():
    """Demonstrate CLI usage examples."""
    print("\nCLI Usage Examples")
    print("-" * 30)

    print("1. Migrate tests:")
    print("   python cli.py migrate /path/to/brobot/tests /path/to/qontinui/tests")
    print()

    print("2. Validate migrated tests:")
    print("   python cli.py validate /path/to/qontinui/tests")
    print()

    print("3. Generate HTML report:")
    print("   python cli.py report /path/to/qontinui/tests --format html --output report.html")
    print()

    print("4. Create configuration file:")
    print("   python cli.py config --create --output migration.json")
    print()

    print("5. Dry run migration:")
    print("   python cli.py migrate /path/to/brobot/tests /path/to/qontinui/tests --dry-run")


def main():
    """Run the complete workflow demonstration."""
    print("Brobot to Qontinui Test Migration Workflow Demonstration")
    print("=" * 60)
    print()

    try:
        from core.models import MigrationConfig
        from minimal_orchestrator import MinimalMigrationOrchestrator

        # Create temporary workspace
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source_dir = workspace / "brobot_tests"
            target_dir = workspace / "qontinui_tests"
            source_dir.mkdir()
            target_dir.mkdir()

            print(f"Workspace: {workspace}")
            print(f"Source: {source_dir}")
            print(f"Target: {target_dir}")
            print()

            # Create sample Java tests
            create_sample_java_tests(source_dir)
            print("Created sample Java test files")
            print()

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
            orchestrator = MinimalMigrationOrchestrator(config)

            # Demonstrate discovery phase
            demonstrate_discovery_phase(orchestrator, source_dir)

            # Demonstrate validation phase
            demonstrate_validation_phase(orchestrator, target_dir)

            # Demonstrate reporting phase
            demonstrate_reporting_phase(target_dir)

            # Show progress
            print("\nMigration Progress:")
            print("-" * 30)
            progress = orchestrator.get_migration_progress()
            for key, value in progress.items():
                print(f"  {key}: {value}")

            # Demonstrate CLI usage
            demonstrate_cli_usage()

            print("\n" + "=" * 60)
            print("Workflow demonstration completed successfully!")
            print("The migration system is ready for use.")

            return 0

    except Exception as e:
        print(f"Demonstration failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
