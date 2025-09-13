"""
Integration tests for the migration orchestrator.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from ..orchestrator import TestMigrationOrchestrator
from ..core.models import (
    MigrationConfig,
    TestFile,
    TestType,
    TestResult,
    TestResults,
    TestFailure,
    FailureType,
    SuspectedCause
)


class TestMigrationOrchestrator:
    """Test cases for the TestMigrationOrchestrator class."""
    
    @pytest.fixture
    def temp_directories(self):
        """Create temporary directories for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_dir = temp_path / "source"
            target_dir = temp_path / "target"
            source_dir.mkdir()
            target_dir.mkdir()
            
            yield source_dir, target_dir
    
    @pytest.fixture
    def sample_config(self, temp_directories):
        """Create a sample migration configuration."""
        source_dir, target_dir = temp_directories
        return MigrationConfig(
            source_directories=[source_dir],
            target_directory=target_dir,
            preserve_structure=True,
            enable_mock_migration=True,
            diagnostic_level="detailed",
            parallel_execution=False,  # Disable for testing
            comparison_mode="behavioral"
        )
    
    @pytest.fixture
    def orchestrator(self, sample_config):
        """Create a migration orchestrator instance."""
        return TestMigrationOrchestrator(sample_config)
    
    @pytest.fixture
    def sample_java_test(self, temp_directories):
        """Create a sample Java test file."""
        source_dir, _ = temp_directories
        test_file = source_dir / "SampleTest.java"
        test_content = """
package com.example.test;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Assertions;

public class SampleTest {
    
    @Test
    public void testSample() {
        Assertions.assertEquals(1, 1);
    }
}
"""
        test_file.write_text(test_content)
        return test_file
    
    def test_orchestrator_initialization(self, sample_config):
        """Test that the orchestrator initializes correctly."""
        orchestrator = TestMigrationOrchestrator(sample_config)
        
        assert orchestrator.config == sample_config
        assert orchestrator.scanner is not None
        assert orchestrator.classifier is not None
        assert orchestrator.translator is not None
        assert orchestrator.runner is not None
        assert orchestrator.failure_analyzer is not None
        assert orchestrator.result_validator is not None
        assert orchestrator.coverage_tracker is not None
        assert orchestrator.diagnostic_reporter is not None
        
        # Check initial migration state
        assert orchestrator.migration_state["discovered_tests"] == []
        assert orchestrator.migration_state["migrated_tests"] == []
        assert orchestrator.migration_state["failed_migrations"] == []
        assert orchestrator.migration_state["execution_results"] is None
        assert orchestrator.migration_state["validation_results"] is None
    
    def test_orchestrator_default_config(self):
        """Test orchestrator with default configuration."""
        orchestrator = TestMigrationOrchestrator()
        
        assert orchestrator.config is not None
        assert orchestrator.config.preserve_structure is True
        assert orchestrator.config.enable_mock_migration is True
    
    @patch('qontinui.src.qontinui.test_migration.orchestrator.BrobotTestScanner')
    @patch('qontinui.src.qontinui.test_migration.orchestrator.TestClassifier')
    @patch('qontinui.src.qontinui.test_migration.orchestrator.HybridTestTranslator')
    @patch('qontinui.src.qontinui.test_migration.orchestrator.PytestRunner')
    def test_migrate_test_suite_success(
        self, 
        mock_runner, 
        mock_translator, 
        mock_classifier, 
        mock_scanner,
        orchestrator,
        temp_directories,
        sample_java_test
    ):
        """Test successful test suite migration."""
        source_dir, target_dir = temp_directories
        
        # Mock test file
        test_file = TestFile(
            path=sample_java_test,
            test_type=TestType.UNIT,
            class_name="SampleTest",
            package="com.example.test"
        )
        
        # Configure mocks
        mock_scanner.return_value.scan_directory.return_value = [test_file]
        mock_classifier.return_value.classify_test.return_value = TestType.UNIT
        mock_classifier.return_value.analyze_mock_usage.return_value = []
        mock_translator.return_value.translate_test_file.return_value = "def test_sample():\n    assert 1 == 1"
        
        # Mock successful test execution
        mock_execution_results = TestResults(
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            skipped_tests=0,
            execution_time=0.1,
            individual_results=[
                TestResult(
                    test_name="test_sample",
                    test_file="sample_test.py",
                    passed=True,
                    execution_time=0.1
                )
            ]
        )
        mock_runner.return_value.run_test_suite.return_value = mock_execution_results
        
        # Execute migration
        results = orchestrator.migrate_test_suite(source_dir, target_dir)
        
        # Verify results
        assert results.total_tests == 1
        assert results.passed_tests == 1
        assert results.failed_tests == 0
        
        # Verify migration state
        assert len(orchestrator.migration_state["discovered_tests"]) == 1
        assert len(orchestrator.migration_state["migrated_tests"]) == 1
        assert len(orchestrator.migration_state["failed_migrations"]) == 0
    
    @patch('qontinui.src.qontinui.test_migration.orchestrator.BrobotTestScanner')
    def test_migrate_test_suite_no_tests_found(
        self, 
        mock_scanner,
        orchestrator,
        temp_directories
    ):
        """Test migration when no tests are found."""
        source_dir, target_dir = temp_directories
        
        # Configure mock to return no tests
        mock_scanner.return_value.scan_directory.return_value = []
        
        # Execute migration
        results = orchestrator.migrate_test_suite(source_dir, target_dir)
        
        # Verify results
        assert results.total_tests == 0
        assert results.passed_tests == 0
        assert results.failed_tests == 0
    
    @patch('qontinui.src.qontinui.test_migration.orchestrator.BrobotTestScanner')
    @patch('qontinui.src.qontinui.test_migration.orchestrator.TestClassifier')
    @patch('qontinui.src.qontinui.test_migration.orchestrator.HybridTestTranslator')
    def test_migrate_test_suite_translation_failure(
        self,
        mock_translator,
        mock_classifier,
        mock_scanner,
        orchestrator,
        temp_directories,
        sample_java_test
    ):
        """Test migration with translation failures."""
        source_dir, target_dir = temp_directories
        
        # Mock test file
        test_file = TestFile(
            path=sample_java_test,
            test_type=TestType.UNIT,
            class_name="SampleTest",
            package="com.example.test"
        )
        
        # Configure mocks
        mock_scanner.return_value.scan_directory.return_value = [test_file]
        mock_classifier.return_value.classify_test.return_value = TestType.UNIT
        mock_classifier.return_value.analyze_mock_usage.return_value = []
        
        # Mock translation failure
        mock_translator.return_value.translate_test_file.side_effect = Exception("Translation failed")
        
        # Execute migration
        results = orchestrator.migrate_test_suite(source_dir, target_dir)
        
        # Verify migration state shows failure
        assert len(orchestrator.migration_state["discovered_tests"]) == 1
        assert len(orchestrator.migration_state["migrated_tests"]) == 0
        assert len(orchestrator.migration_state["failed_migrations"]) == 1
        
        # Check failed migration details
        failed_migration = orchestrator.migration_state["failed_migrations"][0]
        assert failed_migration["success"] is False
        assert "Translation failed" in failed_migration["error"]
    
    @patch('qontinui.src.qontinui.test_migration.orchestrator.PytestRunner')
    def test_validate_migration(self, mock_runner, orchestrator, temp_directories):
        """Test migration validation."""
        _, target_dir = temp_directories
        
        # Create a sample migrated test file
        test_file = target_dir / "test_sample.py"
        test_file.write_text("def test_sample():\n    assert 1 == 1")
        
        # Mock test execution results
        mock_results = TestResults(
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            skipped_tests=0,
            execution_time=0.1,
            individual_results=[]
        )
        mock_runner.return_value.run_test_suite.return_value = mock_results
        
        # Execute validation
        results = orchestrator.validate_migration(target_dir)
        
        # Verify results
        assert results.total_tests == 1
        assert results.passed_tests == 1
        assert results.failed_tests == 0
    
    def test_get_migration_progress(self, orchestrator):
        """Test migration progress tracking."""
        # Initial progress
        progress = orchestrator.get_migration_progress()
        
        assert progress["discovered_tests"] == 0
        assert progress["migrated_tests"] == 0
        assert progress["failed_migrations"] == 0
        assert progress["execution_status"] == "pending"
        assert progress["validation_status"] == "pending"
        
        # Update migration state
        orchestrator.migration_state["discovered_tests"] = [Mock(), Mock()]
        orchestrator.migration_state["migrated_tests"] = [Mock()]
        orchestrator.migration_state["failed_migrations"] = [Mock()]
        orchestrator.migration_state["execution_results"] = Mock()
        
        # Check updated progress
        progress = orchestrator.get_migration_progress()
        
        assert progress["discovered_tests"] == 2
        assert progress["migrated_tests"] == 1
        assert progress["failed_migrations"] == 1
        assert progress["execution_status"] == "completed"
        assert progress["validation_status"] == "pending"
    
    @patch('qontinui.src.qontinui.test_migration.orchestrator.TestFailureAnalyzer')
    def test_recover_from_failure(self, mock_analyzer, orchestrator):
        """Test failure recovery mechanism."""
        # Mock failure analysis
        mock_analysis = Mock()
        mock_analysis.suggested_fixes = ["Fix suggestion 1", "Fix suggestion 2"]
        mock_analysis.confidence = 0.8
        mock_analyzer.return_value.analyze_failure.return_value = mock_analysis
        
        # Test recovery attempt
        error_info = {
            "test_file": "test_sample.py",
            "error_message": "Test failed",
            "stack_trace": "Stack trace here"
        }
        
        # Note: Current implementation returns False as automated fixes are not implemented
        result = orchestrator.recover_from_failure("test_sample", error_info)
        
        # Verify analysis was called
        mock_analyzer.return_value.analyze_failure.assert_called_once()
        
        # Current implementation should return False
        assert result is False
    
    def test_generate_target_path_preserve_structure(self, orchestrator, temp_directories):
        """Test target path generation with structure preservation."""
        source_dir, target_dir = temp_directories
        
        # Create test file with nested structure
        nested_dir = source_dir / "com" / "example" / "test"
        nested_dir.mkdir(parents=True)
        java_file = nested_dir / "SampleTest.java"
        java_file.write_text("// Test content")
        
        test_file = TestFile(
            path=java_file,
            test_type=TestType.UNIT,
            class_name="SampleTest",
            package="com.example.test"
        )
        
        # Generate target path
        target_path = orchestrator._generate_target_path(test_file, target_dir)
        
        # Verify structure is preserved and name is converted
        assert target_path.name == "Sample_test.py"
        assert "com" in str(target_path)
        assert "example" in str(target_path)
    
    def test_generate_target_path_flat_structure(self, temp_directories):
        """Test target path generation without structure preservation."""
        source_dir, target_dir = temp_directories
        
        # Create config with flat structure
        config = MigrationConfig(
            source_directories=[source_dir],
            target_directory=target_dir,
            preserve_structure=False
        )
        orchestrator = TestMigrationOrchestrator(config)
        
        # Create test file
        java_file = source_dir / "SampleTest.java"
        java_file.write_text("// Test content")
        
        test_file = TestFile(
            path=java_file,
            test_type=TestType.UNIT,
            class_name="SampleTest",
            package="com.example.test"
        )
        
        # Generate target path
        target_path = orchestrator._generate_target_path(test_file, target_dir)
        
        # Verify flat structure
        assert target_path.parent == target_dir
        assert target_path.name == "Sample_test.py"
    
    def test_error_handling_in_migration(self, orchestrator, temp_directories):
        """Test error handling during migration process."""
        source_dir, target_dir = temp_directories
        
        # Test with non-existent source directory
        non_existent = source_dir / "non_existent"
        
        results = orchestrator.migrate_test_suite(non_existent, target_dir)
        
        # Should handle gracefully and return empty results
        assert results.total_tests == 0
    
    def test_logging_configuration(self, sample_config):
        """Test logging configuration based on diagnostic level."""
        # Test detailed logging
        config_detailed = sample_config
        config_detailed.diagnostic_level = "detailed"
        orchestrator_detailed = TestMigrationOrchestrator(config_detailed)
        assert orchestrator_detailed.logger.level == 10  # DEBUG level
        
        # Test normal logging
        config_normal = sample_config
        config_normal.diagnostic_level = "normal"
        orchestrator_normal = TestMigrationOrchestrator(config_normal)
        assert orchestrator_normal.logger.level == 20  # INFO level
        
        # Test minimal logging
        config_minimal = sample_config
        config_minimal.diagnostic_level = "minimal"
        orchestrator_minimal = TestMigrationOrchestrator(config_minimal)
        assert orchestrator_minimal.logger.level == 30  # WARNING level


class TestMigrationOrchestratorIntegration:
    """Integration tests for the migration orchestrator with real components."""
    
    @pytest.fixture
    def integration_setup(self):
        """Set up integration test environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_dir = temp_path / "source"
            target_dir = temp_path / "target"
            source_dir.mkdir()
            target_dir.mkdir()
            
            # Create a simple Java test file
            java_test = source_dir / "SimpleTest.java"
            java_content = """
package com.example;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class SimpleTest {
    
    @Test
    public void testAddition() {
        assertEquals(2, 1 + 1);
    }
    
    @Test
    public void testSubtraction() {
        assertEquals(0, 1 - 1);
    }
}
"""
            java_test.write_text(java_content)
            
            config = MigrationConfig(
                source_directories=[source_dir],
                target_directory=target_dir,
                preserve_structure=False,
                enable_mock_migration=True,
                diagnostic_level="detailed",
                parallel_execution=False
            )
            
            yield source_dir, target_dir, config
    
    def test_end_to_end_migration_workflow(self, integration_setup):
        """Test complete end-to-end migration workflow."""
        source_dir, target_dir, config = integration_setup
        
        # Create orchestrator
        orchestrator = TestMigrationOrchestrator(config)
        
        # Execute migration
        results = orchestrator.migrate_test_suite(source_dir, target_dir)
        
        # Verify migration completed
        assert orchestrator.migration_state["discovered_tests"]
        
        # Check that Python test files were created
        python_files = list(target_dir.glob("*.py"))
        assert len(python_files) > 0
        
        # Verify migration progress
        progress = orchestrator.get_migration_progress()
        assert progress["discovered_tests"] > 0
    
    @pytest.mark.slow
    def test_migration_with_real_pytest_execution(self, integration_setup):
        """Test migration with actual pytest execution (marked as slow)."""
        source_dir, target_dir, config = integration_setup
        
        # Create orchestrator
        orchestrator = TestMigrationOrchestrator(config)
        
        # Execute migration
        results = orchestrator.migrate_test_suite(source_dir, target_dir)
        
        # If migration was successful, we should have execution results
        if orchestrator.migration_state["migrated_tests"]:
            assert orchestrator.migration_state["execution_results"] is not None