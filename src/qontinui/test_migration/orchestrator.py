"""
Migration orchestrator for coordinating the complete test migration process.
"""

import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from .config import TestMigrationConfig
    from .core.interfaces import MigrationOrchestrator
    from .core.models import (
        MigrationConfig,
        TestFile,
        TestResult,
        TestResults,
        TestFailure,
        FailureType,
        SuspectedCause
    )
    from .discovery.scanner import BrobotTestScanner
    from .discovery.classifier import TestClassifier
    from .execution.hybrid_test_translator import HybridTestTranslator
    from .execution.pytest_runner import PytestRunner
    from .validation.test_failure_analyzer import TestFailureAnalyzer
    from .validation.result_validator import ResultValidator
    from .validation.coverage_tracker import CoverageTracker
    from .validation.diagnostic_reporter import DiagnosticReporter, DiagnosticReporterImpl
except ImportError:
    # Handle direct execution case
    from config import TestMigrationConfig
    from core.interfaces import MigrationOrchestrator
    from core.models import (
        MigrationConfig,
        TestFile,
        TestResult,
        TestResults,
        TestFailure,
        FailureType,
        SuspectedCause
    )
    from discovery.scanner import BrobotTestScanner
    from discovery.classifier import TestClassifier
    from execution.hybrid_test_translator import HybridTestTranslator
    from execution.pytest_runner import PytestRunner
    from validation.test_failure_analyzer import TestFailureAnalyzer
    from validation.result_validator import ResultValidator
    from validation.coverage_tracker import CoverageTracker
    from validation.diagnostic_reporter import DiagnosticReporter, DiagnosticReporterImpl


class TestMigrationOrchestrator(MigrationOrchestrator):
    """
    Orchestrates the complete test migration process from Java Brobot tests
    to Python Qontinui tests.
    
    This class coordinates:
    - Test discovery and classification
    - Test translation and generation
    - Test execution and validation
    - Error handling and recovery
    - Progress tracking and reporting
    """
    
    def __init__(self, config: Optional[MigrationConfig] = None):
        """
        Initialize the migration orchestrator.
        
        Args:
            config: Migration configuration (uses default if None)
        """
        self.config = config or TestMigrationConfig.create_default_config([], Path("tests/migrated"))
        self.logger = self._setup_logging()
        
        # Initialize components
        self.scanner = BrobotTestScanner()
        self.classifier = TestClassifier()
        self.translator = HybridTestTranslator()
        self.runner = PytestRunner()
        self.failure_analyzer = TestFailureAnalyzer()
        self.result_validator = ResultValidator()
        self.coverage_tracker = CoverageTracker(
            java_source_dir=config.source_directories[0] if config.source_directories else Path("."),
            python_target_dir=config.target_directory
        )
        self.diagnostic_reporter = DiagnosticReporterImpl()
        
        # Migration state
        self.migration_state = {
            "discovered_tests": [],
            "migrated_tests": [],
            "failed_migrations": [],
            "execution_results": None,
            "validation_results": None
        }
    
    def migrate_test_suite(self, source_path: Path, target_path: Path) -> TestResults:
        """
        Migrate a complete test suite from Java to Python.
        
        Args:
            source_path: Path to Java test source directory
            target_path: Path to Python test target directory
            
        Returns:
            TestResults with migration and execution results
        """
        self.logger.info(f"Starting test migration from {source_path} to {target_path}")
        
        try:
            # Phase 1: Discovery and Classification
            self.logger.info("Phase 1: Discovering and classifying tests")
            discovered_tests = self._discover_tests(source_path)
            
            if not discovered_tests:
                self.logger.warning("No tests discovered in source directory")
                return self._create_empty_results("No tests found to migrate")
            
            self.migration_state["discovered_tests"] = discovered_tests
            self.logger.info(f"Discovered {len(discovered_tests)} test files")
            
            # Phase 2: Translation and Generation
            self.logger.info("Phase 2: Translating and generating Python tests")
            migration_results = self._migrate_tests(discovered_tests, target_path)
            
            successful_migrations = [r for r in migration_results if r["success"]]
            failed_migrations = [r for r in migration_results if not r["success"]]
            
            self.migration_state["migrated_tests"] = successful_migrations
            self.migration_state["failed_migrations"] = failed_migrations
            
            self.logger.info(f"Successfully migrated {len(successful_migrations)} tests")
            if failed_migrations:
                self.logger.warning(f"Failed to migrate {len(failed_migrations)} tests")
            
            # Phase 3: Execution and Validation
            self.logger.info("Phase 3: Executing and validating migrated tests")
            execution_results = self._execute_migrated_tests(target_path)
            
            self.migration_state["execution_results"] = execution_results
            
            # Phase 4: Analysis and Reporting
            self.logger.info("Phase 4: Analyzing results and generating reports")
            final_results = self._analyze_and_report(execution_results, migration_results)
            
            self.logger.info("Migration process completed")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Migration failed with error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return self._create_error_results(str(e))
    
    def validate_migration(self, migrated_tests: Path) -> TestResults:
        """
        Validate a previously migrated test suite.
        
        Args:
            migrated_tests: Path to migrated Python tests
            
        Returns:
            TestResults with validation results
        """
        self.logger.info(f"Validating migrated tests at {migrated_tests}")
        
        try:
            # Execute tests
            execution_results = self._execute_migrated_tests(migrated_tests)
            
            # Validate results
            validation_results = self.result_validator.validate_test_results(execution_results)
            
            # Generate diagnostic report
            if execution_results.failed_tests > 0:
                self._analyze_failures(execution_results)
            
            return execution_results
            
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return self._create_error_results(str(e))
    
    def recover_from_failure(self, failed_test: str, error_info: Dict) -> bool:
        """
        Attempt to recover from a migration failure.
        
        Args:
            failed_test: Name of the failed test
            error_info: Error information dictionary
            
        Returns:
            True if recovery was successful
        """
        self.logger.info(f"Attempting recovery for failed test: {failed_test}")
        
        try:
            # Analyze the failure
            failure = TestFailure(
                test_name=failed_test,
                test_file=error_info.get("test_file", ""),
                error_message=error_info.get("error_message", ""),
                stack_trace=error_info.get("stack_trace", ""),
                failure_type=FailureType.UNKNOWN,
                suspected_cause=SuspectedCause.UNKNOWN
            )
            
            analysis = self.failure_analyzer.analyze_failure(failure)
            
            # Apply suggested fixes if available
            if analysis.suggested_fixes and analysis.confidence > 0.7:
                return self._apply_automated_fixes(failed_test, analysis)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {str(e)}")
            return False
    
    def get_migration_progress(self) -> Dict:
        """
        Get current migration progress information.
        
        Returns:
            Dictionary with progress details
        """
        return {
            "discovered_tests": len(self.migration_state["discovered_tests"]),
            "migrated_tests": len(self.migration_state["migrated_tests"]),
            "failed_migrations": len(self.migration_state["failed_migrations"]),
            "execution_status": "completed" if self.migration_state["execution_results"] else "pending",
            "validation_status": "completed" if self.migration_state["validation_results"] else "pending"
        }
    
    def _discover_tests(self, source_path: Path) -> List[TestFile]:
        """Discover and classify test files in the source directory."""
        try:
            # Scan for test files
            test_files = self.scanner.scan_directory(source_path)
            
            # Classify each test file
            for test_file in test_files:
                test_file.test_type = self.classifier.classify_test(test_file)
                test_file.mock_usage = self.classifier.analyze_mock_usage(test_file)
            
            return test_files
            
        except Exception as e:
            self.logger.error(f"Test discovery failed: {str(e)}")
            raise
    
    def _migrate_tests(self, test_files: List[TestFile], target_path: Path) -> List[Dict]:
        """Migrate test files to Python equivalents."""
        migration_results = []
        
        # Ensure target directory exists
        target_path.mkdir(parents=True, exist_ok=True)
        
        for test_file in test_files:
            try:
                self.logger.debug(f"Migrating test file: {test_file.path}")
                
                # Translate the test file
                python_code = self.translator.translate_test_file(test_file)
                
                # Generate target file path
                target_file = self._generate_target_path(test_file, target_path)
                
                # Write the migrated test
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_text(python_code, encoding='utf-8')
                
                migration_results.append({
                    "success": True,
                    "source_file": str(test_file.path),
                    "target_file": str(target_file),
                    "test_type": test_file.test_type.value,
                    "error": None
                })
                
                self.logger.debug(f"Successfully migrated: {test_file.path} -> {target_file}")
                
            except Exception as e:
                error_msg = f"Failed to migrate {test_file.path}: {str(e)}"
                self.logger.error(error_msg)
                
                migration_results.append({
                    "success": False,
                    "source_file": str(test_file.path),
                    "target_file": None,
                    "test_type": test_file.test_type.value,
                    "error": error_msg
                })
        
        return migration_results
    
    def _execute_migrated_tests(self, target_path: Path) -> TestResults:
        """Execute the migrated Python tests."""
        try:
            # Configure test environment
            self.runner.configure_test_environment({
                "verbose": self.config.diagnostic_level == "detailed",
                "parallel": self.config.parallel_execution,
                "capture_output": True,
                "collect_coverage": True
            })
            
            # Run the test suite
            results = self.runner.run_test_suite(target_path)
            
            # Update coverage tracking
            self.coverage_tracker.update_coverage(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Test execution failed: {str(e)}")
            raise
    
    def _analyze_and_report(self, execution_results: TestResults, migration_results: List[Dict]) -> TestResults:
        """Analyze results and generate comprehensive reports."""
        try:
            # Analyze any test failures
            if execution_results.failed_tests > 0:
                self._analyze_failures(execution_results)
            
            # Generate migration summary
            migration_summary = self._generate_migration_summary(migration_results, execution_results)
            
            # Generate diagnostic report
            diagnostic_report = self.diagnostic_reporter.generate_migration_summary(execution_results)
            
            # Log summary information
            self.logger.info(f"Migration Summary: {migration_summary}")
            self.logger.info(f"Diagnostic Report: {diagnostic_report}")
            
            return execution_results
            
        except Exception as e:
            self.logger.error(f"Analysis and reporting failed: {str(e)}")
            return execution_results
    
    def _analyze_failures(self, execution_results: TestResults):
        """Analyze test failures to categorize and suggest fixes."""
        for result in execution_results.individual_results:
            if not result.passed and result.error_message:
                failure = TestFailure(
                    test_name=result.test_name,
                    test_file=result.test_file,
                    error_message=result.error_message,
                    stack_trace=result.stack_trace or "",
                    failure_type=FailureType.RUNTIME_ERROR,
                    suspected_cause=SuspectedCause.UNKNOWN
                )
                
                analysis = self.failure_analyzer.analyze_failure(failure)
                
                self.logger.warning(f"Test failure analysis for {result.test_name}:")
                self.logger.warning(f"  Migration issue: {analysis.is_migration_issue}")
                self.logger.warning(f"  Code issue: {analysis.is_code_issue}")
                self.logger.warning(f"  Confidence: {analysis.confidence:.2f}")
                
                if analysis.suggested_fixes:
                    self.logger.info(f"  Suggested fixes: {analysis.suggested_fixes}")
    
    def _generate_target_path(self, test_file: TestFile, target_path: Path) -> Path:
        """Generate the target path for a migrated test file."""
        # Convert Java file name to Python test file name
        python_name = test_file.path.stem.replace("Test", "_test") + ".py"
        
        if self.config.preserve_structure:
            # Preserve directory structure
            relative_path = test_file.path.relative_to(test_file.path.parents[2])  # Remove src/test/java
            target_dir = target_path / relative_path.parent
        else:
            # Flat structure
            target_dir = target_path
        
        return target_dir / python_name
    
    def _generate_migration_summary(self, migration_results: List[Dict], execution_results: TestResults) -> Dict:
        """Generate a summary of the migration process."""
        successful_migrations = [r for r in migration_results if r["success"]]
        failed_migrations = [r for r in migration_results if not r["success"]]
        
        return {
            "total_discovered": len(migration_results),
            "successful_migrations": len(successful_migrations),
            "failed_migrations": len(failed_migrations),
            "migration_success_rate": len(successful_migrations) / len(migration_results) if migration_results else 0,
            "total_tests_executed": execution_results.total_tests,
            "tests_passed": execution_results.passed_tests,
            "tests_failed": execution_results.failed_tests,
            "test_success_rate": execution_results.passed_tests / execution_results.total_tests if execution_results.total_tests > 0 else 0,
            "execution_time": execution_results.execution_time
        }
    
    def _apply_automated_fixes(self, failed_test: str, analysis) -> bool:
        """Apply automated fixes based on failure analysis."""
        # This is a placeholder for automated fix application
        # In practice, this would implement specific fix strategies
        self.logger.info(f"Automated fix application not yet implemented for {failed_test}")
        return False
    
    def _create_empty_results(self, message: str) -> TestResults:
        """Create empty test results with a message."""
        return TestResults(
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            skipped_tests=0,
            execution_time=0.0,
            individual_results=[]
        )
    
    def _create_error_results(self, error_message: str) -> TestResults:
        """Create error test results."""
        return TestResults(
            total_tests=1,
            passed_tests=0,
            failed_tests=1,
            skipped_tests=0,
            execution_time=0.0,
            individual_results=[
                TestResult(
                    test_name="migration_error",
                    test_file="orchestrator",
                    passed=False,
                    execution_time=0.0,
                    error_message=error_message,
                    stack_trace="",
                    output=""
                )
            ]
        )
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the orchestrator."""
        logger = logging.getLogger("test_migration_orchestrator")
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        # Set log level based on config
        if self.config.diagnostic_level == "detailed":
            logger.setLevel(logging.DEBUG)
        elif self.config.diagnostic_level == "normal":
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
        
        return logger