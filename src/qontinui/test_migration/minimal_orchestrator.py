"""
Minimal migration orchestrator for testing core functionality.
"""

import logging
from pathlib import Path

from config import TestMigrationConfig
from core.models import MigrationConfig, TestFile, TestResult, TestResults
from discovery.scanner import BrobotTestScanner
from execution.pytest_runner import PytestRunner


class MinimalMigrationOrchestrator:
    """
    Minimal migration orchestrator for testing core functionality.
    """

    def __init__(self, config: MigrationConfig | None = None):
        """Initialize the minimal orchestrator."""
        self.config = config or TestMigrationConfig.create_default_config(
            [], Path("tests/migrated")
        )
        self.logger = self._setup_logging()

        # Initialize basic components
        self.scanner = BrobotTestScanner()
        self.runner = PytestRunner()

        # Migration state
        self.migration_state = {
            "discovered_tests": [],
            "migrated_tests": [],
            "failed_migrations": [],
            "execution_results": None,
        }

    def discover_tests(self, source_path: Path) -> list[TestFile]:
        """Discover tests in the source directory."""
        self.logger.info(f"Discovering tests in {source_path}")

        try:
            test_files = self.scanner.scan_directory(source_path)
            self.migration_state["discovered_tests"] = test_files
            self.logger.info(f"Discovered {len(test_files)} test files")
            return test_files

        except Exception as e:
            self.logger.error(f"Test discovery failed: {str(e)}")
            raise

    def validate_migration(self, migrated_tests: Path) -> TestResults:
        """Validate migrated tests by running them."""
        self.logger.info(f"Validating migrated tests at {migrated_tests}")

        try:
            # Configure test environment
            self.runner.configure_test_environment(
                {
                    "verbose": self.config.diagnostic_level == "detailed",
                    "parallel": self.config.parallel_execution,
                    "capture_output": True,
                }
            )

            # Run the test suite
            results = self.runner.run_test_suite(migrated_tests)
            self.migration_state["execution_results"] = results

            return results

        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            return self._create_error_results(str(e))

    def get_migration_progress(self) -> dict:
        """Get current migration progress."""
        discovered = self.migration_state.get("discovered_tests", [])
        migrated = self.migration_state.get("migrated_tests", [])
        failed = self.migration_state.get("failed_migrations", [])
        return {
            "discovered_tests": len(discovered) if discovered else 0,
            "migrated_tests": len(migrated) if migrated else 0,
            "failed_migrations": len(failed) if failed else 0,
            "execution_status": (
                "completed" if self.migration_state["execution_results"] else "pending"
            ),
        }

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
                    output="",
                )
            ],
        )

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the orchestrator."""
        logger = logging.getLogger("minimal_migration_orchestrator")

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
