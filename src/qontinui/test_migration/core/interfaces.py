"""
Core interfaces for the test migration system.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from .models import (
    Dependency,
    FailureAnalysis,
    GuiModel,
    MockUsage,
    TestFailure,
    TestFile,
    TestResult,
    TestResults,
    TestType,
)


class TestScanner(ABC):
    """Interface for scanning and discovering test files."""

    @abstractmethod
    def scan_directory(self, path: Path) -> list[TestFile]:
        """Scan a directory for test files."""
        pass

    @abstractmethod
    def classify_test_type(self, test_file: TestFile) -> TestType:
        """Classify the type of test (unit vs integration)."""
        pass

    @abstractmethod
    def extract_dependencies(self, test_file: TestFile) -> list[Dependency]:
        """Extract dependencies from a test file."""
        pass


class TestTranslator(ABC):
    """Interface for translating Java tests to Python."""

    @abstractmethod
    def translate_test_file(self, test_file: TestFile) -> str:
        """Translate a complete test file to Python."""
        pass

    @abstractmethod
    def translate_test_method(self, method_code: str) -> str:
        """Translate a single test method to Python."""
        pass

    @abstractmethod
    def translate_assertions(self, assertion_code: str) -> str:
        """Translate JUnit assertions to pytest assertions."""
        pass


class MockAnalyzer(ABC):
    """Interface for analyzing mock usage in tests."""

    @abstractmethod
    def identify_mock_usage(self, test_file: TestFile) -> list[MockUsage]:
        """Identify mock usage patterns in a test file."""
        pass

    @abstractmethod
    def extract_gui_model(self, mock_usage: MockUsage) -> GuiModel | None:
        """Extract GUI model from Brobot mock usage."""
        pass


class MockGenerator(ABC):
    """Interface for generating equivalent mocks."""

    @abstractmethod
    def create_equivalent_mock(self, mock_usage: MockUsage) -> str:
        """Create equivalent Qontinui mock code."""
        pass

    @abstractmethod
    def preserve_state_simulation(self, gui_model: GuiModel) -> str:
        """Generate code to preserve GUI state simulation."""
        pass


class TestRunner(ABC):
    """Interface for running tests."""

    @abstractmethod
    def run_test_suite(self, test_directory: Path) -> TestResults:
        """Run a complete test suite."""
        pass

    @abstractmethod
    def run_single_test(self, test_file: Path) -> TestResult:
        """Run a single test file."""
        pass

    @abstractmethod
    def configure_test_environment(self, config: dict) -> None:
        """Configure the test execution environment."""
        pass


class FailureAnalyzer(ABC):
    """Interface for analyzing test failures."""

    @abstractmethod
    def analyze_failure(self, failure: TestFailure) -> FailureAnalysis:
        """Analyze a test failure to determine its cause."""
        pass

    @abstractmethod
    def is_migration_issue(self, failure: TestFailure) -> bool:
        """Determine if failure is due to migration issues."""
        pass

    @abstractmethod
    def is_code_issue(self, failure: TestFailure) -> bool:
        """Determine if failure is due to code issues."""
        pass

    @abstractmethod
    def suggest_fixes(self, analysis: FailureAnalysis) -> list[str]:
        """Suggest fixes for the analyzed failure."""
        pass


class BehaviorComparator(ABC):
    """Interface for comparing test behavior between Java and Python."""

    @abstractmethod
    def compare_outputs(self, java_output: str, python_output: str) -> bool:
        """Compare outputs from equivalent tests."""
        pass

    @abstractmethod
    def compare_behavior(self, java_test: TestFile, python_test: Path) -> bool:
        """Compare behavioral equivalence of tests."""
        pass


class DiagnosticReporter(ABC):
    """Interface for generating diagnostic reports."""

    @abstractmethod
    def generate_failure_report(self, analysis: FailureAnalysis) -> str:
        """Generate a detailed failure analysis report."""
        pass

    @abstractmethod
    def generate_migration_summary(self, results: TestResults) -> str:
        """Generate a summary of migration results."""
        pass


class MigrationOrchestrator(ABC):
    """Interface for orchestrating the complete migration process."""

    @abstractmethod
    def migrate_test_suite(self, source_path: Path, target_path: Path) -> TestResults:
        """Migrate a complete test suite from Java to Python."""
        pass

    @abstractmethod
    def validate_migration(self, migrated_tests: Path) -> TestResults:
        """Validate the migrated test suite."""
        pass
