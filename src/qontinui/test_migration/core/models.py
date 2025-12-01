"""
Core data models for the test migration system.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TestType(Enum):
    """Types of tests that can be migrated."""

    UNIT = "unit"
    INTEGRATION = "integration"
    UNKNOWN = "unknown"


class FailureType(Enum):
    """Types of test failures."""

    SYNTAX_ERROR = "syntax_error"
    ASSERTION_ERROR = "assertion_error"
    DEPENDENCY_ERROR = "dependency_error"
    MOCK_ERROR = "mock_error"
    RUNTIME_ERROR = "runtime_error"
    UNKNOWN = "unknown"


class SuspectedCause(Enum):
    """Suspected causes of test failures."""

    MIGRATION_ISSUE = "migration_issue"
    CODE_ISSUE = "code_issue"
    ENVIRONMENT_ISSUE = "environment_issue"
    UNKNOWN = "unknown"


@dataclass
class Dependency:
    """Represents a test dependency."""

    java_import: str
    python_equivalent: str | None = None
    requires_adaptation: bool = False
    adapter_function: str | None = None


@dataclass
class GuiModel:
    """Represents a GUI model used in Brobot mocks."""

    model_name: str
    elements: dict[str, Any] = field(default_factory=dict)
    actions: list[str] = field(default_factory=list)
    state_properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockUsage:
    """Represents mock usage in a test."""

    mock_type: str  # "brobot_mock", "spring_mock", etc.
    mock_class: str
    gui_model: GuiModel | None = None
    simulation_scope: str = "method"  # "method", "class", "suite"
    configuration: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestMethod:
    """Represents a test method."""

    name: str
    annotations: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    body: str = ""
    assertions: list[str] = field(default_factory=list)
    mock_usage: list[MockUsage] = field(default_factory=list)


@dataclass
class TestFile:
    """Represents a test file to be migrated."""

    path: Path
    test_type: TestType
    class_name: str
    package: str = ""
    dependencies: list[Dependency] = field(default_factory=list)
    mock_usage: list[MockUsage] = field(default_factory=list)
    test_methods: list[TestMethod] = field(default_factory=list)
    setup_methods: list[TestMethod] = field(default_factory=list)
    teardown_methods: list[TestMethod] = field(default_factory=list)


@dataclass
class TestFailure:
    """Represents a test failure."""

    test_name: str
    test_file: str
    error_message: str
    stack_trace: str
    failure_type: FailureType
    suspected_cause: SuspectedCause


@dataclass
class FailureAnalysis:
    """Analysis of a test failure."""

    is_migration_issue: bool
    is_code_issue: bool
    confidence: float
    suggested_fixes: list[str] = field(default_factory=list)
    diagnostic_info: dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyMapping:
    """Mapping between Java and Python dependencies."""

    java_import: str
    python_import: str
    adaptation_required: bool = False
    adapter_function: Callable[..., Any] | None = None


@dataclass
class MockMapping:
    """Mapping between Brobot and Qontinui mocks."""

    brobot_mock_class: str
    qontinui_mock_class: str
    state_preservation: bool = True
    gui_model_mapping: dict[str, str] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of a single test execution."""

    test_name: str
    test_file: str
    passed: bool
    execution_time: float
    error_message: str | None = None
    stack_trace: str | None = None
    output: str = ""


@dataclass
class TestResults:
    """Results of a test suite execution."""

    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    execution_time: float
    individual_results: list[TestResult] = field(default_factory=list)


@dataclass
class MigrationConfig:
    """Configuration for the migration process."""

    source_directories: list[Path]
    target_directory: Path
    preserve_structure: bool = True
    enable_mock_migration: bool = True
    diagnostic_level: str = "detailed"
    parallel_execution: bool = True
    comparison_mode: str = "behavioral"  # "behavioral", "output", "both"
    java_test_patterns: list[str] = field(
        default_factory=lambda: ["*Test.java", "*Tests.java"]
    )
    exclude_patterns: list[str] = field(default_factory=list)
