"""
Tests for core data models.
"""

from pathlib import Path

from qontinui.test_migration.core.models import (
    Dependency,
    FailureType,
    GuiModel,
    MigrationConfig,
    MockUsage,
    SuspectedCause,
    TestFile,
    TestMethod,
    TestType,
)


class TestDependency:
    """Tests for the Dependency model."""

    def test_dependency_creation(self):
        """Test creating a basic dependency."""
        dep = Dependency(
            java_import="org.junit.jupiter.api.Test",
            python_equivalent="pytest",
            requires_adaptation=True,
        )

        assert dep.java_import == "org.junit.jupiter.api.Test"
        assert dep.python_equivalent == "pytest"
        assert dep.requires_adaptation is True
        assert dep.adapter_function is None

    def test_dependency_with_adapter(self):
        """Test creating a dependency with adapter function."""
        dep = Dependency(
            java_import="org.junit.jupiter.api.Assertions.assertEquals",
            python_equivalent="assert",
            requires_adaptation=True,
            adapter_function="convert_assertEquals",
        )

        assert dep.adapter_function == "convert_assertEquals"


class TestGuiModel:
    """Tests for the GuiModel model."""

    def test_gui_model_creation(self):
        """Test creating a GUI model."""
        model = GuiModel(
            model_name="TestWindow",
            elements={"button1": {"type": "button", "text": "Click Me"}},
            actions=["click", "hover"],
            state_properties={"visible": True, "enabled": True},
        )

        assert model.model_name == "TestWindow"
        assert "button1" in model.elements
        assert "click" in model.actions
        assert model.state_properties["visible"] is True

    def test_gui_model_defaults(self):
        """Test GUI model with default values."""
        model = GuiModel(model_name="EmptyWindow")

        assert model.model_name == "EmptyWindow"
        assert model.elements == {}
        assert model.actions == []
        assert model.state_properties == {}


class TestMockUsage:
    """Tests for the MockUsage model."""

    def test_mock_usage_creation(self):
        """Test creating mock usage."""
        gui_model = GuiModel(model_name="TestWindow")
        mock_usage = MockUsage(
            mock_type="brobot_mock",
            mock_class="BrobotMock",
            gui_model=gui_model,
            simulation_scope="method",
        )

        assert mock_usage.mock_type == "brobot_mock"
        assert mock_usage.mock_class == "BrobotMock"
        assert mock_usage.gui_model == gui_model
        assert mock_usage.simulation_scope == "method"
        assert mock_usage.configuration == {}


class TestTestMethod:
    """Tests for the TestMethod model."""

    def test_test_method_creation(self):
        """Test creating a test method."""
        method = TestMethod(
            name="testExample",
            annotations=["@Test"],
            parameters=["String input"],
            body='assertEquals("expected", result);',
            assertions=['assertEquals("expected", result)'],
        )

        assert method.name == "testExample"
        assert "@Test" in method.annotations
        assert "String input" in method.parameters
        assert method.body == 'assertEquals("expected", result);'
        assert 'assertEquals("expected", result)' in method.assertions

    def test_test_method_defaults(self):
        """Test test method with default values."""
        method = TestMethod(name="testBasic")

        assert method.name == "testBasic"
        assert method.annotations == []
        assert method.parameters == []
        assert method.body == ""
        assert method.assertions == []
        assert method.mock_usage == []


class TestTestFile:
    """Tests for the TestFile model."""

    def test_test_file_creation(self):
        """Test creating a test file."""
        test_file = TestFile(
            path=Path("SampleTest.java"),
            test_type=TestType.UNIT,
            class_name="SampleTest",
            package="com.example.test",
        )

        assert test_file.path == Path("SampleTest.java")
        assert test_file.test_type == TestType.UNIT
        assert test_file.class_name == "SampleTest"
        assert test_file.package == "com.example.test"
        assert test_file.dependencies == []
        assert test_file.mock_usage == []
        assert test_file.test_methods == []
        assert test_file.setup_methods == []
        assert test_file.teardown_methods == []


class TestMigrationConfig:
    """Tests for the MigrationConfig model."""

    def test_migration_config_creation(self):
        """Test creating migration configuration."""
        config = MigrationConfig(
            source_directories=[Path("java/tests")],
            target_directory=Path("python/tests"),
            preserve_structure=True,
            enable_mock_migration=True,
        )

        assert config.source_directories == [Path("java/tests")]
        assert config.target_directory == Path("python/tests")
        assert config.preserve_structure is True
        assert config.enable_mock_migration is True
        assert config.diagnostic_level == "detailed"
        assert config.parallel_execution is True
        assert config.comparison_mode == "behavioral"

    def test_migration_config_defaults(self):
        """Test migration configuration with defaults."""
        config = MigrationConfig(source_directories=[Path("src")], target_directory=Path("tests"))

        assert config.java_test_patterns == ["*Test.java", "*Tests.java"]
        assert config.exclude_patterns == []


class TestEnums:
    """Tests for enum types."""

    def test_test_type_enum(self):
        """Test TestType enum values."""
        assert TestType.UNIT.value == "unit"
        assert TestType.INTEGRATION.value == "integration"
        assert TestType.UNKNOWN.value == "unknown"

    def test_failure_type_enum(self):
        """Test FailureType enum values."""
        assert FailureType.SYNTAX_ERROR.value == "syntax_error"
        assert FailureType.ASSERTION_ERROR.value == "assertion_error"
        assert FailureType.DEPENDENCY_ERROR.value == "dependency_error"
        assert FailureType.MOCK_ERROR.value == "mock_error"
        assert FailureType.RUNTIME_ERROR.value == "runtime_error"

    def test_suspected_cause_enum(self):
        """Test SuspectedCause enum values."""
        assert SuspectedCause.MIGRATION_ISSUE.value == "migration_issue"
        assert SuspectedCause.CODE_ISSUE.value == "code_issue"
        assert SuspectedCause.ENVIRONMENT_ISSUE.value == "environment_issue"
        assert SuspectedCause.UNKNOWN.value == "unknown"
