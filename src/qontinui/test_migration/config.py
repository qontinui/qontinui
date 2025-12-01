"""
Configuration management for the test migration system.
"""

import os
from pathlib import Path

from core.models import MigrationConfig


class TestMigrationConfig:
    """Configuration manager for the test migration system."""

    # Default Java test patterns
    DEFAULT_JAVA_TEST_PATTERNS = ["*Test.java", "*Tests.java", "Test*.java"]

    # Default exclude patterns
    DEFAULT_EXCLUDE_PATTERNS = [
        "*/target/*",
        "*/build/*",
        "*/.git/*",
        "*/node_modules/*",
    ]

    # Default dependency mappings
    DEFAULT_DEPENDENCY_MAPPINGS = {
        "org.junit.jupiter.api.Test": "pytest.mark.test",
        "org.junit.jupiter.api.BeforeEach": "pytest.fixture(autouse=True)",
        "org.junit.jupiter.api.AfterEach": "pytest.fixture(autouse=True)",
        "org.junit.jupiter.api.BeforeAll": "pytest.fixture(scope='module', autouse=True)",
        "org.junit.jupiter.api.AfterAll": "pytest.fixture(scope='module', autouse=True)",
        "org.junit.jupiter.api.Assertions.assertEquals": "assert {} == {}",
        "org.junit.jupiter.api.Assertions.assertTrue": "assert {}",
        "org.junit.jupiter.api.Assertions.assertFalse": "assert not {}",
        "org.junit.jupiter.api.Assertions.assertNull": "assert {} is None",
        "org.junit.jupiter.api.Assertions.assertNotNull": "assert {} is not None",
        "org.junit.jupiter.api.Assertions.assertThrows": "pytest.raises({})",
        "org.mockito.Mockito.mock": "unittest.mock.Mock",
        "org.mockito.Mockito.when": "unittest.mock.Mock.return_value",
        "org.springframework.boot.test.context.SpringBootTest": "pytest.mark.integration",
        "org.springframework.test.context.junit.jupiter.SpringJUnitConfig": "pytest.mark.integration",
    }

    @classmethod
    def create_default_config(
        cls, source_directories: list[Path], target_directory: Path
    ) -> MigrationConfig:
        """Create a default migration configuration."""
        return MigrationConfig(
            source_directories=source_directories,
            target_directory=target_directory,
            preserve_structure=True,
            enable_mock_migration=True,
            diagnostic_level="detailed",
            parallel_execution=True,
            comparison_mode="behavioral",
            java_test_patterns=cls.DEFAULT_JAVA_TEST_PATTERNS.copy(),
            exclude_patterns=cls.DEFAULT_EXCLUDE_PATTERNS.copy(),
        )

    @classmethod
    def from_environment(cls) -> MigrationConfig:
        """Create configuration from environment variables."""
        source_dirs = []
        if source_env := os.getenv("BROBOT_SOURCE_DIRS"):
            source_dirs = [Path(p.strip()) for p in source_env.split(",")]

        target_dir = Path(os.getenv("QONTINUI_TARGET_DIR", "tests/migrated"))

        return cls.create_default_config(source_dirs, target_dir)

    @classmethod
    def get_dependency_mapping(cls) -> dict[str, str]:
        """Get the default dependency mapping."""
        return cls.DEFAULT_DEPENDENCY_MAPPINGS.copy()

    @classmethod
    def get_brobot_mock_mappings(cls) -> dict[str, str]:
        """Get mappings for Brobot-specific mock classes."""
        return {
            "io.github.jspinak.brobot.mock.Mock": "qontinui.test_migration.mocks.QontinuiMock",
            "io.github.jspinak.brobot.mock.MockBuilder": "qontinui.test_migration.mocks.QontinuiMockBuilder",
            "io.github.jspinak.brobot.mock.GuiMock": "qontinui.test_migration.mocks.GuiMock",
            "io.github.jspinak.brobot.actions.BrobotSettings": "qontinui.core.settings.QontinuiSettings",
            "io.github.jspinak.brobot.database.services.AllStatesInProject": "qontinui.core.state.StateManager",
        }

    @classmethod
    def get_pytest_markers(cls) -> dict[str, str]:
        """Get pytest markers for different test types."""
        return {
            "unit": "pytest.mark.unit",
            "integration": "pytest.mark.integration",
            "slow": "pytest.mark.slow",
            "requires_gui": "pytest.mark.requires_gui",
            "mock_test": "pytest.mark.mock_test",
        }
