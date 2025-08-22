"""
Tests for the configuration module.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from qontinui.test_migration.config import TestMigrationConfig
from qontinui.test_migration.core.models import MigrationConfig


class TestTestMigrationConfig:
    """Tests for the TestMigrationConfig class."""
    
    def test_create_default_config(self):
        """Test creating default configuration."""
        source_dirs = [Path("java/tests")]
        target_dir = Path("python/tests")
        
        config = TestMigrationConfig.create_default_config(source_dirs, target_dir)
        
        assert isinstance(config, MigrationConfig)
        assert config.source_directories == source_dirs
        assert config.target_directory == target_dir
        assert config.preserve_structure is True
        assert config.enable_mock_migration is True
        assert config.diagnostic_level == "detailed"
        assert config.parallel_execution is True
        assert config.comparison_mode == "behavioral"
        assert "*Test.java" in config.java_test_patterns
        assert "*Tests.java" in config.java_test_patterns
        assert "*/target/*" in config.exclude_patterns
    
    @patch.dict(os.environ, {
        "BROBOT_SOURCE_DIRS": "java/src/test, java/integration/test",
        "QONTINUI_TARGET_DIR": "python/migrated_tests"
    })
    def test_from_environment(self):
        """Test creating configuration from environment variables."""
        config = TestMigrationConfig.from_environment()
        
        assert len(config.source_directories) == 2
        assert Path("java/src/test") in config.source_directories
        assert Path("java/integration/test") in config.source_directories
        assert config.target_directory == Path("python/migrated_tests")
    
    @patch.dict(os.environ, {}, clear=True)
    def test_from_environment_defaults(self):
        """Test creating configuration from environment with defaults."""
        config = TestMigrationConfig.from_environment()
        
        assert config.source_directories == []
        assert config.target_directory == Path("tests/migrated")
    
    def test_get_dependency_mapping(self):
        """Test getting dependency mappings."""
        mappings = TestMigrationConfig.get_dependency_mapping()
        
        assert "org.junit.jupiter.api.Test" in mappings
        assert "org.junit.jupiter.api.Assertions.assertEquals" in mappings
        assert "org.mockito.Mockito.mock" in mappings
        assert "org.springframework.boot.test.context.SpringBootTest" in mappings
        
        # Verify it returns a copy (not the original)
        original_size = len(mappings)
        mappings["new_key"] = "new_value"
        new_mappings = TestMigrationConfig.get_dependency_mapping()
        assert len(new_mappings) == original_size
    
    def test_get_brobot_mock_mappings(self):
        """Test getting Brobot mock mappings."""
        mappings = TestMigrationConfig.get_brobot_mock_mappings()
        
        assert "io.github.jspinak.brobot.mock.Mock" in mappings
        assert "io.github.jspinak.brobot.mock.MockBuilder" in mappings
        assert "io.github.jspinak.brobot.mock.GuiMock" in mappings
        assert "io.github.jspinak.brobot.actions.BrobotSettings" in mappings
        
        # Check that mappings point to Qontinui equivalents
        assert "qontinui.test_migration.mocks" in mappings["io.github.jspinak.brobot.mock.Mock"]
        assert "qontinui.core.settings" in mappings["io.github.jspinak.brobot.actions.BrobotSettings"]
    
    def test_get_pytest_markers(self):
        """Test getting pytest markers."""
        markers = TestMigrationConfig.get_pytest_markers()
        
        assert "unit" in markers
        assert "integration" in markers
        assert "slow" in markers
        assert "requires_gui" in markers
        assert "mock_test" in markers
        
        # Verify marker format
        assert markers["unit"] == "pytest.mark.unit"
        assert markers["integration"] == "pytest.mark.integration"
    
    def test_default_patterns(self):
        """Test default patterns are correctly defined."""
        assert "*Test.java" in TestMigrationConfig.DEFAULT_JAVA_TEST_PATTERNS
        assert "*Tests.java" in TestMigrationConfig.DEFAULT_JAVA_TEST_PATTERNS
        assert "Test*.java" in TestMigrationConfig.DEFAULT_JAVA_TEST_PATTERNS
        
        assert "*/target/*" in TestMigrationConfig.DEFAULT_EXCLUDE_PATTERNS
        assert "*/build/*" in TestMigrationConfig.DEFAULT_EXCLUDE_PATTERNS
        assert "*/.git/*" in TestMigrationConfig.DEFAULT_EXCLUDE_PATTERNS