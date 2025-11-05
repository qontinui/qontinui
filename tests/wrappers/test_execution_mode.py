"""Tests for ExecutionMode configuration."""

import os

from qontinui.config.execution_mode import (
    ExecutionModeConfig,
    MockMode,
    get_execution_mode,
    reset_execution_mode,
    set_execution_mode,
)


class TestMockMode:
    """Test MockMode enum."""

    def test_mock_mode_values(self):
        """Test MockMode enum values."""
        assert MockMode.REAL.value == "real"
        assert MockMode.MOCK.value == "mock"
        assert MockMode.SCREENSHOT.value == "screenshot"


class TestExecutionModeConfig:
    """Test ExecutionModeConfig class."""

    def test_default_mode_is_real(self):
        """Test default mode is REAL."""
        config = ExecutionModeConfig()
        assert config.mode == MockMode.REAL
        assert config.is_real()
        assert not config.is_mock()
        assert not config.is_screenshot_mode()

    def test_mock_mode(self):
        """Test mock mode configuration."""
        config = ExecutionModeConfig(mode=MockMode.MOCK)
        assert config.mode == MockMode.MOCK
        assert config.is_mock()
        assert not config.is_real()
        assert not config.is_screenshot_mode()

    def test_screenshot_mode(self, tmp_path):
        """Test screenshot mode configuration."""
        screenshot_dir = tmp_path / "screenshots"
        screenshot_dir.mkdir()

        config = ExecutionModeConfig(mode=MockMode.SCREENSHOT, screenshot_dir=str(screenshot_dir))
        assert config.mode == MockMode.SCREENSHOT
        assert config.is_screenshot_mode()
        assert config.is_real()  # Screenshot mode is considered "real"
        assert not config.is_mock()

    def test_screenshot_dir_precedence(self, tmp_path):
        """Test that screenshot directory takes precedence over mock mode."""
        screenshot_dir = tmp_path / "screenshots"
        screenshot_dir.mkdir()

        # Even with MOCK mode, if screenshot_dir exists, is_mock() returns False
        config = ExecutionModeConfig(mode=MockMode.MOCK, screenshot_dir=str(screenshot_dir))
        assert not config.is_mock()  # Screenshots take precedence

    def test_screenshot_dir_nonexistent(self, tmp_path):
        """Test screenshot mode with non-existent directory."""
        screenshot_dir = tmp_path / "nonexistent"

        config = ExecutionModeConfig(mode=MockMode.SCREENSHOT, screenshot_dir=str(screenshot_dir))
        # Directory doesn't exist, so screenshot mode is not active
        assert not config.is_screenshot_mode()

    def test_cache_enabled(self):
        """Test cache configuration."""
        config1 = ExecutionModeConfig(cache_enabled=True)
        assert config1.cache_enabled

        config2 = ExecutionModeConfig(cache_enabled=False)
        assert not config2.cache_enabled

    def test_from_env_default(self):
        """Test loading from environment with defaults."""
        # Clear any existing env vars
        os.environ.pop("QONTINUI_MOCK_MODE", None)
        os.environ.pop("QONTINUI_SCREENSHOT_DIR", None)
        os.environ.pop("QONTINUI_MOCK_CACHE", None)

        config = ExecutionModeConfig.from_env()
        assert config.mode == MockMode.REAL
        assert config.screenshot_dir is None
        assert config.cache_enabled is True

    def test_from_env_mock_mode(self):
        """Test loading mock mode from environment."""
        os.environ["QONTINUI_MOCK_MODE"] = "mock"

        try:
            config = ExecutionModeConfig.from_env()
            assert config.mode == MockMode.MOCK
            assert config.is_mock()
        finally:
            os.environ.pop("QONTINUI_MOCK_MODE", None)

    def test_from_env_screenshot_mode(self, tmp_path):
        """Test loading screenshot mode from environment."""
        screenshot_dir = tmp_path / "screenshots"
        screenshot_dir.mkdir()

        os.environ["QONTINUI_MOCK_MODE"] = "screenshot"
        os.environ["QONTINUI_SCREENSHOT_DIR"] = str(screenshot_dir)

        try:
            config = ExecutionModeConfig.from_env()
            assert config.mode == MockMode.SCREENSHOT
            assert config.screenshot_dir == str(screenshot_dir)
            assert config.is_screenshot_mode()
        finally:
            os.environ.pop("QONTINUI_MOCK_MODE", None)
            os.environ.pop("QONTINUI_SCREENSHOT_DIR", None)

    def test_from_env_cache_disabled(self):
        """Test loading cache disabled from environment."""
        os.environ["QONTINUI_MOCK_CACHE"] = "false"

        try:
            config = ExecutionModeConfig.from_env()
            assert not config.cache_enabled
        finally:
            os.environ.pop("QONTINUI_MOCK_CACHE", None)

    def test_from_env_invalid_mode(self):
        """Test loading invalid mode from environment."""
        os.environ["QONTINUI_MOCK_MODE"] = "invalid"

        try:
            config = ExecutionModeConfig.from_env()
            # Should default to REAL on invalid value
            assert config.mode == MockMode.REAL
        finally:
            os.environ.pop("QONTINUI_MOCK_MODE", None)


class TestGlobalExecutionMode:
    """Test global execution mode management."""

    def setup_method(self):
        """Reset execution mode before each test."""
        reset_execution_mode()

    def teardown_method(self):
        """Reset execution mode after each test."""
        reset_execution_mode()

    def test_get_execution_mode_default(self):
        """Test getting default execution mode."""
        mode = get_execution_mode()
        assert isinstance(mode, ExecutionModeConfig)
        # Default should be loaded from environment (REAL if not set)

    def test_set_execution_mode(self):
        """Test setting execution mode."""
        config = ExecutionModeConfig(mode=MockMode.MOCK)
        set_execution_mode(config)

        retrieved = get_execution_mode()
        assert retrieved.mode == MockMode.MOCK
        assert retrieved.is_mock()

    def test_set_execution_mode_persistence(self):
        """Test execution mode persists across get calls."""
        config = ExecutionModeConfig(mode=MockMode.MOCK)
        set_execution_mode(config)

        # Multiple get calls should return same config
        mode1 = get_execution_mode()
        mode2 = get_execution_mode()

        assert mode1.is_mock()
        assert mode2.is_mock()

    def test_reset_execution_mode(self):
        """Test resetting execution mode."""
        # Set custom mode
        config = ExecutionModeConfig(mode=MockMode.MOCK)
        set_execution_mode(config)
        assert get_execution_mode().is_mock()

        # Reset
        reset_execution_mode()

        # Should reload from environment (default REAL)
        mode = get_execution_mode()
        # Can't assert specific mode as it depends on environment
        assert isinstance(mode, ExecutionModeConfig)

    def test_mode_switching(self):
        """Test switching between modes."""
        # Start in mock mode
        set_execution_mode(ExecutionModeConfig(mode=MockMode.MOCK))
        assert get_execution_mode().is_mock()

        # Switch to real mode
        set_execution_mode(ExecutionModeConfig(mode=MockMode.REAL))
        assert get_execution_mode().is_real()
        assert not get_execution_mode().is_mock()

        # Switch to screenshot mode
        set_execution_mode(ExecutionModeConfig(mode=MockMode.SCREENSHOT))
        assert get_execution_mode().mode == MockMode.SCREENSHOT
