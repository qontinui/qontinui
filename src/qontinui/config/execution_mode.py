"""Execution mode configuration for mock vs real automation.

Provides configuration and mode detection for routing automation
to mock implementations (historical playback) or real HAL implementations
(actual GUI automation).
"""

import logging
import os
from enum import Enum

logger = logging.getLogger(__name__)


class MockMode(Enum):
    """Mock execution modes."""

    REAL = "real"  # Full automation via HAL
    MOCK = "mock"  # Historical data playback
    SCREENSHOT = "screenshot"  # Screenshot-based testing


class ExecutionModeConfig:
    """Configuration for execution mode.

    Controls whether automation uses real implementations (HAL layer)
    or mock implementations (historical snapshots).

    Example:
        # Set mock mode
        config = ExecutionModeConfig(mode=MockMode.MOCK)
        set_execution_mode(config)

        # Now all automation will use mock implementations
        find = Find()
        result = find.perform(...)  # Uses MockFind

    Example:
        # Set real mode (default)
        config = ExecutionModeConfig(mode=MockMode.REAL)
        set_execution_mode(config)

        # Now all automation will use real HAL
        find = Find()
        result = find.perform(...)  # Uses HAL screen capture + OpenCV
    """

    def __init__(
        self,
        mode: MockMode = MockMode.REAL,
        screenshot_dir: str | None = None,
        cache_enabled: bool = True,
    ):
        """Initialize execution mode configuration.

        Args:
            mode: Execution mode (REAL, MOCK, or SCREENSHOT)
            screenshot_dir: Directory for screenshot-based testing
            cache_enabled: Enable screenshot caching in mock mode
        """
        self.mode = mode
        self.screenshot_dir = screenshot_dir
        self.cache_enabled = cache_enabled

        logger.debug(
            f"ExecutionModeConfig initialized: mode={mode.value}, "
            f"screenshot_dir={screenshot_dir}, cache={cache_enabled}"
        )

    def is_mock(self) -> bool:
        """Check if running in mock mode.

        Screenshots take precedence over mock mode (like Brobot pattern).
        If screenshots are configured and exist, we're not in pure mock mode.

        Returns:
            True if in mock mode, False otherwise
        """
        # Screenshots take precedence (realistic testing wins)
        if self.screenshot_dir and os.path.exists(self.screenshot_dir):
            logger.debug(f"Screenshot directory exists: {self.screenshot_dir}, not using mock mode")
            return False

        is_mock = self.mode == MockMode.MOCK
        logger.debug(f"is_mock check: {is_mock} (mode={self.mode.value})")
        return is_mock

    def is_screenshot_mode(self) -> bool:
        """Check if running in screenshot-based testing mode.

        Returns:
            True if in screenshot mode with valid directory
        """
        return (
            self.mode == MockMode.SCREENSHOT
            and self.screenshot_dir is not None
            and os.path.exists(self.screenshot_dir)
        )

    def is_real(self) -> bool:
        """Check if running in real automation mode.

        Returns:
            True if using real HAL implementations
        """
        return self.mode == MockMode.REAL or self.is_screenshot_mode()

    @staticmethod
    def from_env() -> "ExecutionModeConfig":
        """Create configuration from environment variables.

        Environment variables:
            QONTINUI_MOCK_MODE: "real", "mock", or "screenshot" (default: "real")
            QONTINUI_SCREENSHOT_DIR: Directory for screenshots (optional)
            QONTINUI_MOCK_CACHE: "true" or "false" (default: "true")

        Returns:
            ExecutionModeConfig instance based on environment

        Example:
            # Set environment
            os.environ["QONTINUI_MOCK_MODE"] = "mock"

            # Load config
            config = ExecutionModeConfig.from_env()
            assert config.is_mock() == True
        """
        mode_str = os.getenv("QONTINUI_MOCK_MODE", "real")
        try:
            mode = MockMode(mode_str)
        except ValueError:
            logger.warning(f"Invalid QONTINUI_MOCK_MODE: {mode_str}, defaulting to REAL")
            mode = MockMode.REAL

        screenshot_dir = os.getenv("QONTINUI_SCREENSHOT_DIR")
        cache_enabled = os.getenv("QONTINUI_MOCK_CACHE", "true").lower() == "true"

        config = ExecutionModeConfig(mode, screenshot_dir, cache_enabled)
        logger.info(f"ExecutionModeConfig loaded from environment: {mode.value}")
        return config


# Global instance (can be overridden)
_execution_mode: ExecutionModeConfig | None = None


def get_execution_mode() -> ExecutionModeConfig:
    """Get current execution mode configuration.

    Returns the global execution mode configuration. If not set,
    creates one from environment variables.

    Returns:
        Current ExecutionModeConfig instance

    Example:
        mode = get_execution_mode()
        if mode.is_mock():
            print("Running in mock mode")
    """
    global _execution_mode
    if _execution_mode is None:
        _execution_mode = ExecutionModeConfig.from_env()
        logger.debug("Global execution mode initialized from environment")
    return _execution_mode


def set_execution_mode(config: ExecutionModeConfig):
    """Set execution mode configuration.

    Updates the global execution mode. All subsequent automation
    will use this configuration for routing to mock vs real.

    Args:
        config: New ExecutionModeConfig to use globally

    Example:
        # Switch to mock mode
        set_execution_mode(ExecutionModeConfig(mode=MockMode.MOCK))

        # All automation now uses mock implementations
        find = Find()
        result = find.perform(...)  # Uses MockFind
    """
    global _execution_mode
    _execution_mode = config
    logger.info(f"Global execution mode set to: {config.mode.value}")


def reset_execution_mode():
    """Reset execution mode to environment defaults.

    Clears the global configuration and forces reload from
    environment variables on next access.

    Example:
        # Override temporarily
        set_execution_mode(ExecutionModeConfig(mode=MockMode.MOCK))

        # Reset back to environment
        reset_execution_mode()

        # Now uses QONTINUI_MOCK_MODE env var again
        mode = get_execution_mode()
    """
    global _execution_mode
    _execution_mode = None
    logger.debug("Global execution mode reset to None (will reload from environment)")
