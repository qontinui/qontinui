"""Framework settings - ported from Qontinui framework.

Global configuration settings using singleton pattern with Pydantic validation.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from .qontinui_properties import QontinuiProperties

logger = logging.getLogger(__name__)


class FrameworkSettings:
    """Global configuration settings for the Qontinui framework.

    Port of FrameworkSettings from Qontinui framework with Python improvements.

    FrameworkSettings serves as the central configuration repository for the
    entire framework, controlling behavior across all components from action
    execution to data collection. These settings enable customization of the
    framework's operation without modifying code, supporting different
    deployment scenarios from development to production.

    This implementation uses:
    - Singleton pattern for global access
    - Pydantic for validation
    - Property decorators for controlled access
    - Environment variable support

    Setting categories:
    - Mouse Control: Fine-tune mouse action timing and movement
    - Mock Mode: Configure simulated execution for testing
    - Data Collection: Control screenshot capture and dataset building
    - Visual Analysis: Configure color profiling and image processing
    - Testing Support: Settings for unit tests and testing scenarios

    Usage:
        # Get singleton instance
        settings = FrameworkSettings.get_instance()

        # Access settings
        delay = settings.mouse_move_delay

        # Update settings
        settings.mock = True

        # Load from file
        settings.load_from_yaml('config.yaml')

        # Save current configuration
        settings.save_to_yaml('current_config.yaml')
    """

    _instance: Optional["FrameworkSettings"] = None

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize settings if not already done."""
        if not self._initialized:
            self._properties = QontinuiProperties()
            self._initialized = True
            logger.info("FrameworkSettings initialized with defaults")

    @classmethod
    def get_instance(cls) -> "FrameworkSettings":
        """Get the singleton instance.

        Returns:
            The global FrameworkSettings instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset to default configuration."""
        cls._instance = None
        logger.info("FrameworkSettings reset to defaults")

    # Mouse Control Properties
    @property
    def mouse_move_delay(self) -> float:
        """Delay for mouse movement in seconds."""
        return self._properties.mouse.move_delay

    @mouse_move_delay.setter
    def mouse_move_delay(self, value: float) -> None:
        self._properties.mouse.move_delay = value

    @property
    def pause_before_mouse_down(self) -> float:
        """Pause before mouse down in seconds."""
        return self._properties.mouse.pause_before_down

    @pause_before_mouse_down.setter
    def pause_before_mouse_down(self, value: float) -> None:
        self._properties.mouse.pause_before_down = value

    @property
    def pause_after_mouse_down(self) -> float:
        """Pause after mouse down in seconds."""
        return self._properties.mouse.pause_after_down

    @pause_after_mouse_down.setter
    def pause_after_mouse_down(self, value: float) -> None:
        self._properties.mouse.pause_after_down = value

    @property
    def pause_before_mouse_up(self) -> float:
        """Pause before mouse up in seconds."""
        return self._properties.mouse.pause_before_up

    @pause_before_mouse_up.setter
    def pause_before_mouse_up(self, value: float) -> None:
        self._properties.mouse.pause_before_up = value

    @property
    def pause_after_mouse_up(self) -> float:
        """Pause after mouse up in seconds."""
        return self._properties.mouse.pause_after_up

    @pause_after_mouse_up.setter
    def pause_after_mouse_up(self, value: float) -> None:
        self._properties.mouse.pause_after_up = value

    # Core Settings
    @property
    def mock(self) -> bool:
        """Enable mock mode for testing without GUI."""
        return self._properties.core.mock

    @mock.setter
    def mock(self, value: bool) -> None:
        self._properties.core.mock = value
        logger.info(f"Mock mode {'enabled' if value else 'disabled'}")

    @property
    def headless(self) -> bool:
        """Run in headless mode without display."""
        return self._properties.core.headless

    @headless.setter
    def headless(self, value: bool) -> None:
        self._properties.core.headless = value

    @property
    def image_path(self) -> str:
        """Path to image resources."""
        return self._properties.core.image_path

    @image_path.setter
    def image_path(self, value: str) -> None:
        self._properties.core.image_path = value

    # Screenshot Settings
    @property
    def save_snapshots(self) -> bool:
        """Save screenshots during execution."""
        return self._properties.screenshot.save_snapshots

    @save_snapshots.setter
    def save_snapshots(self, value: bool) -> None:
        self._properties.screenshot.save_snapshots = value

    @property
    def screenshot_path(self) -> str:
        """Path to save screenshots."""
        return self._properties.screenshot.path

    @screenshot_path.setter
    def screenshot_path(self, value: str) -> None:
        self._properties.screenshot.path = value

    @property
    def max_history(self) -> int:
        """Maximum screenshot history to maintain."""
        return self._properties.screenshot.max_history

    @max_history.setter
    def max_history(self, value: int) -> None:
        self._properties.screenshot.max_history = value

    # Mock Timings
    @property
    def mock_click_duration(self) -> float:
        """Simulated click duration in seconds."""
        return self._properties.mock.click_duration

    @mock_click_duration.setter
    def mock_click_duration(self, value: float) -> None:
        self._properties.mock.click_duration = value

    @property
    def mock_type_duration(self) -> float:
        """Simulated typing duration in seconds."""
        return self._properties.mock.type_duration

    @mock_type_duration.setter
    def mock_type_duration(self, value: float) -> None:
        self._properties.mock.type_duration = value

    @property
    def mock_find_duration(self) -> float:
        """Simulated find duration in seconds."""
        return self._properties.mock.find_duration

    @mock_find_duration.setter
    def mock_find_duration(self, value: float) -> None:
        self._properties.mock.find_duration = value

    # Illustration Settings
    @property
    def illustration_enabled(self) -> bool:
        """Enable action illustrations."""
        return self._properties.illustration.enabled

    @illustration_enabled.setter
    def illustration_enabled(self, value: bool) -> None:
        self._properties.illustration.enabled = value

    @property
    def show_click_illustration(self) -> bool:
        """Illustrate click actions."""
        return self._properties.illustration.show_click

    @show_click_illustration.setter
    def show_click_illustration(self, value: bool) -> None:
        self._properties.illustration.show_click = value

    # Analysis Settings
    @property
    def kmeans_clusters(self) -> int:
        """Number of k-means clusters for color analysis."""
        return self._properties.analysis.kmeans_clusters

    @kmeans_clusters.setter
    def kmeans_clusters(self, value: int) -> None:
        self._properties.analysis.kmeans_clusters = value

    @property
    def color_tolerance(self) -> int:
        """Color matching tolerance."""
        return self._properties.analysis.color_tolerance

    @color_tolerance.setter
    def color_tolerance(self, value: int) -> None:
        self._properties.analysis.color_tolerance = value

    # Dataset Settings
    @property
    def collect_dataset(self) -> bool:
        """Enable dataset collection."""
        return self._properties.dataset.collect

    @collect_dataset.setter
    def collect_dataset(self, value: bool) -> None:
        self._properties.dataset.collect = value

    @property
    def dataset_path(self) -> str:
        """Path to save datasets."""
        return self._properties.dataset.path

    @dataset_path.setter
    def dataset_path(self, value: str) -> None:
        self._properties.dataset.path = value

    # Testing Settings
    @property
    def timeout_multiplier(self) -> float:
        """Multiply timeouts during testing."""
        return self._properties.testing.timeout_multiplier

    @timeout_multiplier.setter
    def timeout_multiplier(self, value: float) -> None:
        self._properties.testing.timeout_multiplier = value

    # Configuration Methods
    def get_properties(self) -> QontinuiProperties:
        """Get the underlying QontinuiProperties object.

        Returns:
            The properties object
        """
        return self._properties

    def update_from_dict(self, config: dict[str, Any]) -> None:
        """Update configuration from dictionary.

        Args:
            config: Configuration dictionary
        """
        # Update properties using Pydantic validation
        for key, value in config.items():
            if hasattr(self._properties, key) and isinstance(value, dict):
                # Update nested config
                nested = getattr(self._properties, key)
                for nested_key, nested_value in value.items():
                    if hasattr(nested, nested_key):
                        setattr(nested, nested_key, nested_value)
            elif hasattr(self, key):
                # Update via property setter for validation
                setattr(self, key, value)

        logger.info("Configuration updated from dictionary")

    def load_from_yaml(self, path: Path) -> None:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file
        """
        self._properties = QontinuiProperties.from_yaml(path)
        logger.info(f"Configuration loaded from {path}")

    def load_from_env_file(self, path: Path) -> None:
        """Load configuration from .env file.

        Args:
            path: Path to .env file
        """
        self._properties = QontinuiProperties.from_env_file(path)
        logger.info(f"Configuration loaded from {path}")

    def save_to_yaml(self, path: Path) -> None:
        """Save current configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        self._properties.to_yaml(path)
        logger.info(f"Configuration saved to {path}")

    def save_to_env_file(self, path: Path) -> None:
        """Save current configuration to .env file.

        Args:
            path: Path to save .env file
        """
        self._properties.to_env_file(path)
        logger.info(f"Configuration saved to {path}")

    def to_dict(self) -> dict[str, Any]:
        """Export configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self._properties.model_dump()

    def validate(self) -> list[str]:
        """Validate current configuration.

        Returns:
            List of validation warnings/errors
        """
        warnings = []

        # Check paths exist
        if not Path(self.image_path).exists() and not self.mock:
            warnings.append(f"Image path does not exist: {self.image_path}")

        if self.save_snapshots and not Path(self.screenshot_path).exists():
            warnings.append(f"Screenshot path does not exist: {self.screenshot_path}")

        if self.collect_dataset and not Path(self.dataset_path).exists():
            warnings.append(f"Dataset path does not exist: {self.dataset_path}")

        # Check for conflicting settings
        if self.headless and self.illustration_enabled:
            warnings.append("Illustration enabled in headless mode (will be ignored)")

        if self.mock and self.save_snapshots:
            warnings.append("Screenshot saving enabled in mock mode (no real screenshots)")

        return warnings


# Global instance accessor
def get_settings() -> FrameworkSettings:
    """Get the global FrameworkSettings instance.

    Returns:
        The singleton FrameworkSettings
    """
    return FrameworkSettings.get_instance()


# Convenience functions for common operations
def enable_mock_mode() -> None:
    """Enable mock mode globally."""
    get_settings().mock = True


def disable_mock_mode() -> None:
    """Disable mock mode globally."""
    get_settings().mock = False


def configure(**kwargs) -> None:
    """Configure framework settings with keyword arguments.

    Args:
        **kwargs: Setting values to update
    """
    settings = get_settings()
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
        else:
            logger.warning(f"Unknown setting: {key}")
