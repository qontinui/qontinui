"""Framework settings - ported from Qontinui framework.

Global configuration settings using singleton pattern with Pydantic validation.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from .qontinui_properties import (
    QontinuiProperties,
)

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
    - Themed configuration groups for organization
    - Environment variable support

    Configuration Groups:
    - core: Essential framework settings (image paths, mock mode, etc.)
    - mouse: Mouse action timing and movement configuration
    - mock: Simulated execution timings for testing
    - screenshot: Screen capture and history settings
    - illustration: Visual feedback and annotation settings
    - analysis: Color profiling and image processing
    - recording: Screen recording configuration
    - dataset: AI training data generation
    - testing: Test execution configuration
    - monitor: Monitor configuration settings
    - dpi: DPI and scaling configuration
    - capture: Screen capture provider configuration
    - sikuli: SikuliX integration settings
    - startup: Application startup configuration
    - automation: Automation failure handling
    - autoscaling: Automatic pattern scaling
    - logging: Comprehensive logging configuration
    - highlight: Visual highlighting configuration
    - console: Console action reporting
    - image_debug: Image debugging configuration
    - gui_access: GUI access verification

    Access settings via themed groups:
        settings = FrameworkSettings.get_instance()

        # Mouse settings
        settings.mouse.move_delay = 0.5
        settings.mouse.click_delay = 0.1

        # Core settings
        settings.core.mock = True
        settings.core.headless = False

        # Screenshot settings
        settings.screenshot.save_snapshots = True
        settings.screenshot.path = "screenshots/"

        # Logging settings
        settings.logging.global_level = "DEBUG"
        settings.logging.actions_level = "INFO"

    Usage:
        # Get singleton instance
        settings = FrameworkSettings.get_instance()

        # Access themed settings
        delay = settings.mouse.move_delay
        mock = settings.core.mock

        # Update settings with validation
        settings.mouse.move_delay = 0.3
        settings.core.mock = True

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
            cls._instance._initialized = False  # type: ignore[has-type]
        return cls._instance

    def __init__(self) -> None:
        """Initialize settings if not already done."""
        if not self._initialized:  # type: ignore[has-type]
            props = QontinuiProperties()

            # Initialize themed configuration groups
            self.core = props.core
            self.mouse = props.mouse
            self.mock = props.mock
            self.screenshot = props.screenshot
            self.illustration = props.illustration
            self.analysis = props.analysis
            self.recording = props.recording
            self.dataset = props.dataset
            self.testing = props.testing
            self.monitor = props.monitor
            self.dpi = props.dpi
            self.capture = props.capture
            self.sikuli = props.sikuli
            self.startup = props.startup
            self.automation = props.automation
            self.autoscaling = props.autoscaling
            self.logging = props.logging
            self.highlight = props.highlight
            self.console = props.console
            self.image_debug = props.image_debug
            self.gui_access = props.gui_access

            # Keep properties object for serialization
            self._properties = props
            self._initialized = True
            logger.info(
                "FrameworkSettings initialized with themed configuration groups"
            )

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
            config: Configuration dictionary with nested structure
                   matching themed configuration groups
        """
        # Update properties using Pydantic validation
        for key, value in config.items():
            if hasattr(self, key) and isinstance(value, dict):
                # Update nested config group
                config_group = getattr(self, key)
                for nested_key, nested_value in value.items():
                    if hasattr(config_group, nested_key):
                        setattr(config_group, nested_key, nested_value)

        # Update properties object
        for key in vars(self).keys():
            if key.startswith("_"):
                continue
            if hasattr(self._properties, key):
                setattr(self._properties, key, getattr(self, key))

        logger.info("Configuration updated from dictionary")

    def load_from_yaml(self, path: Path) -> None:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file
        """
        self._properties = QontinuiProperties.from_yaml(path)

        # Update all configuration groups
        self.core = self._properties.core
        self.mouse = self._properties.mouse
        self.mock = self._properties.mock
        self.screenshot = self._properties.screenshot
        self.illustration = self._properties.illustration
        self.analysis = self._properties.analysis
        self.recording = self._properties.recording
        self.dataset = self._properties.dataset
        self.testing = self._properties.testing
        self.monitor = self._properties.monitor
        self.dpi = self._properties.dpi
        self.capture = self._properties.capture
        self.sikuli = self._properties.sikuli
        self.startup = self._properties.startup
        self.automation = self._properties.automation
        self.autoscaling = self._properties.autoscaling
        self.logging = self._properties.logging
        self.highlight = self._properties.highlight
        self.console = self._properties.console
        self.image_debug = self._properties.image_debug
        self.gui_access = self._properties.gui_access

        logger.info(f"Configuration loaded from {path}")

    def load_from_env_file(self, path: Path) -> None:
        """Load configuration from .env file.

        Args:
            path: Path to .env file
        """
        self._properties = QontinuiProperties.from_env_file(path)

        # Update all configuration groups
        self.core = self._properties.core
        self.mouse = self._properties.mouse
        self.mock = self._properties.mock
        self.screenshot = self._properties.screenshot
        self.illustration = self._properties.illustration
        self.analysis = self._properties.analysis
        self.recording = self._properties.recording
        self.dataset = self._properties.dataset
        self.testing = self._properties.testing
        self.monitor = self._properties.monitor
        self.dpi = self._properties.dpi
        self.capture = self._properties.capture
        self.sikuli = self._properties.sikuli
        self.startup = self._properties.startup
        self.automation = self._properties.automation
        self.autoscaling = self._properties.autoscaling
        self.logging = self._properties.logging
        self.highlight = self._properties.highlight
        self.console = self._properties.console
        self.image_debug = self._properties.image_debug
        self.gui_access = self._properties.gui_access

        logger.info(f"Configuration loaded from {path}")

    def save_to_yaml(self, path: Path) -> None:
        """Save current configuration to YAML file.

        Args:
            path: Path to save YAML file
        """
        # Sync all config groups back to properties
        for key in vars(self).keys():
            if key.startswith("_"):
                continue
            if hasattr(self._properties, key):
                setattr(self._properties, key, getattr(self, key))

        self._properties.to_yaml(path)
        logger.info(f"Configuration saved to {path}")

    def save_to_env_file(self, path: Path) -> None:
        """Save current configuration to .env file.

        Args:
            path: Path to save .env file
        """
        # Sync all config groups back to properties
        for key in vars(self).keys():
            if key.startswith("_"):
                continue
            if hasattr(self._properties, key):
                setattr(self._properties, key, getattr(self, key))

        self._properties.to_env_file(path)
        logger.info(f"Configuration saved to {path}")

    def to_dict(self) -> dict[str, Any]:
        """Export configuration as dictionary.

        Returns:
            Configuration dictionary with all themed groups
        """
        # Sync all config groups back to properties
        for key in vars(self).keys():
            if key.startswith("_"):
                continue
            if hasattr(self._properties, key):
                setattr(self._properties, key, getattr(self, key))

        result: dict[str, Any] = self._properties.model_dump()
        return result

    def validate(self) -> list[str]:
        """Validate current configuration.

        Returns:
            List of validation warnings/errors
        """
        warnings = []

        # Check paths exist
        if not Path(self.core.image_path).exists() and not self.core.mock:
            warnings.append(f"Image path does not exist: {self.core.image_path}")

        if self.screenshot.save_snapshots and not Path(self.screenshot.path).exists():
            warnings.append(f"Screenshot path does not exist: {self.screenshot.path}")

        if self.dataset.collect and not Path(self.dataset.path).exists():
            warnings.append(f"Dataset path does not exist: {self.dataset.path}")

        # Check for conflicting settings
        if self.core.headless and self.illustration.enabled:
            warnings.append("Illustration enabled in headless mode (will be ignored)")

        if self.core.mock and self.screenshot.save_snapshots:
            warnings.append(
                "Screenshot saving enabled in mock mode (no real screenshots)"
            )

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
    get_settings().core.mock = True


def disable_mock_mode() -> None:
    """Disable mock mode globally."""
    get_settings().core.mock = False


def configure(**kwargs) -> None:
    """Configure framework settings with keyword arguments.

    Args:
        **kwargs: Nested configuration groups as dictionaries
                 Example: configure(mouse={'move_delay': 0.3}, core={'mock': True})
    """
    settings = get_settings()
    for key, value in kwargs.items():
        if hasattr(settings, key):
            if isinstance(value, dict):
                # Update nested config group
                config_group = getattr(settings, key)
                for nested_key, nested_value in value.items():
                    if hasattr(config_group, nested_key):
                        setattr(config_group, nested_key, nested_value)
                    else:
                        logger.warning(f"Unknown setting: {key}.{nested_key}")
            else:
                # Direct attribute update (shouldn't happen with themed groups)
                setattr(settings, key, value)
        else:
            logger.warning(f"Unknown configuration group: {key}")
