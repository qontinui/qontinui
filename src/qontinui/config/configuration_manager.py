"""Configuration manager - centralized configuration management.

Provides a unified interface for all configuration needs.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

from .execution_environment import ExecutionEnvironment, ExecutionMode, get_environment
from .framework_settings import FrameworkSettings, get_settings
from .qontinui_properties import QontinuiProperties

logger = logging.getLogger(__name__)


class ConfigurationManager:
    """Centralized configuration management for Qontinui.

    This manager provides:
    - Unified configuration interface
    - Environment-based configuration
    - Configuration file loading (YAML, JSON, .env)
    - Configuration validation
    - Configuration profiles
    - Dynamic reconfiguration

    Usage:
        # Get manager instance
        config = ConfigurationManager.get_instance()

        # Load configuration
        config.load_profile('production')

        # Access settings
        settings = config.get_settings()

        # Update specific values
        config.update(mock=True)

        # Validate configuration
        warnings = config.validate()
    """

    _instance: Optional["ConfigurationManager"] = None

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize configuration manager."""
        if not self._initialized:
            self.settings = get_settings()
            self.environment = get_environment()
            self._profiles: dict[str, dict[str, Any]] = self._load_default_profiles()
            self._config_sources: list[str] = []
            self._initialized = True

            # Auto-load configuration
            self._auto_load_configuration()

            logger.info(f"ConfigurationManager initialized in {self.environment.mode.value} mode")

    @classmethod
    def get_instance(cls) -> "ConfigurationManager":
        """Get the singleton instance.

        Returns:
            ConfigurationManager instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_default_profiles(self) -> dict[str, dict[str, Any]]:
        """Load default configuration profiles.

        Returns:
            Dictionary of profile configurations
        """
        return {
            "development": {
                "mock": False,
                "save_snapshots": True,
                "illustration_enabled": True,
                "collect_dataset": True,
                "timeout_multiplier": 1.0,
            },
            "testing": {
                "mock": True,
                "save_snapshots": False,
                "illustration_enabled": False,
                "collect_dataset": False,
                "timeout_multiplier": 2.0,
                "max_history": 10,
            },
            "production": {
                "mock": False,
                "save_snapshots": False,
                "illustration_enabled": False,
                "collect_dataset": False,
                "timeout_multiplier": 1.0,
                "headless": False,
            },
            "ci_cd": {
                "mock": True,
                "headless": True,
                "save_snapshots": False,
                "illustration_enabled": False,
                "collect_dataset": False,
                "timeout_multiplier": 3.0,
            },
            "performance": {
                "mouse_move_delay": 0.1,
                "pause_before_mouse_down": 0.0,
                "pause_after_mouse_down": 0.0,
                "pause_before_mouse_up": 0.0,
                "pause_after_mouse_up": 0.0,
                "save_snapshots": False,
                "illustration_enabled": False,
            },
            "debug": {
                "save_snapshots": True,
                "illustration_enabled": True,
                "max_history": 100,
                "collect_dataset": True,
                "screenshot_on_error": True,
            },
        }

    def _auto_load_configuration(self) -> None:
        """Automatically load configuration from various sources."""
        # 1. Load from environment-specific config file
        env_config = Path(f"config.{self.environment.mode.value}.yaml")
        if env_config.exists():
            self.load_from_file(env_config)
            logger.info(f"Loaded environment config: {env_config}")

        # 2. Load from default config file
        elif Path("config.yaml").exists():
            self.load_from_file(Path("config.yaml"))
            logger.info("Loaded default config.yaml")

        elif Path("qontinui.yaml").exists():
            self.load_from_file(Path("qontinui.yaml"))
            logger.info("Loaded qontinui.yaml")

        # 3. Load from .env file
        if Path(".env").exists():
            self._load_env_file(Path(".env"))
            logger.info("Loaded .env file")

        # 4. Apply profile based on execution mode
        profile_map = {
            ExecutionMode.DEVELOPMENT: "development",
            ExecutionMode.TESTING: "testing",
            ExecutionMode.PRODUCTION: "production",
            ExecutionMode.CI_CD: "ci_cd",
            ExecutionMode.STAGING: "production",
        }

        profile = profile_map.get(self.environment.mode)
        if profile:
            self.load_profile(profile)
            logger.info(f"Applied {profile} profile")

    def load_from_file(self, path: Path) -> None:
        """Load configuration from file.

        Supports YAML, JSON, and .env files.

        Args:
            path: Path to configuration file
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        suffix = path.suffix.lower()

        if suffix in [".yaml", ".yml"]:
            self.settings.load_from_yaml(path)
        elif suffix == ".json":
            import json

            with open(path) as f:
                config = json.load(f)
            self.settings.update_from_dict(config)
        elif suffix == ".env":
            self._load_env_file(path)
        else:
            raise ValueError(f"Unsupported configuration file type: {suffix}")

        self._config_sources.append(str(path))
        logger.info(f"Loaded configuration from {path}")

    def _load_env_file(self, path: Path) -> None:
        """Load configuration from .env file.

        Args:
            path: Path to .env file
        """
        try:
            from dotenv import load_dotenv

            load_dotenv(path)

            # Map specific environment variables
            env_mappings = {
                "QONTINUI_MOCK": ("mock", lambda x: x.lower() == "true"),
                "QONTINUI_HEADLESS": ("headless", lambda x: x.lower() == "true"),
                "QONTINUI_IMAGE_PATH": ("image_path", str),
                "QONTINUI_SCREENSHOT_PATH": ("screenshot_path", str),
                "QONTINUI_DATASET_PATH": ("dataset_path", str),
                "QONTINUI_TIMEOUT_MULTIPLIER": ("timeout_multiplier", float),
            }

            for env_var, (setting, converter) in env_mappings.items():
                value = os.environ.get(env_var)
                if value is not None:
                    try:
                        converted = converter(value)
                        setattr(self.settings, setting, converted)
                    except Exception as e:
                        logger.warning(f"Failed to set {setting} from {env_var}: {e}")

            self._config_sources.append(f"{path} (env)")

        except ImportError:
            logger.warning("python-dotenv not installed, skipping .env file")

    def load_profile(self, profile_name: str) -> None:
        """Load a configuration profile.

        Args:
            profile_name: Name of the profile to load
        """
        if profile_name not in self._profiles:
            raise ValueError(f"Unknown profile: {profile_name}")

        profile = self._profiles[profile_name]
        self.settings.update_from_dict(profile)

        self._config_sources.append(f"profile:{profile_name}")
        logger.info(f"Applied configuration profile: {profile_name}")

    def add_profile(self, name: str, config: dict[str, Any]) -> None:
        """Add a custom configuration profile.

        Args:
            name: Profile name
            config: Profile configuration
        """
        self._profiles[name] = config
        logger.info(f"Added configuration profile: {name}")

    def update(self, **kwargs) -> None:
        """Update configuration values.

        Args:
            **kwargs: Configuration values to update
        """
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")

    def get_settings(self) -> FrameworkSettings:
        """Get framework settings.

        Returns:
            FrameworkSettings instance
        """
        return self.settings

    def get_properties(self) -> QontinuiProperties:
        """Get Qontinui properties.

        Returns:
            QontinuiProperties instance
        """
        return self.settings.get_properties()

    def get_environment(self) -> ExecutionEnvironment:
        """Get execution environment.

        Returns:
            ExecutionEnvironment instance
        """
        return self.environment

    def validate(self) -> list[str]:
        """Validate current configuration.

        Returns:
            List of validation warnings
        """
        warnings = []

        # Validate settings
        warnings.extend(self.settings.validate())

        # Environment-specific validation
        if self.environment.is_headless() and self.settings.illustration_enabled:
            warnings.append("Illustration enabled in headless environment")

        if not self.environment.supports_screenshots() and self.settings.save_snapshots:
            warnings.append("Screenshot saving enabled but environment doesn't support it")

        # Mode-specific validation
        if self.environment.mode == ExecutionMode.PRODUCTION:
            if self.settings.mock:
                warnings.append("Mock mode enabled in production")
            if self.settings.collect_dataset:
                warnings.append("Dataset collection enabled in production")

        return warnings

    def save_current(self, path: Path, format: str = "yaml") -> None:
        """Save current configuration to file.

        Args:
            path: Path to save configuration
            format: File format ('yaml', 'json', 'env')
        """
        if format == "yaml":
            self.settings.save_to_yaml(path)
        elif format == "json":
            import json

            with open(path, "w") as f:
                json.dump(self.settings.to_dict(), f, indent=2)
        elif format == "env":
            self.settings.save_to_env_file(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved configuration to {path}")

    def get_info(self) -> dict[str, Any]:
        """Get configuration information.

        Returns:
            Configuration information dictionary
        """
        return {
            "mode": self.environment.mode.value,
            "environment": self.environment.get_info(),
            "config_sources": self._config_sources,
            "available_profiles": list(self._profiles.keys()),
            "validation_warnings": self.validate(),
            "settings_summary": {
                "mock": self.settings.mock,
                "headless": self.settings.headless,
                "save_snapshots": self.settings.save_snapshots,
                "collect_dataset": self.settings.collect_dataset,
            },
        }

    def reset(self) -> None:
        """Reset configuration to defaults."""
        FrameworkSettings.reset()
        self.settings = get_settings()
        self._config_sources = []
        logger.info("Configuration reset to defaults")


# Global accessor
def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager.

    Returns:
        ConfigurationManager instance
    """
    return ConfigurationManager.get_instance()
