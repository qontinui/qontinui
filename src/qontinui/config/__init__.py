"""Configuration package - ported from Qontinui framework with Python improvements.

Provides comprehensive configuration management using Pydantic for validation
and type safety, with support for multiple configuration sources.

Key improvements over Java/Spring approach:
- Pydantic for type-safe configuration with validation
- Native Python property decorators
- Environment variable support with python-decouple
- YAML/JSON/TOML configuration file support
- Configuration profiles for different environments
- Automatic environment detection and adjustment
- Singleton pattern with global access

Usage:
    # Simple access to settings
    from qontinui.config import get_settings, enable_mock_mode

    settings = get_settings()
    settings.mock = True

    # Using configuration manager
    from qontinui.config import get_config_manager

    config = get_config_manager()
    config.load_profile('production')

    # Environment-aware configuration
    from qontinui.config import get_environment

    env = get_environment()
    if env.is_headless():
        # Adjust for headless environment
        pass
"""

from .configuration_manager import ConfigurationManager, get_config_manager
from .execution_environment import (
    DisplayServer,
    ExecutionEnvironment,
    ExecutionMode,
    Platform,
    SystemInfo,
    get_environment,
)
from .framework_settings import (
    FrameworkSettings,
    configure,
    disable_mock_mode,
    enable_mock_mode,
    get_settings,
)
from .qontinui_properties import (
    AnalysisConfig,
    CoreConfig,
    DatasetConfig,
    IllustrationConfig,
    MockConfig,
    MouseConfig,
    QontinuiProperties,
    RecordingConfig,
    ScreenshotConfig,
    TestingConfig,
)

__all__ = [
    # Properties
    "QontinuiProperties",
    "CoreConfig",
    "MouseConfig",
    "MockConfig",
    "ScreenshotConfig",
    "IllustrationConfig",
    "AnalysisConfig",
    "RecordingConfig",
    "DatasetConfig",
    "TestingConfig",
    # Settings
    "FrameworkSettings",
    "get_settings",
    "enable_mock_mode",
    "disable_mock_mode",
    "configure",
    # Environment
    "ExecutionEnvironment",
    "ExecutionMode",
    "Platform",
    "DisplayServer",
    "SystemInfo",
    "get_environment",
    # Manager
    "ConfigurationManager",
    "get_config_manager",
]
