"""Configuration exceptions.

This module contains exceptions for configuration validation,
missing configuration, and invalid settings.
"""

from .base_exceptions import QontinuiException


class ConfigurationException(QontinuiException):
    """Base exception for configuration errors."""

    pass


class InvalidConfigurationException(ConfigurationException):
    """Raised when configuration is invalid."""

    def __init__(self, config_key: str, reason: str, **kwargs) -> None:
        """Initialize with config details."""
        super().__init__(
            f"Invalid configuration for '{config_key}': {reason}",
            error_code="INVALID_CONFIG",
            context={"config_key": config_key, "reason": reason, **kwargs},
        )


class MissingConfigurationException(ConfigurationException):
    """Raised when required configuration is missing."""

    def __init__(self, config_key: str, **kwargs) -> None:
        """Initialize with missing config key."""
        super().__init__(
            f"Required configuration '{config_key}' is missing",
            error_code="MISSING_CONFIG",
            context={"config_key": config_key, **kwargs},
        )


class ConfigurationError(QontinuiException):
    """Raised when configuration is invalid."""

    def __init__(self, config_key: str, reason: str, **kwargs) -> None:
        """Initialize with configuration details."""
        super().__init__(
            f"Configuration error for '{config_key}': {reason}",
            error_code="CONFIGURATION_ERROR",
            context={"config_key": config_key, "reason": reason, **kwargs},
        )
