"""Configuration exception.

Exception thrown when configuration is invalid or missing.
"""

from .qontinui_runtime_exception import QontinuiRuntimeException


class ConfigurationError(QontinuiRuntimeException):
    """Exception thrown when configuration is invalid or missing.

    Raised during configuration parsing or validation.
    """

    def __init__(
        self,
        message: str = "Configuration error",
        cause: Exception | None = None,
        config_key: str | None = None,
    ):
        """Initialize configuration exception.

        Args:
            message: Error message
            cause: Underlying exception that caused this error
            config_key: Configuration key that caused the error (if applicable)
        """
        super().__init__(message, cause)
        self.config_key = config_key
