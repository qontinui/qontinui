"""Log levels - ported from Qontinui framework.

Enumeration of log levels for the logging system.
"""

from enum import Enum


class LogLevel(Enum):
    """Log levels for the logging system.

    Port of LogLevel from Qontinui framework enum.
    """

    TRACE = "TRACE"
    """Trace level - most detailed logging."""

    DEBUG = "DEBUG"
    """Debug level - debugging information."""

    INFO = "INFO"
    """Info level - informational messages."""

    WARN = "WARN"
    """Warning level - warning messages."""

    ERROR = "ERROR"
    """Error level - error messages."""

    FATAL = "FATAL"
    """Fatal level - fatal error messages."""

    OFF = "OFF"
    """Off - no logging."""

    def to_python_level(self) -> int:
        """Convert to Python logging level.

        Returns:
            Python logging level integer
        """
        mapping = {
            LogLevel.TRACE: 5,  # Below DEBUG
            LogLevel.DEBUG: 10,  # logging.DEBUG
            LogLevel.INFO: 20,  # logging.INFO
            LogLevel.WARN: 30,  # logging.WARNING
            LogLevel.ERROR: 40,  # logging.ERROR
            LogLevel.FATAL: 50,  # logging.CRITICAL
            LogLevel.OFF: 100,  # Above all levels
        }
        return mapping.get(self, 20)
