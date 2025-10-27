"""Hardware Abstraction Layer for Qontinui.

This module provides abstraction over different GUI automation backends,
allowing for flexible switching between implementations via environment variables.
"""

from .config import HALConfig
from .container import HALContainer
from .factory import HALFactory
from .initialization import HALInitializationError, initialize_hal, shutdown_hal

__all__ = [
    "HALFactory",
    "HALConfig",
    "HALContainer",
    "initialize_hal",
    "shutdown_hal",
    "HALInitializationError",
]
