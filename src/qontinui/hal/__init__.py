"""Hardware Abstraction Layer for Qontinui.

This module provides abstraction over different GUI automation backends,
allowing for flexible switching between implementations via environment variables.
"""

from .config import HALConfig
from .factory import HALFactory

__all__ = ["HALFactory", "HALConfig"]
