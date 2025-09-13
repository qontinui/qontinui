"""Hardware Abstraction Layer for Qontinui.

This module provides abstraction over different GUI automation backends,
allowing for flexible switching between implementations via environment variables.
"""

from .factory import HALFactory
from .config import HALConfig

__all__ = ['HALFactory', 'HALConfig']