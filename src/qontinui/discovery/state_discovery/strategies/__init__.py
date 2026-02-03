"""State discovery strategies.

This module contains the different strategy implementations for state discovery:
- LegacyStrategy: ID-based co-occurrence analysis (original implementation)
- FingerprintStrategy: Enhanced discovery using element fingerprints
"""

from .fingerprint import FingerprintStrategy
from .legacy import LegacyStrategy

__all__ = [
    "LegacyStrategy",
    "FingerprintStrategy",
]
