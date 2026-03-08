"""State discovery strategies.

This module contains the strategy implementations for state discovery:
- FingerprintStrategy: Enhanced discovery using element fingerprints (with ID fallback)
"""

from .fingerprint import FingerprintStrategy

__all__ = [
    "FingerprintStrategy",
]
