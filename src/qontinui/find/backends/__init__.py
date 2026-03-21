"""Detection backends for the CascadeDetector.

Provides a unified detection fallback chain that wraps existing detection
mechanisms behind a common DetectionBackend interface.
"""

from .base import DetectionBackend, DetectionResult
from .cascade import CascadeDetector, MatchSettings
from .invariant_match_backend import InvariantMatchBackend

# OmniParser modules use lazy imports to avoid circular dependencies
# (omniparser_detector -> omniparser_config -> __init__ -> omniparser_backend -> omniparser_detector).
# Import them directly when needed:
#   from qontinui.find.backends.omniparser_backend import OmniParserBackend
#   from qontinui.find.backends.omniparser_config import OmniParserSettings
#   from qontinui.find.backends.omniparser_service_backend import OmniParserServiceBackend

__all__ = [
    "CascadeDetector",
    "DetectionBackend",
    "DetectionResult",
    "InvariantMatchBackend",
    "MatchSettings",
]
