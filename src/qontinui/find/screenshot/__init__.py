"""Screenshot capture abstraction for finding operations.

Provides clean interface for capturing screenshots with optional caching.
"""

from .cached_provider import CachedScreenshotProvider
from .pure_actions_provider import PureActionsScreenshotProvider
from .screenshot_provider import ScreenshotProvider

__all__ = [
    "ScreenshotProvider",
    "PureActionsScreenshotProvider",
    "CachedScreenshotProvider",
]
