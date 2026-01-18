"""Action caching for deterministic replay.

This module provides caching of successful action targets to enable
fast, deterministic replay of automation workflows.

Key Features:
- Cache successful element locations by state + pattern
- Validate cached entries against current screen
- Automatic invalidation on validation failure
- Configurable age and size limits

Example:
    >>> from qontinui.cache import ActionCache, get_action_cache
    >>>
    >>> # Use global cache
    >>> cache = get_action_cache()
    >>>
    >>> # Build cache key
    >>> key = cache.build_key(pattern, state_id="login_page", action_type="click")
    >>>
    >>> # Try cache first
    >>> result = cache.try_get(key, screenshot, pattern)
    >>> if result.hit:
    >>>     # Use cached coordinates
    >>>     x, y = result.entry.coordinates.x, result.entry.coordinates.y
    >>> else:
    >>>     # Do normal pattern matching
    >>>     matches = matcher.find_matches(screenshot, pattern)
    >>>     if matches:
    >>>         # Store in cache for next time
    >>>         cache.store(key, (match.x, match.y), match.region, match.similarity, pattern)
"""

from .action_cache import ActionCache, get_action_cache, set_action_cache
from .cache_storage import CacheStorage
from .cache_types import (
    CachedCoordinates,
    CacheEntry,
    CacheResult,
    CacheStats,
    ValidationPattern,
)

__all__ = [
    # Main cache class
    "ActionCache",
    "get_action_cache",
    "set_action_cache",
    # Storage
    "CacheStorage",
    # Types
    "CachedCoordinates",
    "CacheEntry",
    "CacheResult",
    "CacheStats",
    "ValidationPattern",
]
