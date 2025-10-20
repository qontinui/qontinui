"""
Screenshot Cache Module

Provides thread-safe screenshot caching to enable parallel template matching
on the same screen capture. This avoids redundant screen captures and
significantly improves performance when searching for multiple patterns.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CachedScreenshot:
    """Container for cached screenshot with timestamp"""

    image: np.ndarray
    timestamp: float


class ScreenshotCache:
    """
    Thread-safe screenshot cache with TTL.

    Caches screenshots for a configurable TTL (default 100ms) to allow
    multiple parallel template matches on the same screen capture. This
    eliminates redundant screen captures which typically take 50-200ms each.

    Example:
        # Without cache (wasteful):
        for pattern in 10_patterns:
            screenshot = capture()  # 10 × 50ms = 500ms wasted
            match = find(screenshot, pattern)

        # With cache (optimal):
        cache = ScreenshotCache()
        for pattern in 10_patterns:
            screenshot = await cache.get_screenshot(capture)  # 1 × 50ms
            match = find(screenshot, pattern)

    Thread Safety:
        Uses asyncio.Lock for safe concurrent access from multiple tasks.
    """

    def __init__(self, ttl_seconds: float = 0.1):
        """
        Initialize screenshot cache.

        Args:
            ttl_seconds: Time-to-live for cached screenshots in seconds.
                        Default 100ms is optimal for parallel searches.
                        Longer TTL risks stale screenshots, shorter reduces benefit.
        """
        self.ttl = ttl_seconds
        self._cache: CachedScreenshot | None = None
        self._lock = asyncio.Lock()

    async def get_screenshot(self, capture_func: Callable[[], Any]) -> np.ndarray:
        """
        Get cached screenshot or capture new one.

        If a screenshot exists in cache and is within TTL, returns it.
        Otherwise captures new screenshot, caches it, and returns it.

        Args:
            capture_func: Function that captures screenshot (e.g., Screen.capture).
                         Should return numpy array. Will be called in thread pool
                         to avoid blocking event loop.

        Returns:
            Screenshot as numpy array (H×W×C)

        Example:
            from qontinui.screen import Screen

            cache = ScreenshotCache()
            screenshot = await cache.get_screenshot(Screen.capture)
        """
        async with self._lock:
            now = time.time()

            # Check if cache valid
            if self._cache and (now - self._cache.timestamp) < self.ttl:
                return self._cache.image

            # Capture new screenshot in thread pool (I/O bound)
            screenshot = await asyncio.to_thread(capture_func)

            # Cache it
            self._cache = CachedScreenshot(image=screenshot, timestamp=now)

            return screenshot

    def clear(self):
        """
        Clear the cache.

        Forces next get_screenshot() to capture fresh screenshot
        regardless of TTL.
        """
        self._cache = None

    @property
    def is_valid(self) -> bool:
        """
        Check if cache currently contains valid screenshot.

        Returns:
            True if cache has screenshot within TTL
        """
        if not self._cache:
            return False

        now = time.time()
        return (now - self._cache.timestamp) < self.ttl

    @property
    def age_ms(self) -> float | None:
        """
        Get age of cached screenshot in milliseconds.

        Returns:
            Age in milliseconds, or None if no cached screenshot
        """
        if not self._cache:
            return None

        now = time.time()
        return (now - self._cache.timestamp) * 1000
