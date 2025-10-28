"""Cached screenshot provider for performance optimization.

Provides time-based caching to avoid redundant screenshot captures
during rapid successive find operations.
"""

import time
from dataclasses import dataclass

from PIL import Image

from ...model.element import Region
from .screenshot_provider import ScreenshotProvider


@dataclass
class CacheEntry:
    """Cache entry for screenshot data.

    Attributes:
        image: Cached PIL Image.
        timestamp: When the screenshot was captured.
        region: Region that was captured (None for full screen).
    """

    image: Image.Image
    timestamp: float
    region: Region | None


class CachedScreenshotProvider(ScreenshotProvider):
    """Screenshot provider with time-based caching.

    Caches screenshots for a short time-to-live (TTL) to improve performance
    when multiple find operations occur in rapid succession. This is especially
    useful for finding multiple patterns on the same screen state.

    Cache invalidation is based on:
    - TTL expiration
    - Region mismatch (full screen vs. specific region)
    """

    def __init__(
        self,
        provider: ScreenshotProvider,
        ttl_seconds: float = 0.1
    ) -> None:
        """Initialize cached provider.

        Args:
            provider: Underlying screenshot provider to wrap.
            ttl_seconds: Time-to-live for cached screenshots in seconds.
                        Default 0.1s (100ms) balances performance and freshness.
        """
        self.provider = provider
        self.ttl = ttl_seconds
        self._cache: CacheEntry | None = None

    def capture(self, region: Region | None = None) -> Image.Image:
        """Capture screenshot with caching.

        Returns cached screenshot if:
        - Cache exists and hasn't expired (within TTL)
        - Requested region matches cached region

        Otherwise captures a new screenshot and updates cache.

        Args:
            region: Optional region to capture. If None, captures entire screen.

        Returns:
            PIL Image of the captured screenshot.

        Raises:
            RuntimeError: If screenshot capture fails.
        """
        current_time = time.time()

        # Check if cache is valid
        if self._is_cache_valid(current_time, region):
            # Cache hit - return cached image
            # Type assertion: _is_cache_valid ensures _cache is not None
            assert self._cache is not None
            return self._cache.image

        # Cache miss - capture new screenshot
        image = self.provider.capture(region)

        # Update cache
        self._cache = CacheEntry(
            image=image,
            timestamp=current_time,
            region=region
        )

        return image

    def _is_cache_valid(
        self,
        current_time: float,
        region: Region | None
    ) -> bool:
        """Check if cached screenshot is still valid.

        Args:
            current_time: Current timestamp.
            region: Requested region.

        Returns:
            True if cache exists, hasn't expired, and matches region.
        """
        if self._cache is None:
            return False

        # Check TTL expiration
        if current_time - self._cache.timestamp > self.ttl:
            return False

        # Check region match
        # Both None = match (full screen)
        # One None, one Region = mismatch
        # Both Region = check equality
        if region is None and self._cache.region is None:
            return True

        if region is None or self._cache.region is None:
            return False

        # Compare regions
        return (
            region.x == self._cache.region.x
            and region.y == self._cache.region.y
            and region.width == self._cache.region.width
            and region.height == self._cache.region.height
        )

    def clear_cache(self) -> None:
        """Clear the cached screenshot.

        Useful for forcing a fresh screenshot capture on the next call.
        """
        self._cache = None

    def get_cache_age(self) -> float | None:
        """Get age of cached screenshot in seconds.

        Returns:
            Age of cache in seconds, or None if no cache exists.
        """
        if self._cache is None:
            return None
        return time.time() - self._cache.timestamp
