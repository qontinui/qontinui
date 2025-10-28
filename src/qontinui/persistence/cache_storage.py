"""In-memory cache storage with TTL support.

This module provides a simple in-memory cache with time-based expiration
and LRU eviction policies.
"""

from datetime import datetime
from typing import Any

from ..logging import get_logger

logger = get_logger(__name__)


class CacheStorage:
    """In-memory cache with TTL and LRU eviction.

    Features:
        - Time-based expiration (TTL)
        - Size limits with LRU eviction
        - Per-key TTL configuration
        - Automatic cleanup of expired entries
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0) -> None:
        """Initialize cache storage.

        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default TTL in seconds (default: 5 minutes)
        """
        self._cache: dict[str, Any] = {}
        self._timestamps: dict[str, float] = {}
        self._ttls: dict[str, float] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl

        logger.debug(
            "cache_storage_initialized",
            max_size=max_size,
            default_ttl=default_ttl,
        )

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set cache value with optional TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL in seconds (uses default_ttl if not provided)
        """
        # Enforce size limit using LRU eviction
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_oldest()

        self._cache[key] = value
        self._timestamps[key] = datetime.now().timestamp()
        self._ttls[key] = ttl if ttl is not None else self.default_ttl

        logger.debug("cache_set", key=key, ttl=self._ttls[key])

    def get(self, key: str, default: Any = None) -> Any:
        """Get cache value.

        Args:
            key: Cache key
            default: Default value if not found or expired

        Returns:
            Cached value or default
        """
        if key not in self._cache:
            return default

        # Check expiration
        age = datetime.now().timestamp() - self._timestamps[key]
        if age > self._ttls[key]:
            logger.debug("cache_expired", key=key, age=age)
            self._remove_key(key)
            return default

        logger.debug("cache_hit", key=key)
        return self._cache[key]

    def delete(self, key: str) -> bool:
        """Delete cache entry.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        if key in self._cache:
            self._remove_key(key)
            logger.debug("cache_deleted", key=key)
            return True

        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        self._timestamps.clear()
        self._ttls.clear()

        logger.debug("cache_cleared", count=count)

    def size(self) -> int:
        """Get current cache size.

        Returns:
            Number of entries in cache
        """
        return len(self._cache)

    def has_key(self, key: str) -> bool:
        """Check if key exists and is not expired.

        Args:
            key: Cache key

        Returns:
            True if key exists and is valid
        """
        if key not in self._cache:
            return False

        # Check expiration
        age = datetime.now().timestamp() - self._timestamps[key]
        if age > self._ttls[key]:
            self._remove_key(key)
            return False

        return True

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "default_ttl": self.default_ttl,
            "keys": list(self._cache.keys()),
        }

    def _evict_oldest(self) -> None:
        """Evict oldest cache entry using LRU policy."""
        if not self._timestamps:
            return

        oldest_key = min(self._timestamps, key=lambda k: self._timestamps[k])
        logger.debug("cache_evicted_lru", key=oldest_key)
        self._remove_key(oldest_key)

    def _remove_key(self, key: str) -> None:
        """Remove key from all tracking dictionaries.

        Args:
            key: Key to remove
        """
        del self._cache[key]
        del self._timestamps[key]
        del self._ttls[key]
