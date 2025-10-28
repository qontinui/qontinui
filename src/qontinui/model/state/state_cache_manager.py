"""StateCacheManager - LRU cache for frequently accessed states.

Manages caching of state names using LRU eviction policy.
"""

import logging
import threading

logger = logging.getLogger(__name__)


class StateCacheManager:
    """Manages LRU cache for state access optimization.

    Single responsibility: Optimize state access through LRU caching.
    """

    def __init__(self, max_cache_size: int = 100) -> None:
        """Initialize the cache manager.

        Args:
            max_cache_size: Maximum number of states to cache
        """
        self._max_cache_size = max_cache_size
        self._cache_order: list[str] = []  # LRU cache order
        self._lock = threading.RLock()
        logger.debug(f"Cache manager initialized with size {max_cache_size}")

    def access(self, name: str, is_active: bool = False) -> None:
        """Record access to a state and update cache.

        Args:
            name: State name that was accessed
            is_active: Whether the state is currently active
        """
        with self._lock:
            # Move to end (most recently used)
            if name in self._cache_order:
                self._cache_order.remove(name)
            self._cache_order.append(name)

            # Evict least recently used if cache is full
            self._evict_if_needed(active_states={name} if is_active else set())

    def remove(self, name: str) -> None:
        """Remove a state from the cache.

        Args:
            name: State name to remove
        """
        with self._lock:
            if name in self._cache_order:
                self._cache_order.remove(name)
                logger.debug(f"Removed '{name}' from cache")

    def get_cached_states(self) -> list[str]:
        """Get list of cached state names in LRU order.

        Returns:
            List of cached state names (least to most recently used)
        """
        with self._lock:
            return self._cache_order.copy()

    def is_cached(self, name: str) -> bool:
        """Check if a state is in the cache.

        Args:
            name: State name to check

        Returns:
            True if state is cached
        """
        with self._lock:
            return name in self._cache_order

    def cache_size(self) -> int:
        """Get current cache size.

        Returns:
            Number of cached states
        """
        with self._lock:
            return len(self._cache_order)

    def set_max_size(self, max_size: int) -> None:
        """Set maximum cache size.

        Args:
            max_size: New maximum cache size
        """
        with self._lock:
            self._max_cache_size = max_size
            # Evict if needed to meet new size
            while len(self._cache_order) > self._max_cache_size:
                evicted = self._cache_order.pop(0)
                logger.debug(f"Evicted '{evicted}' due to cache resize")

    def _evict_if_needed(self, active_states: set[str]) -> None:
        """Evict least recently used states if cache is full.

        Args:
            active_states: Set of currently active state names to protect from eviction
        """
        while len(self._cache_order) > self._max_cache_size:
            # Find first non-active state to evict
            evicted = None
            for i, name in enumerate(self._cache_order):
                if name not in active_states:
                    evicted = self._cache_order.pop(i)
                    logger.debug(f"Evicted '{evicted}' from cache (LRU)")
                    break

            # If all cached states are active, evict oldest active state
            if evicted is None and self._cache_order:
                evicted = self._cache_order.pop(0)
                logger.warning(
                    f"Evicted active state '{evicted}' from cache (all states active)"
                )
                break

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache_order.clear()
            logger.debug("Cache cleared")

    def get_statistics(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            return {
                "cache_size": len(self._cache_order),
                "max_cache_size": self._max_cache_size,
                "utilization_percent": int(
                    (len(self._cache_order) / max(1, self._max_cache_size)) * 100
                ),
            }
