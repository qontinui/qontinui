"""Action caching for deterministic replay.

Caches successful action targets (coordinates, regions) for fast replay.
When cached entries fail validation, they are automatically invalidated.
"""

import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .cache_storage import CacheStorage
from .cache_types import (
    CachedCoordinates,
    CacheEntry,
    CacheResult,
    CacheStats,
    ValidationPattern,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ..model.element import Pattern, Region

logger = logging.getLogger(__name__)


class ActionCache:
    """Cache for action targets enabling deterministic replay.

    Stores successful element locations (coordinates, regions) keyed by
    a combination of state, action type, and target pattern. On cache hit,
    the cached coordinates are validated against the current screen before
    being returned.

    This enables "write once, run forever" automation where stable workflows
    execute without any pattern matching overhead.

    Attributes:
        enabled: Whether caching is enabled.
        storage: Underlying storage backend.
    """

    def __init__(
        self,
        cache_dir: "Path | str | None" = None,
        enabled: bool = True,
        validation_similarity: float = 0.9,
        max_age_seconds: float | None = None,
        max_size_bytes: int | None = None,
    ) -> None:
        """Initialize action cache.

        Args:
            cache_dir: Directory for cache files. Defaults to ~/.qontinui/cache
            enabled: Whether caching is enabled. Default True.
            validation_similarity: Minimum similarity for validating cached entries.
                                  Range 0.0-1.0. Default 0.9.
            max_age_seconds: Maximum age of cache entries in seconds. None for no limit.
            max_size_bytes: Maximum cache size in bytes. None for no limit.
        """
        self.enabled = enabled
        self.validation_similarity = validation_similarity
        self.max_age_seconds = max_age_seconds
        self.max_size_bytes = max_size_bytes

        self.storage = CacheStorage(cache_dir)

        # Runtime statistics
        self._hits = 0
        self._misses = 0
        self._invalidations = 0

    def build_key(
        self,
        pattern: "Pattern",
        state_id: str | None = None,
        action_type: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Build a cache key from pattern and context.

        Args:
            pattern: The target pattern being searched for.
            state_id: Optional state machine state ID.
            action_type: Optional action type (click, type, etc.).
            context: Optional additional context (screen resolution, etc.).

        Returns:
            SHA-256 hash string as cache key.
        """
        # Build key components
        components: dict[str, Any] = {
            "pattern_hash": self._hash_pattern(pattern),
            "pattern_name": pattern.name,
        }

        if state_id:
            components["state_id"] = state_id

        if action_type:
            components["action_type"] = action_type

        if context:
            # Only include serializable context
            serializable_context: dict[str, Any] = {}
            for k, v in context.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    serializable_context[k] = v
                elif isinstance(v, (list, tuple)):
                    serializable_context[k] = list(v)
            if serializable_context:
                components["context"] = serializable_context

        # Create deterministic JSON and hash it
        payload = json.dumps(components, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _hash_pattern(self, pattern: "Pattern") -> str:
        """Create a hash of pattern pixel data.

        Args:
            pattern: Pattern with pixel data.

        Returns:
            SHA-256 hash of pixel data.
        """
        if pattern.pixel_data is None:
            return hashlib.sha256(pattern.name.encode()).hexdigest()

        # Hash the raw pixel data
        pixel_bytes = pattern.pixel_data.tobytes()
        return hashlib.sha256(pixel_bytes).hexdigest()

    def try_get(
        self,
        key: str,
        screenshot: np.ndarray | None = None,
        pattern: "Pattern | None" = None,
    ) -> CacheResult:
        """Attempt to retrieve and validate a cached entry.

        Args:
            key: Cache key from build_key().
            screenshot: Current screenshot for validation. If None, skips validation.
            pattern: Original pattern for validation. If None, skips validation.

        Returns:
            CacheResult with hit=True if valid entry found, False otherwise.
        """
        if not self.enabled:
            return CacheResult(hit=False, invalidation_reason="Cache disabled")

        entry = self.storage.get(key)
        if entry is None:
            self._misses += 1
            return CacheResult(hit=False, invalidation_reason="Cache miss")

        # Check age limit
        if self.max_age_seconds is not None:
            age = time.time() - entry.timestamp
            if age > self.max_age_seconds:
                self._invalidations += 1
                self.storage.delete(key)
                return CacheResult(
                    hit=False,
                    invalidation_reason=f"Entry expired (age={age:.0f}s)",
                )

        # Validate if we have screenshot and pattern
        if screenshot is not None and pattern is not None:
            is_valid, reason = self._validate_entry(entry, screenshot, pattern)
            if not is_valid:
                self._invalidations += 1
                self.storage.delete(key)
                return CacheResult(hit=False, invalidation_reason=reason)

        # Update hit count
        entry.hit_count += 1
        self.storage.put(key, entry)

        self._hits += 1
        logger.debug(f"Cache hit for key {key[:16]}... (hit #{entry.hit_count})")

        return CacheResult(hit=True, entry=entry)

    def _validate_entry(
        self,
        entry: CacheEntry,
        screenshot: np.ndarray,
        pattern: "Pattern",
    ) -> tuple[bool, str | None]:
        """Validate a cached entry against current screen.

        Args:
            entry: Cached entry to validate.
            screenshot: Current screenshot.
            pattern: Pattern to validate against.

        Returns:
            Tuple of (is_valid, reason_if_invalid).
        """
        coords = entry.coordinates

        # Check if coordinates are within screen bounds
        if (
            coords.region_x < 0
            or coords.region_y < 0
            or coords.region_x + coords.region_width > screenshot.shape[1]
            or coords.region_y + coords.region_height > screenshot.shape[0]
        ):
            return False, "Cached region outside screen bounds"

        # Extract region from screenshot
        region = screenshot[
            coords.region_y : coords.region_y + coords.region_height,
            coords.region_x : coords.region_x + coords.region_width,
        ]

        # Quick validation: compare region to pattern
        if pattern.pixel_data is not None:
            template = pattern.pixel_data
            if len(template.shape) == 3 and template.shape[2] == 4:
                template = template[:, :, :3]  # Remove alpha

            # Check size match
            if region.shape[:2] != template.shape[:2]:
                return (
                    False,
                    f"Size mismatch: cached={region.shape[:2]}, template={template.shape[:2]}",
                )

            # Calculate similarity using normalized cross-correlation
            try:
                result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
                similarity = float(result[0, 0]) if result.size > 0 else 0.0
            except cv2.error:
                return False, "OpenCV validation error"

            if similarity < self.validation_similarity:
                return (
                    False,
                    f"Similarity {similarity:.3f} below threshold {self.validation_similarity}",
                )

        return True, None

    def store(
        self,
        key: str,
        coordinates: tuple[int, int],
        region: "Region",
        confidence: float,
        pattern: "Pattern | None" = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store a successful action result in cache.

        Args:
            key: Cache key from build_key().
            coordinates: (x, y) center coordinates of the match.
            region: Bounding region of the match.
            confidence: Match confidence (0.0-1.0).
            pattern: Optional pattern for validation storage.
            metadata: Optional additional metadata.

        Returns:
            True if stored successfully.
        """
        if not self.enabled:
            return False

        cached_coords = CachedCoordinates(
            x=coordinates[0],
            y=coordinates[1],
            region_x=region.x,
            region_y=region.y,
            region_width=region.width,
            region_height=region.height,
        )

        validation_pattern = None
        if pattern is not None and pattern.pixel_data is not None:
            validation_pattern = ValidationPattern(
                content_hash=self._hash_pattern(pattern),
                width=pattern.pixel_data.shape[1],
                height=pattern.pixel_data.shape[0],
            )

        entry = CacheEntry(
            coordinates=cached_coords,
            confidence=confidence,
            timestamp=time.time(),
            validation_pattern=validation_pattern,
            metadata=metadata or {},
        )

        success = self.storage.put(key, entry)

        if success:
            logger.debug(
                f"Cached action for key {key[:16]}... at ({coordinates[0]}, {coordinates[1]})"
            )

            # Prune if over size limit
            if self.max_size_bytes is not None:
                self.storage.prune_by_size(self.max_size_bytes)

        return success

    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry.

        Args:
            key: Cache key to invalidate.

        Returns:
            True if invalidated (or didn't exist).
        """
        self._invalidations += 1
        return self.storage.delete(key)

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared.
        """
        count = self.storage.clear()
        self._hits = 0
        self._misses = 0
        self._invalidations = 0
        logger.info(f"Cleared {count} cache entries")
        return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats with current statistics.
        """
        return CacheStats(
            total_entries=len(self.storage.list_keys()),
            hits=self._hits,
            misses=self._misses,
            invalidations=self._invalidations,
            size_bytes=self.storage.get_size_bytes(),
        )

    def prune(self) -> int:
        """Prune expired and excess entries.

        Returns:
            Number of entries pruned.
        """
        count = 0

        if self.max_age_seconds is not None:
            count += self.storage.prune_old_entries(self.max_age_seconds)

        if self.max_size_bytes is not None:
            count += self.storage.prune_by_size(self.max_size_bytes)

        if count > 0:
            logger.info(f"Pruned {count} cache entries")

        return count


# Global cache instance
_default_cache: ActionCache | None = None


def get_action_cache() -> ActionCache:
    """Get the global action cache instance.

    Returns:
        The global ActionCache instance.
    """
    global _default_cache
    if _default_cache is None:
        _default_cache = ActionCache()
    return _default_cache


def set_action_cache(cache: ActionCache) -> None:
    """Set the global action cache instance.

    Args:
        cache: ActionCache instance to use globally.
    """
    global _default_cache
    _default_cache = cache
