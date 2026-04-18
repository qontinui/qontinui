"""Shared LRU cache for per-screenshot CNN feature maps.

Multiple CNN-based detection tiers (ScaleAdaptiveBackend, future neural
backends) share the same haystack screenshot within a single cascade
invocation.  Re-extracting VGG features for every tier would cost
~50-150ms per screenshot on CPU.  This module exposes a tiny, bounded,
content-addressed cache that backends can consult before running a
forward pass.

The cache key is a BLAKE2b digest of the raw screenshot bytes.  BLAKE2b
is measurably faster than SHA-256 on modern CPUs and its output is
collision-resistant for our purposes — we only need to tell two
screenshots apart, not defend against adversarial inputs.

Intentionally tiny (default 4 entries).  VGG feature maps are large
tensors; caching more would bloat process RSS and defeat the whole
point of the unload-on-idle policy used by CNN backends.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any

__all__ = ["ScreenshotFeatureCache", "get_feature_cache", "hash_screenshot"]


def hash_screenshot(image_bytes: bytes | memoryview | Any) -> str:
    """Compute a fast content hash for a screenshot buffer.

    Accepts raw bytes, a memoryview, or any object exposing ``tobytes()``
    (e.g. a numpy array).  Returns a 16-char BLAKE2b hex digest — short
    enough to log cheaply, long enough to avoid accidental collisions
    across the 4-entry cache.
    """
    if hasattr(image_bytes, "tobytes"):
        image_bytes = image_bytes.tobytes()
    if isinstance(image_bytes, memoryview):
        image_bytes = image_bytes.tobytes()
    if not isinstance(image_bytes, bytes):
        raise TypeError(
            "hash_screenshot expects bytes, memoryview, or tobytes()-capable object"
        )
    return hashlib.blake2b(image_bytes, digest_size=8).hexdigest()


class ScreenshotFeatureCache:
    """Bounded LRU cache mapping screenshot hash -> opaque feature payload.

    The payload is opaque — callers decide what to store.  A typical
    payload is a dict keyed by backend name, e.g.
    ``{"scale_adaptive_vgg13_relu4": tensor}``, so that different
    backends sharing the same screenshot hash can each park their own
    intermediate representation.

    Thread-safe: all operations are guarded by a lock.  Contention is
    expected to be negligible because cascade invocations are serialized
    per screenshot in practice.
    """

    def __init__(self, max_entries: int = 4) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def get(self, screenshot_hash: str, backend_key: str) -> Any | None:
        """Return the cached entry for (hash, backend_key) or None.

        Marks the hash as most-recently-used on a hit.
        """
        with self._lock:
            bucket = self._entries.get(screenshot_hash)
            if bucket is None:
                return None
            # Touch LRU ordering — move to end (most recently used).
            self._entries.move_to_end(screenshot_hash)
            return bucket.get(backend_key)

    def put(
        self,
        screenshot_hash: str,
        backend_key: str,
        payload: Any,
    ) -> None:
        """Insert or overwrite the entry for (hash, backend_key).

        Evicts the least-recently-used hash if the cache is full.
        """
        with self._lock:
            bucket = self._entries.get(screenshot_hash)
            if bucket is None:
                bucket = {}
                self._entries[screenshot_hash] = bucket
            bucket[backend_key] = payload
            self._entries.move_to_end(screenshot_hash)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def has(self, screenshot_hash: str, backend_key: str) -> bool:
        with self._lock:
            bucket = self._entries.get(screenshot_hash)
            return bucket is not None and backend_key in bucket

    def clear(self) -> None:
        """Drop all entries.  Exposed for tests."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


_singleton: ScreenshotFeatureCache | None = None
_singleton_lock = threading.Lock()


def get_feature_cache() -> ScreenshotFeatureCache:
    """Return the process-wide screenshot feature cache (lazily created)."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = ScreenshotFeatureCache(max_entries=4)
    return _singleton
