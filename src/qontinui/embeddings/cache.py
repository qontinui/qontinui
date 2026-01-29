"""Embedding cache for efficient repeated lookups.

This module provides caching functionality to avoid re-computing embeddings
for the same text, reducing latency and API costs.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .provider import EmbeddingProvider


class EmbeddingCache:
    """LRU cache for embeddings with optional disk persistence.

    This cache stores embeddings in memory with LRU eviction policy.
    Optionally, embeddings can be persisted to disk for cross-session reuse.

    Attributes:
        max_size: Maximum number of embeddings to cache in memory.
        cache_dir: Directory for disk persistence (None for memory-only).
    """

    cache_dir: Path | None

    def __init__(
        self,
        max_size: int = 10000,
        cache_dir: str | None = None,
        persist_to_disk: bool = True,
    ) -> None:
        """Initialize the embedding cache.

        Args:
            max_size: Maximum number of embeddings to cache in memory.
            cache_dir: Directory for disk persistence. Uses temp dir if None.
            persist_to_disk: Whether to persist embeddings to disk.
        """
        self.max_size = max_size
        self.persist_to_disk = persist_to_disk

        # Initialize memory cache with LRU ordering
        self._cache: OrderedDict[str, NDArray[np.float32]] = OrderedDict()

        # Set up disk cache directory
        if persist_to_disk:
            if cache_dir:
                self.cache_dir = Path(cache_dir)
            else:
                self.cache_dir = Path(tempfile.gettempdir()) / "qontinui_embeddings"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        # Statistics
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str, model: str) -> str:
        """Create a cache key from text and model name.

        Args:
            text: The text that was embedded.
            model: The model name used for embedding.

        Returns:
            A unique cache key.
        """
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]

    def get(self, text: str, model: str) -> NDArray[np.float32] | None:
        """Get a cached embedding.

        Args:
            text: The text that was embedded.
            model: The model name used for embedding.

        Returns:
            The cached embedding, or None if not found.
        """
        key = self._make_key(text, model)

        # Check memory cache first
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

        # Check disk cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.npy"
            if cache_file.exists():
                try:
                    embedding = cast("NDArray[np.float32]", np.load(cache_file))
                    # Add to memory cache
                    self._add_to_memory_cache(key, embedding)
                    self._hits += 1
                    return embedding
                except Exception:
                    # Corrupted cache file, delete it
                    cache_file.unlink(missing_ok=True)

        self._misses += 1
        return None

    def get_batch(
        self, texts: list[str], model: str
    ) -> tuple[dict[int, NDArray[np.float32]], list[int]]:
        """Get cached embeddings for multiple texts.

        Args:
            texts: List of texts to look up.
            model: The model name used for embedding.

        Returns:
            Tuple of (dict mapping indices to embeddings, list of missing indices).
        """
        found: dict[int, NDArray[np.float32]] = {}
        missing: list[int] = []

        for i, text in enumerate(texts):
            embedding = self.get(text, model)
            if embedding is not None:
                found[i] = embedding
            else:
                missing.append(i)

        return found, missing

    def set(self, text: str, model: str, embedding: NDArray[np.float32]) -> None:
        """Cache an embedding.

        Args:
            text: The text that was embedded.
            model: The model name used for embedding.
            embedding: The embedding vector.
        """
        key = self._make_key(text, model)

        # Add to memory cache
        self._add_to_memory_cache(key, embedding)

        # Persist to disk
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.npy"
            try:
                np.save(cache_file, embedding)
            except Exception:
                pass  # Disk write failure is not critical

    def set_batch(self, texts: list[str], model: str, embeddings: NDArray[np.float32]) -> None:
        """Cache multiple embeddings.

        Args:
            texts: List of texts that were embedded.
            model: The model name used for embedding.
            embeddings: The embedding vectors (2D array).
        """
        for text, embedding in zip(texts, embeddings, strict=False):
            self.set(text, model, embedding)

    def _add_to_memory_cache(self, key: str, embedding: NDArray[np.float32]) -> None:
        """Add an embedding to the memory cache with LRU eviction.

        Args:
            key: Cache key.
            embedding: Embedding vector.
        """
        # Remove oldest entries if at capacity
        while len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)

        self._cache[key] = embedding

    def clear(self) -> None:
        """Clear the cache (both memory and disk)."""
        self._cache.clear()

        if self.cache_dir:
            for cache_file in self.cache_dir.glob("*.npy"):
                cache_file.unlink(missing_ok=True)

        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, int | float]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics.
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        stats = {
            "memory_size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
        }

        if self.cache_dir:
            disk_files = list(self.cache_dir.glob("*.npy"))
            stats["disk_size"] = len(disk_files)

        return stats


class CachedEmbeddingProvider(EmbeddingProvider):
    """Wrapper that adds caching to any embedding provider.

    This class wraps an existing embedding provider and adds caching
    functionality to avoid redundant embedding computations.

    Example:
        >>> from qontinui.embeddings import (
        ...     SentenceTransformersProvider,
        ...     CachedEmbeddingProvider,
        ...     EmbeddingConfig,
        ... )
        >>> config = EmbeddingConfig()
        >>> base_provider = SentenceTransformersProvider(config)
        >>> provider = CachedEmbeddingProvider(base_provider)
        >>> embedding1 = provider.embed("Submit button")  # Computed
        >>> embedding2 = provider.embed("Submit button")  # Cached
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        cache: EmbeddingCache | None = None,
    ) -> None:
        """Initialize the cached provider.

        Args:
            provider: The base embedding provider to wrap.
            cache: Optional custom cache. Creates default cache if None.
        """
        super().__init__(provider.config)
        self._provider = provider

        if cache is None:
            cache = EmbeddingCache(
                max_size=provider.config.cache_max_size,
                cache_dir=provider.config.cache_dir,
                persist_to_disk=provider.config.cache_enabled,
            )
        self._cache = cache

    @property
    def dimension(self) -> int:
        """Get the embedding dimension from the base provider."""
        return self._provider.dimension

    def _get_model_name(self) -> str:
        """Get the model name for cache keys."""
        return self.config.get_model_name()

    def embed(self, text: str) -> NDArray[np.float32]:
        """Generate embedding with caching.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector.
        """
        model = self._get_model_name()

        # Check cache first
        cached = self._cache.get(text, model)
        if cached is not None:
            return cached

        # Generate embedding
        embedding = self._provider.embed(text)

        # Cache the result
        self._cache.set(text, model, embedding)

        return embedding

    def embed_batch(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for multiple texts with caching.

        Args:
            texts: List of texts to embed.

        Returns:
            A 2D numpy array of embeddings.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)

        model = self._get_model_name()

        # Check cache for each text
        found, missing_indices = self._cache.get_batch(texts, model)

        # If all cached, return immediately
        if not missing_indices:
            return np.vstack([found[i] for i in range(len(texts))])

        # Generate missing embeddings
        missing_texts = [texts[i] for i in missing_indices]
        missing_embeddings = self._provider.embed_batch(missing_texts)

        # Cache the new embeddings
        self._cache.set_batch(missing_texts, model, missing_embeddings)

        # Combine cached and new embeddings
        result = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, embedding in found.items():
            result[i] = embedding
        for idx, i in enumerate(missing_indices):
            result[i] = missing_embeddings[idx]

        return result

    def is_available(self) -> bool:
        """Check if the base provider is available."""
        return self._provider.is_available()

    def get_info(self) -> dict[str, str | int | bool]:
        """Get information including cache stats."""
        info = self._provider.get_info()
        info["cached"] = True
        info["cache_stats"] = self._cache.stats()  # type: ignore
        return info

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    def cache_stats(self) -> dict[str, int | float]:
        """Get cache statistics."""
        return self._cache.stats()
