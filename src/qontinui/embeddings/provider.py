"""Abstract embedding provider interface.

This module defines the base interface for embedding providers and provides
factory functions for creating provider instances.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .config import EmbeddingConfig, EmbeddingProviderType


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    All embedding providers must implement this interface to ensure
    consistent behavior across different backends.

    Attributes:
        config: The embedding configuration
        dimension: The embedding dimension
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize the embedding provider.

        Args:
            config: Embedding configuration
        """
        self.config = config
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """Get the embedding dimension.

        Returns:
            The dimension of the embeddings produced by this provider.

        Raises:
            RuntimeError: If dimension cannot be determined.
        """
        if self._dimension is None:
            # Try to get from config
            dim = self.config.get_dimension()
            if dim is not None:
                self._dimension = dim
            else:
                # Generate a test embedding to determine dimension
                test_embedding = self.embed("test")
                self._dimension = len(test_embedding)
        return self._dimension

    @abstractmethod
    def embed(self, text: str) -> NDArray[np.float32]:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a numpy array.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            A 2D numpy array of shape (len(texts), dimension).
        """
        pass

    def similarity(
        self,
        embedding1: NDArray[np.float32],
        embedding2: NDArray[np.float32],
    ) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Cosine similarity score between -1 and 1.
        """
        # Ensure embeddings are normalized
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def similarities_batch(
        self,
        query_embedding: NDArray[np.float32],
        embeddings: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Calculate cosine similarities between a query and multiple embeddings.

        Args:
            query_embedding: The query embedding vector (1D array).
            embeddings: Matrix of embeddings to compare against (2D array).

        Returns:
            Array of similarity scores.
        """
        # Normalize query
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return np.zeros(len(embeddings), dtype=np.float32)
        query_normalized = query_embedding / query_norm

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings_normalized = embeddings / norms

        # Compute dot products
        similarities = np.dot(embeddings_normalized, query_normalized)
        return cast("NDArray[np.float32]", similarities.astype(np.float32))

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is available and properly configured.

        Returns:
            True if the provider can be used, False otherwise.
        """
        pass

    def get_info(self) -> dict[str, str | int | bool]:
        """Get information about this provider.

        Returns:
            Dictionary with provider information.
        """
        return {
            "provider": self.config.provider.value,
            "model": self.config.get_model_name(),
            "dimension": self.dimension,
            "available": self.is_available(),
        }


# Global default provider instance
_default_provider: EmbeddingProvider | None = None


def get_embedding_provider(config: EmbeddingConfig | None = None) -> EmbeddingProvider:
    """Create an embedding provider based on configuration.

    Args:
        config: Embedding configuration. Uses default config if not provided.

    Returns:
        An embedding provider instance.

    Raises:
        ValueError: If the provider type is not supported.
        ImportError: If required dependencies are not installed.
    """
    if config is None:
        config = EmbeddingConfig()

    if config.provider == EmbeddingProviderType.SENTENCE_TRANSFORMERS:
        from .sentence_transformers_provider import SentenceTransformersProvider

        return SentenceTransformersProvider(config)

    elif config.provider == EmbeddingProviderType.OPENAI:
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(config)

    else:
        raise ValueError(f"Unsupported embedding provider: {config.provider}")


def set_default_provider(provider: EmbeddingProvider) -> None:
    """Set the default embedding provider.

    Args:
        provider: The provider to use as default.
    """
    global _default_provider
    _default_provider = provider


def get_default_provider() -> EmbeddingProvider:
    """Get the default embedding provider.

    Creates a default provider if one hasn't been set.

    Returns:
        The default embedding provider.
    """
    global _default_provider
    if _default_provider is None:
        _default_provider = get_embedding_provider()
    return _default_provider
