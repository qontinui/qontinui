"""Sentence Transformers embedding provider.

This module provides embeddings using the sentence-transformers library,
which offers high-quality local embeddings without API calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .config import EmbeddingConfig
from .provider import EmbeddingProvider

# Check if sentence-transformers is available
try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class SentenceTransformersProvider(EmbeddingProvider):
    """Embedding provider using sentence-transformers.

    This provider uses the sentence-transformers library for generating
    embeddings locally. It supports a wide variety of pre-trained models
    optimized for semantic similarity.

    Default model: all-MiniLM-L6-v2 (384 dimensions, fast and efficient)

    Other popular models:
    - all-mpnet-base-v2: Higher quality, 768 dimensions
    - paraphrase-MiniLM-L6-v2: Optimized for paraphrase detection
    - multi-qa-MiniLM-L6-cos-v1: Optimized for question-answer matching

    Example:
        >>> from qontinui.embeddings import EmbeddingConfig, SentenceTransformersProvider
        >>> config = EmbeddingConfig(model_name="all-MiniLM-L6-v2")
        >>> provider = SentenceTransformersProvider(config)
        >>> embedding = provider.embed("Submit button")
        >>> print(f"Dimension: {len(embedding)}")
        Dimension: 384
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize the sentence-transformers provider.

        Args:
            config: Embedding configuration.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        super().__init__(config)

        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformersProvider. "
                "Install it with: pip install sentence-transformers"
            )

        self._model: SentenceTransformer | None = None
        self._device: str | None = None

    def _get_device(self) -> str:
        """Determine the device to use for inference.

        Returns:
            Device string (cpu, cuda, or mps).
        """
        if self._device is not None:
            return self._device

        device = self.config.device
        if device == "auto":
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self._device = device
        return device

    def _load_model(self) -> SentenceTransformer:
        """Load the sentence-transformers model lazily.

        Returns:
            The loaded SentenceTransformer model.
        """
        if self._model is None:
            model_name = self.config.get_model_name()
            device = self._get_device()

            self._model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=self.config.trust_remote_code,
            )

            # Set the embedding dimension
            self._dimension = self._model.get_sentence_embedding_dimension()

        return self._model

    def embed(self, text: str) -> NDArray[np.float32]:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a numpy array.
        """
        model = self._load_model()

        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
        )

        return cast("NDArray[np.float32]", embedding.astype(np.float32))

    def embed_batch(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            A 2D numpy array of shape (len(texts), dimension).
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)

        model = self._load_model()

        embeddings = model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=len(texts) > 100,
        )

        return cast("NDArray[np.float32]", embeddings.astype(np.float32))

    def is_available(self) -> bool:
        """Check if sentence-transformers is available.

        Returns:
            True if the provider can be used.
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            return False

        try:
            # Try to load the model
            self._load_model()
            return True
        except Exception:
            return False

    def get_info(self) -> dict[str, str | int | bool]:
        """Get information about this provider.

        Returns:
            Dictionary with provider information.
        """
        info = super().get_info()
        info["device"] = self._get_device()
        info["library_available"] = HAS_SENTENCE_TRANSFORMERS
        return info
