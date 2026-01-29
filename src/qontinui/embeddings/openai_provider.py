"""OpenAI embedding provider.

This module provides embeddings using the OpenAI API, supporting both
the official OpenAI API and compatible third-party APIs.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .config import EmbeddingConfig
from .provider import EmbeddingProvider

# Check if httpx is available (it should be, as it's in qontinui deps)
try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


class OpenAIProvider(EmbeddingProvider):
    """Embedding provider using OpenAI API.

    This provider uses the OpenAI embeddings API. It also supports
    compatible APIs that follow the OpenAI specification.

    Default model: text-embedding-3-small (1536 dimensions)

    Available models:
    - text-embedding-3-small: Fast, affordable, 1536 dimensions
    - text-embedding-3-large: Higher quality, 3072 dimensions
    - text-embedding-ada-002: Legacy model, 1536 dimensions

    Environment variables:
    - OPENAI_API_KEY: API key (can also be passed in config)
    - OPENAI_API_BASE: Custom API base URL

    Example:
        >>> from qontinui.embeddings import EmbeddingConfig, OpenAIProvider
        >>> config = EmbeddingConfig(
        ...     provider="openai",
        ...     api_key="sk-...",  # Or set OPENAI_API_KEY env var
        ... )
        >>> provider = OpenAIProvider(config)
        >>> embedding = provider.embed("Submit button")
    """

    DEFAULT_API_BASE = "https://api.openai.com/v1"

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize the OpenAI provider.

        Args:
            config: Embedding configuration.

        Raises:
            ImportError: If httpx is not installed.
            ValueError: If no API key is provided or found in environment.
        """
        super().__init__(config)

        if not HAS_HTTPX:
            raise ImportError(
                "httpx is required for OpenAIProvider. " "Install it with: pip install httpx"
            )

        # Get API key from config or environment
        self._api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set it in config or OPENAI_API_KEY environment variable."
            )

        # Get API base URL
        self._api_base = config.api_base or os.getenv("OPENAI_API_BASE") or self.DEFAULT_API_BASE

        # Initialize HTTP client
        self._client = httpx.Client(
            timeout=config.timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    def _get_embeddings_url(self) -> str:
        """Get the embeddings API URL.

        Returns:
            The full URL for the embeddings endpoint.
        """
        base = self._api_base.rstrip("/")
        return f"{base}/embeddings"

    def embed(self, text: str) -> NDArray[np.float32]:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a numpy array.
        """
        embeddings = self.embed_batch([text])
        return embeddings[0]

    def embed_batch(self, texts: list[str]) -> NDArray[np.float32]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            A 2D numpy array of shape (len(texts), dimension).

        Raises:
            httpx.HTTPError: If the API request fails.
        """
        if not texts:
            dim = self.config.get_dimension() or 1536
            return np.array([], dtype=np.float32).reshape(0, dim)

        model_name = self.config.get_model_name()
        all_embeddings: list[NDArray[np.float32]] = []

        # Process in batches
        batch_size = self.config.batch_size
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            response = self._client.post(
                self._get_embeddings_url(),
                json={
                    "model": model_name,
                    "input": batch_texts,
                },
            )
            response.raise_for_status()
            result = response.json()

            # Extract embeddings in the correct order
            batch_embeddings = [None] * len(batch_texts)
            for item in result["data"]:
                idx = item["index"]
                batch_embeddings[idx] = np.array(item["embedding"], dtype=np.float32)

            all_embeddings.extend(batch_embeddings)  # type: ignore

        embeddings = np.vstack(all_embeddings)

        # Normalize if requested
        if self.config.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = embeddings / norms

        # Set dimension if not already set
        if self._dimension is None:
            self._dimension = embeddings.shape[1]

        return embeddings.astype(np.float32)

    def is_available(self) -> bool:
        """Check if the OpenAI API is available.

        Makes a test request to verify connectivity.

        Returns:
            True if the API is accessible.
        """
        if not HAS_HTTPX or not self._api_key:
            return False

        try:
            # Make a minimal test request
            response = self._client.post(
                self._get_embeddings_url(),
                json={
                    "model": self.config.get_model_name(),
                    "input": "test",
                },
            )
            return response.status_code == 200
        except Exception:
            return False

    def get_info(self) -> dict[str, str | int | bool]:
        """Get information about this provider.

        Returns:
            Dictionary with provider information.
        """
        info = super().get_info()
        info["api_base"] = self._api_base
        info["has_api_key"] = bool(self._api_key)
        return info

    def __del__(self) -> None:
        """Close the HTTP client on cleanup."""
        if hasattr(self, "_client"):
            self._client.close()
