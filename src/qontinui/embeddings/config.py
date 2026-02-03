"""Configuration for embedding providers.

This module defines configuration options for different embedding providers
including local models (sentence-transformers) and API-based services (OpenAI).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class EmbeddingProviderType(str, Enum):
    """Supported embedding provider types."""

    SENTENCE_TRANSFORMERS = "sentence-transformers"
    OPENAI = "openai"


class EmbeddingConfig(BaseModel):
    """Configuration for embedding providers.

    Attributes:
        provider: The embedding provider to use
        model_name: Model identifier for the provider
        dimension: Embedding dimension (auto-detected if not specified)
        batch_size: Maximum batch size for embedding generation
        normalize: Whether to normalize embeddings to unit length
        cache_enabled: Whether to cache embeddings
        cache_dir: Directory for caching embeddings
        api_key: API key for API-based providers (OpenAI)
        api_base: Custom API base URL for OpenAI-compatible APIs
        device: Device for local models (cpu, cuda, mps)
    """

    provider: EmbeddingProviderType = Field(
        default=EmbeddingProviderType.SENTENCE_TRANSFORMERS,
        description="The embedding provider to use",
    )

    # Model configuration
    model_name: str | None = Field(
        default=None,
        description="Model name/identifier. Defaults to provider-specific default.",
    )
    dimension: int | None = Field(
        default=None,
        description="Embedding dimension. Auto-detected if not specified.",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=1000,
        description="Maximum batch size for embedding generation",
    )
    normalize: bool = Field(
        default=True,
        description="Whether to normalize embeddings to unit length",
    )

    # Caching
    cache_enabled: bool = Field(
        default=True,
        description="Whether to cache embeddings",
    )
    cache_dir: str | None = Field(
        default=None,
        description="Directory for caching embeddings. Uses temp dir if not specified.",
    )
    cache_max_size: int = Field(
        default=10000,
        ge=100,
        description="Maximum number of embeddings to cache in memory",
    )

    # API configuration (for OpenAI and compatible APIs)
    api_key: str | None = Field(
        default=None,
        description="API key for API-based providers",
    )
    api_base: str | None = Field(
        default=None,
        description="Custom API base URL for OpenAI-compatible APIs",
    )
    timeout: float = Field(
        default=30.0,
        ge=1.0,
        description="Request timeout in seconds",
    )

    # Local model configuration
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="auto",
        description="Device for local models. 'auto' selects best available.",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code when loading models",
    )

    model_config = {"extra": "forbid"}

    def get_model_name(self) -> str:
        """Get the model name, using provider-specific defaults."""
        if self.model_name:
            return self.model_name

        # Provider-specific defaults
        defaults = {
            EmbeddingProviderType.SENTENCE_TRANSFORMERS: "all-MiniLM-L6-v2",
            EmbeddingProviderType.OPENAI: "text-embedding-3-small",
        }
        return defaults.get(self.provider, "all-MiniLM-L6-v2")

    def get_dimension(self) -> int | None:
        """Get the embedding dimension, using provider-specific defaults."""
        if self.dimension:
            return self.dimension

        # Known dimensions for common models
        known_dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return known_dimensions.get(self.get_model_name())
