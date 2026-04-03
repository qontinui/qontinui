"""Embedding system for semantic search in GUI automation.

This module provides embedding generation and semantic similarity search
for UI elements, enabling AI agents to find elements by meaning rather
than exact text matching.

Example usage:
    >>> from qontinui.embeddings import get_embedding_provider, EmbeddingConfig
    >>>
    >>> # Configure provider (defaults to sentence-transformers)
    >>> config = EmbeddingConfig(provider="sentence-transformers")
    >>> provider = get_embedding_provider(config)
    >>>
    >>> # Generate embeddings
    >>> embedding = provider.embed("Submit button")
    >>>
    >>> # Batch embed elements
    >>> texts = ["Login button", "Email input", "Password field"]
    >>> embeddings = provider.embed_batch(texts)
    >>>
    >>> # Search for similar elements
    >>> from qontinui.embeddings import SemanticSearchEngine
    >>> engine = SemanticSearchEngine(provider)
    >>> results = engine.search("sign in", elements_with_embeddings)
"""

from .cache import CachedEmbeddingProvider, EmbeddingCache
from .config import EmbeddingConfig, EmbeddingProviderType
from .openai_provider import OpenAIProvider
from .provider import (
    EmbeddingProvider,
    get_default_provider,
    get_embedding_provider,
    set_default_provider,
)
from .search import SemanticSearchEngine, SemanticSearchResult
from .sentence_transformers_provider import SentenceTransformersProvider

__all__ = [
    # Config
    "EmbeddingConfig",
    "EmbeddingProviderType",
    # Providers
    "EmbeddingProvider",
    "SentenceTransformersProvider",
    "OpenAIProvider",
    "get_embedding_provider",
    "set_default_provider",
    "get_default_provider",
    # Caching
    "EmbeddingCache",
    "CachedEmbeddingProvider",
    # Search
    "SemanticSearchEngine",
    "SemanticSearchResult",
]
