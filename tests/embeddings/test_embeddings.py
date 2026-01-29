"""Tests for the embeddings module.

These tests verify the embedding provider infrastructure works correctly.
Note: Tests that require sentence-transformers are skipped if not installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Load embeddings modules directly to avoid full qontinui import
# which requires many dependencies
_src_path = Path(__file__).parent.parent.parent / "src" / "qontinui" / "embeddings"


def _load_module(name: str, file_name: str):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(name, _src_path / file_name)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {_src_path / file_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load modules in dependency order
_config_mod = _load_module("qontinui.embeddings.config", "config.py")
_provider_mod = _load_module("qontinui.embeddings.provider", "provider.py")
_cache_mod = _load_module("qontinui.embeddings.cache", "cache.py")
_search_mod = _load_module("qontinui.embeddings.search", "search.py")

EmbeddingConfig = _config_mod.EmbeddingConfig
EmbeddingProviderType = _config_mod.EmbeddingProviderType
EmbeddingCache = _cache_mod.EmbeddingCache
get_embedding_provider = _provider_mod.get_embedding_provider
ElementWithEmbedding = _search_mod.ElementWithEmbedding
SemanticSearchEngine = _search_mod.SemanticSearchEngine


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = EmbeddingConfig()
        assert config.provider == EmbeddingProviderType.SENTENCE_TRANSFORMERS
        assert config.batch_size == 32
        assert config.normalize is True
        assert config.cache_enabled is True

    def test_get_model_name_default(self) -> None:
        """Test default model names for different providers."""
        # Default sentence-transformers model
        config = EmbeddingConfig()
        assert config.get_model_name() == "all-MiniLM-L6-v2"

        # Default OpenAI model
        config = EmbeddingConfig(provider=EmbeddingProviderType.OPENAI)
        assert config.get_model_name() == "text-embedding-3-small"

    def test_get_model_name_custom(self) -> None:
        """Test custom model name."""
        config = EmbeddingConfig(model_name="custom-model")
        assert config.get_model_name() == "custom-model"

    def test_get_dimension_known_models(self) -> None:
        """Test dimension lookup for known models."""
        config = EmbeddingConfig(model_name="all-MiniLM-L6-v2")
        assert config.get_dimension() == 384

        config = EmbeddingConfig(model_name="text-embedding-3-small")
        assert config.get_dimension() == 1536


class TestEmbeddingCache:
    """Tests for EmbeddingCache."""

    def test_cache_get_set(self) -> None:
        """Test basic cache operations."""
        cache = EmbeddingCache(max_size=100, persist_to_disk=False)

        # Test set and get
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cache.set("test text", "model-1", embedding)

        retrieved = cache.get("test text", "model-1")
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, embedding)

    def test_cache_miss(self) -> None:
        """Test cache miss returns None."""
        cache = EmbeddingCache(max_size=100, persist_to_disk=False)

        result = cache.get("nonexistent", "model-1")
        assert result is None

    def test_cache_different_models(self) -> None:
        """Test caching same text with different models."""
        cache = EmbeddingCache(max_size=100, persist_to_disk=False)

        embedding1 = np.array([1.0, 2.0], dtype=np.float32)
        embedding2 = np.array([3.0, 4.0], dtype=np.float32)

        cache.set("test", "model-1", embedding1)
        cache.set("test", "model-2", embedding2)

        np.testing.assert_array_equal(cache.get("test", "model-1"), embedding1)
        np.testing.assert_array_equal(cache.get("test", "model-2"), embedding2)

    def test_cache_lru_eviction(self) -> None:
        """Test LRU eviction when cache is full."""
        cache = EmbeddingCache(max_size=3, persist_to_disk=False)

        for i in range(5):
            embedding = np.array([float(i)], dtype=np.float32)
            cache.set(f"text-{i}", "model", embedding)

        # First two should be evicted
        assert cache.get("text-0", "model") is None
        assert cache.get("text-1", "model") is None

        # Last three should still be there
        assert cache.get("text-2", "model") is not None
        assert cache.get("text-3", "model") is not None
        assert cache.get("text-4", "model") is not None

    def test_cache_stats(self) -> None:
        """Test cache statistics."""
        cache = EmbeddingCache(max_size=100, persist_to_disk=False)

        embedding = np.array([1.0], dtype=np.float32)
        cache.set("test", "model", embedding)

        # Hit
        cache.get("test", "model")
        # Miss
        cache.get("nonexistent", "model")

        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestElementWithEmbedding:
    """Tests for ElementWithEmbedding."""

    def test_from_element_combines_text(self) -> None:
        """Test text combination from element properties."""
        element = ElementWithEmbedding.from_element(
            element_id="btn-1",
            text_content="Submit",
            accessible_name="Submit form",
            description="Blue submit button",
        )

        # Should combine all text parts
        assert "Submit" in element.text
        assert "Submit form" in element.text
        assert "Blue submit button" in element.text

    def test_from_element_fallback_to_id(self) -> None:
        """Test fallback to element ID when no text available."""
        element = ElementWithEmbedding.from_element(element_id="btn-1")
        assert element.text == "btn-1"

    def test_from_element_with_placeholder(self) -> None:
        """Test placeholder text inclusion."""
        element = ElementWithEmbedding.from_element(
            element_id="input-1",
            placeholder="Enter your email",
        )

        assert "placeholder: Enter your email" in element.text


class TestSemanticSearchEngine:
    """Tests for SemanticSearchEngine.

    Note: These tests use a mock provider since sentence-transformers
    may not be installed. Real embedding tests would require the full setup.
    """

    def test_add_and_clear_elements(self) -> None:
        """Test adding and clearing elements."""

        # Create a simple mock provider
        class MockProvider:
            def __init__(self):
                self.config = EmbeddingConfig()
                self._dimension = 3

            @property
            def dimension(self) -> int:
                return self._dimension

            def embed(self, text: str) -> np.ndarray:
                # Simple hash-based embedding for testing
                h = hash(text) % 1000
                return np.array(
                    [h / 1000, (h * 2 % 1000) / 1000, (h * 3 % 1000) / 1000], dtype=np.float32
                )

            def embed_batch(self, texts: list[str]) -> np.ndarray:
                return np.vstack([self.embed(t) for t in texts])

            def similarities_batch(self, query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
                # Simple dot product similarity
                return np.dot(embeddings, query)

            def is_available(self) -> bool:
                return True

            def get_info(self) -> dict:
                return {"provider": "mock", "model": "test", "dimension": 3, "available": True}

        provider = MockProvider()
        engine = SemanticSearchEngine(provider, similarity_threshold=0.0)

        # Add elements
        elements = [
            ElementWithEmbedding.from_element("btn-1", text_content="Login"),
            ElementWithEmbedding.from_element("btn-2", text_content="Sign Up"),
        ]
        engine.add_elements(elements)

        stats = engine.get_stats()
        assert stats["element_count"] == 2

        # Clear
        engine.clear()
        stats = engine.get_stats()
        assert stats["element_count"] == 0


# Skip provider tests if sentence-transformers not installed
try:
    from sentence_transformers import SentenceTransformer  # noqa: F401

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
class TestSentenceTransformersProvider:
    """Tests for SentenceTransformersProvider (requires sentence-transformers)."""

    def test_provider_creation(self) -> None:
        """Test creating a sentence-transformers provider."""
        config = EmbeddingConfig(provider=EmbeddingProviderType.SENTENCE_TRANSFORMERS)
        provider = get_embedding_provider(config)
        assert provider.is_available()

    def test_embed_single(self) -> None:
        """Test embedding a single text."""
        config = EmbeddingConfig()
        provider = get_embedding_provider(config)

        embedding = provider.embed("Hello world")
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension

    def test_embed_batch(self) -> None:
        """Test embedding multiple texts."""
        config = EmbeddingConfig()
        provider = get_embedding_provider(config)

        texts = ["Hello", "World", "Test"]
        embeddings = provider.embed_batch(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)

    def test_similarity(self) -> None:
        """Test similarity calculation."""
        config = EmbeddingConfig()
        provider = get_embedding_provider(config)

        # Similar texts should have high similarity
        emb1 = provider.embed("login button")
        emb2 = provider.embed("sign in button")
        emb3 = provider.embed("download file")

        sim_similar = provider.similarity(emb1, emb2)
        sim_different = provider.similarity(emb1, emb3)

        # login/sign in should be more similar than login/download
        assert sim_similar > sim_different
