"""Semantic search engine for UI elements.

This module provides semantic search capabilities using embeddings,
allowing AI agents to find UI elements by meaning rather than exact text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .provider import EmbeddingProvider


class HasEmbedding(Protocol):
    """Protocol for objects that have an embedding."""

    embedding: NDArray[np.float32] | None


@dataclass
class SemanticSearchResult:
    """Result from a semantic search operation.

    Attributes:
        item: The matched item.
        similarity: Cosine similarity score (0 to 1).
        rank: Position in search results (1-indexed).
    """

    item: Any
    similarity: float
    rank: int

    def __repr__(self) -> str:
        return f"SemanticSearchResult(similarity={self.similarity:.3f}, rank={self.rank})"


@dataclass
class ElementWithEmbedding:
    """UI element with its embedding for semantic search.

    This class wraps element data with its embedding and searchable text,
    enabling efficient semantic search operations.
    """

    element_id: str
    text: str  # Combined searchable text
    embedding: NDArray[np.float32] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_element(
        cls,
        element_id: str,
        text_content: str | None = None,
        accessible_name: str | None = None,
        label: str | None = None,
        placeholder: str | None = None,
        description: str | None = None,
        **metadata: Any,
    ) -> ElementWithEmbedding:
        """Create from UI element properties.

        Combines various text fields into a single searchable text.

        Args:
            element_id: Unique element identifier.
            text_content: Visible text content.
            accessible_name: ARIA accessible name.
            label: Associated label text.
            placeholder: Placeholder text (for inputs).
            description: AI-generated description.
            **metadata: Additional metadata to store.

        Returns:
            ElementWithEmbedding instance.
        """
        # Combine text fields for embedding, prioritizing most specific
        text_parts = []

        if description:
            text_parts.append(description)
        if accessible_name:
            text_parts.append(accessible_name)
        if label:
            text_parts.append(label)
        if text_content:
            text_parts.append(text_content)
        if placeholder:
            text_parts.append(f"placeholder: {placeholder}")

        combined_text = " ".join(filter(None, text_parts)) or element_id

        return cls(
            element_id=element_id,
            text=combined_text,
            metadata=metadata,
        )


T = TypeVar("T")


class SemanticSearchEngine:
    """Semantic search engine for UI elements.

    This engine uses embeddings to perform semantic search, finding elements
    by meaning rather than exact text matching. It supports both pre-computed
    embeddings and on-the-fly embedding generation.

    Example:
        >>> from qontinui.embeddings import (
        ...     SemanticSearchEngine,
        ...     get_embedding_provider,
        ... )
        >>> provider = get_embedding_provider()
        >>> engine = SemanticSearchEngine(provider)
        >>>
        >>> # Add elements
        >>> elements = [
        ...     ElementWithEmbedding.from_element("btn-1", text_content="Login"),
        ...     ElementWithEmbedding.from_element("btn-2", text_content="Sign Up"),
        ... ]
        >>> engine.add_elements(elements)
        >>>
        >>> # Search
        >>> results = engine.search("sign in")
        >>> print(results[0].item.element_id)  # "btn-1" (Login is similar to sign in)
    """

    def __init__(
        self,
        provider: EmbeddingProvider,
        similarity_threshold: float = 0.5,
    ) -> None:
        """Initialize the semantic search engine.

        Args:
            provider: Embedding provider for generating embeddings.
            similarity_threshold: Minimum similarity score for results.
        """
        self.provider = provider
        self.similarity_threshold = similarity_threshold

        # Storage for elements and their embeddings
        self._elements: list[ElementWithEmbedding] = []
        self._embeddings: NDArray[np.float32] | None = None

    def add_elements(
        self,
        elements: list[ElementWithEmbedding],
        generate_embeddings: bool = True,
    ) -> None:
        """Add elements to the search index.

        Args:
            elements: Elements to add.
            generate_embeddings: Whether to generate embeddings for elements
                that don't have them.
        """
        for element in elements:
            self._elements.append(element)

        if generate_embeddings:
            self._generate_missing_embeddings()

        self._rebuild_embedding_matrix()

    def clear(self) -> None:
        """Clear all elements from the index."""
        self._elements.clear()
        self._embeddings = None

    def _generate_missing_embeddings(self) -> None:
        """Generate embeddings for elements that don't have them."""
        # Find elements without embeddings
        missing_indices = [
            i for i, el in enumerate(self._elements) if el.embedding is None
        ]

        if not missing_indices:
            return

        # Generate embeddings in batch
        texts = [self._elements[i].text for i in missing_indices]
        embeddings = self.provider.embed_batch(texts)

        # Assign embeddings to elements
        for idx, embedding in zip(missing_indices, embeddings, strict=False):
            self._elements[idx].embedding = embedding

    def _rebuild_embedding_matrix(self) -> None:
        """Rebuild the embedding matrix for efficient search."""
        if not self._elements:
            self._embeddings = None
            return

        # Stack all embeddings into a matrix
        embeddings = [el.embedding for el in self._elements if el.embedding is not None]
        if embeddings:
            self._embeddings = np.vstack(embeddings)
        else:
            self._embeddings = None

    def search(
        self,
        query: str,
        limit: int = 10,
        threshold: float | None = None,
    ) -> list[SemanticSearchResult]:
        """Search for elements semantically similar to the query.

        Args:
            query: Natural language query.
            limit: Maximum number of results to return.
            threshold: Minimum similarity score. Uses default if None.

        Returns:
            List of SemanticSearchResult sorted by similarity.
        """
        if not self._elements or self._embeddings is None:
            return []

        threshold = threshold if threshold is not None else self.similarity_threshold

        # Generate query embedding
        query_embedding = self.provider.embed(query)

        # Calculate similarities
        similarities = self.provider.similarities_batch(query_embedding, self._embeddings)

        # Get indices sorted by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]

        # Build results
        results = []
        for rank, idx in enumerate(sorted_indices[:limit], start=1):
            similarity = float(similarities[idx])
            if similarity < threshold:
                break

            results.append(
                SemanticSearchResult(
                    item=self._elements[idx],
                    similarity=similarity,
                    rank=rank,
                )
            )

        return results

    def find_similar(
        self,
        element_id: str,
        limit: int = 5,
        exclude_self: bool = True,
    ) -> list[SemanticSearchResult]:
        """Find elements similar to a given element.

        Args:
            element_id: ID of the reference element.
            limit: Maximum number of results.
            exclude_self: Whether to exclude the reference element.

        Returns:
            List of similar elements.
        """
        # Find the reference element
        ref_element = None
        ref_idx = None
        for i, el in enumerate(self._elements):
            if el.element_id == element_id:
                ref_element = el
                ref_idx = i
                break

        if ref_element is None or ref_element.embedding is None:
            return []

        # Calculate similarities
        if self._embeddings is None:
            return []

        similarities = self.provider.similarities_batch(
            ref_element.embedding, self._embeddings
        )

        # Get indices sorted by similarity
        sorted_indices = np.argsort(similarities)[::-1]

        # Build results
        results = []
        rank = 1
        for idx in sorted_indices:
            if exclude_self and idx == ref_idx:
                continue

            if len(results) >= limit:
                break

            similarity = float(similarities[idx])
            if similarity < self.similarity_threshold:
                break

            results.append(
                SemanticSearchResult(
                    item=self._elements[idx],
                    similarity=similarity,
                    rank=rank,
                )
            )
            rank += 1

        return results

    def update_element(
        self,
        element_id: str,
        new_text: str | None = None,
        regenerate_embedding: bool = True,
    ) -> bool:
        """Update an element's text and optionally regenerate its embedding.

        Args:
            element_id: ID of the element to update.
            new_text: New searchable text for the element.
            regenerate_embedding: Whether to regenerate the embedding.

        Returns:
            True if the element was found and updated.
        """
        for _i, el in enumerate(self._elements):
            if el.element_id == element_id:
                if new_text:
                    el.text = new_text
                if regenerate_embedding:
                    el.embedding = self.provider.embed(el.text)
                self._rebuild_embedding_matrix()
                return True
        return False

    def remove_element(self, element_id: str) -> bool:
        """Remove an element from the search index.

        Args:
            element_id: ID of the element to remove.

        Returns:
            True if the element was found and removed.
        """
        for i, el in enumerate(self._elements):
            if el.element_id == element_id:
                del self._elements[i]
                self._rebuild_embedding_matrix()
                return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the search index.

        Returns:
            Dictionary with index statistics.
        """
        return {
            "element_count": len(self._elements),
            "embedding_dimension": self.provider.dimension,
            "similarity_threshold": self.similarity_threshold,
            "provider_info": self.provider.get_info(),
        }
