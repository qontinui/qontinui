"""Element matching and similarity computation."""

from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class MatchResult:
    """Result of element matching."""

    element_id: str
    similarity_score: float
    bbox: tuple[int, int, int, int]
    metadata: dict[str, Any]


class ElementMatcher:
    """Match UI elements using various similarity metrics."""

    def __init__(self, use_faiss: bool = True, embedding_dim: int = 512):
        """Initialize ElementMatcher.

        Args:
            use_faiss: Whether to use FAISS for efficient similarity search
            embedding_dim: Dimension of embedding vectors
        """
        self.use_faiss = use_faiss
        self.embedding_dim = embedding_dim
        self.index = None
        self.element_metadata = {}

        if use_faiss:
            self._initialize_faiss()

    def _initialize_faiss(self):
        """Initialize FAISS index for similarity search."""
        try:
            # Create FAISS index for cosine similarity
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            # Alternatively, use IndexHNSWFlat for larger datasets
            # self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
            print(f"FAISS index initialized with dimension {self.embedding_dim}")
        except ImportError:
            print("FAISS not available, using sklearn for similarity search")
            self.use_faiss = False
        except Exception as e:
            print(f"Failed to initialize FAISS: {e}")
            self.use_faiss = False

    def add_elements(self, embeddings: np.ndarray, metadata: list[dict[str, Any]]):
        """Add elements to the search index.

        Args:
            embeddings: Array of embedding vectors (n_elements x embedding_dim)
            metadata: List of metadata dicts for each element
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match metadata entries")

        # Normalize embeddings for cosine similarity
        embeddings = self._normalize_embeddings(embeddings)

        if self.use_faiss and self.index is not None:
            # Add to FAISS index
            self.index.add(embeddings.astype(np.float32))

            # Store metadata
            base_idx = len(self.element_metadata)
            for i, meta in enumerate(metadata):
                self.element_metadata[base_idx + i] = meta
        else:
            # Store embeddings and metadata for sklearn fallback
            if not hasattr(self, "_embeddings"):
                self._embeddings = []
                self._metadata = []

            self._embeddings.extend(embeddings)
            self._metadata.extend(metadata)

    def find_similar(
        self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.7
    ) -> list[MatchResult]:
        """Find similar elements to a query embedding.

        Args:
            query_embedding: Query embedding vector
            k: Number of top matches to return
            threshold: Minimum similarity threshold

        Returns:
            List of MatchResult objects
        """
        # Normalize query embedding
        query_embedding = self._normalize_embeddings(query_embedding.reshape(1, -1))

        if self.use_faiss and self.index is not None:
            return self._find_similar_faiss(query_embedding, k, threshold)
        else:
            return self._find_similar_sklearn(query_embedding, k, threshold)

    def _find_similar_faiss(
        self, query_embedding: np.ndarray, k: int, threshold: float
    ) -> list[MatchResult]:
        """Find similar elements using FAISS.

        Args:
            query_embedding: Normalized query embedding
            k: Number of results
            threshold: Similarity threshold

        Returns:
            List of MatchResult objects
        """
        # Search in FAISS index
        k = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)

        results = []
        for dist, idx in zip(distances[0], indices[0], strict=False):
            if dist >= threshold and idx != -1:
                meta = self.element_metadata.get(idx, {})
                result = MatchResult(
                    element_id=meta.get("id", f"element_{idx}"),
                    similarity_score=float(dist),
                    bbox=meta.get("bbox", (0, 0, 0, 0)),
                    metadata=meta,
                )
                results.append(result)

        return results

    def _find_similar_sklearn(
        self, query_embedding: np.ndarray, k: int, threshold: float
    ) -> list[MatchResult]:
        """Find similar elements using sklearn.

        Args:
            query_embedding: Normalized query embedding
            k: Number of results
            threshold: Similarity threshold

        Returns:
            List of MatchResult objects
        """
        if not hasattr(self, "_embeddings") or not self._embeddings:
            return []

        # Compute cosine similarities
        embeddings_array = np.array(self._embeddings)
        similarities = cosine_similarity(query_embedding, embeddings_array)[0]

        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            sim = similarities[idx]
            if sim >= threshold:
                meta = self._metadata[idx]
                result = MatchResult(
                    element_id=meta.get("id", f"element_{idx}"),
                    similarity_score=float(sim),
                    bbox=meta.get("bbox", (0, 0, 0, 0)),
                    metadata=meta,
                )
                results.append(result)

        return results

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity.

        Args:
            embeddings: Embedding vectors

        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms

    def match_template(
        self, template_image: np.ndarray, screenshot: np.ndarray, method: str = "correlation"
    ) -> list[MatchResult]:
        """Match template image in screenshot using traditional CV.

        Args:
            template_image: Template image to find
            screenshot: Screenshot to search in
            method: Matching method ('correlation', 'squared_diff', 'ccoeff')

        Returns:
            List of MatchResult objects
        """
        import cv2

        # Convert images to grayscale
        if len(template_image.shape) == 3:
            template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template_image

        if len(screenshot.shape) == 3:
            screen_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        else:
            screen_gray = screenshot

        # Get template dimensions
        h, w = template_gray.shape

        # Choose matching method
        methods = {
            "correlation": cv2.TM_CCORR_NORMED,
            "squared_diff": cv2.TM_SQDIFF_NORMED,
            "ccoeff": cv2.TM_CCOEFF_NORMED,
        }
        cv_method = methods.get(method, cv2.TM_CCOEFF_NORMED)

        # Perform template matching
        result = cv2.matchTemplate(screen_gray, template_gray, cv_method)

        # Find all matches above threshold
        threshold = 0.8
        if method == "squared_diff":
            # For SQDIFF, lower values mean better match
            locations = np.where(result <= 1 - threshold)
        else:
            locations = np.where(result >= threshold)

        matches = []
        for pt in zip(*locations[::-1], strict=False):
            x, y = pt
            score = result[y, x]

            # Convert score for squared_diff
            if method == "squared_diff":
                score = 1 - score

            match = MatchResult(
                element_id=f"template_match_{x}_{y}",
                similarity_score=float(score),
                bbox=(x, y, w, h),
                metadata={"method": method, "position": (x, y)},
            )
            matches.append(match)

        # Sort by score
        matches.sort(key=lambda m: m.similarity_score, reverse=True)

        return matches

    def clear_index(self):
        """Clear the search index and metadata."""
        if self.use_faiss and self.index is not None:
            # Reset FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        self.element_metadata = {}

        if hasattr(self, "_embeddings"):
            self._embeddings = []
            self._metadata = []

    def save_index(self, path: str):
        """Save the search index to disk.

        Args:
            path: Path to save the index
        """
        import pickle

        if self.use_faiss and self.index is not None:
            # Save FAISS index
            faiss.write_index(self.index, f"{path}.faiss")

            # Save metadata
            with open(f"{path}.meta", "wb") as f:
                pickle.dump(self.element_metadata, f)
        else:
            # Save sklearn data
            data = {
                "embeddings": self._embeddings if hasattr(self, "_embeddings") else [],
                "metadata": self._metadata if hasattr(self, "_metadata") else [],
            }
            with open(path, "wb") as f:
                pickle.dump(data, f)

    def load_index(self, path: str):
        """Load search index from disk.

        Args:
            path: Path to load the index from
        """
        import pickle

        if self.use_faiss:
            try:
                # Load FAISS index
                self.index = faiss.read_index(f"{path}.faiss")

                # Load metadata
                with open(f"{path}.meta", "rb") as f:
                    self.element_metadata = pickle.load(f)
            except FileNotFoundError:
                print(f"Index files not found at {path}")
        else:
            # Load sklearn data
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                    self._embeddings = data.get("embeddings", [])
                    self._metadata = data.get("metadata", [])
            except FileNotFoundError:
                print(f"Index file not found at {path}")
