"""Vector database for efficient similarity search using FAISS.

This module provides vector storage and retrieval for state embeddings,
enabling fast similarity-based matching and state recognition.
"""

import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from ..config import get_settings
from ..exceptions import StorageReadException, StorageWriteException, VectorDatabaseException
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class VectorMetadata:
    """Metadata associated with a vector.

    Attributes:
        id: Unique identifier
        state_name: Associated state name
        element_type: Type of UI element
        timestamp: When added
        source: Source image/screen
        properties: Additional properties
    """

    id: str
    state_name: str | None = None
    element_type: str | None = None
    timestamp: str | None = None
    source: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class VectorStore:
    """FAISS-based vector storage and retrieval.

    Features:
        - Multiple index types (Flat, IVF, HNSW)
        - GPU acceleration support
        - Metadata storage
        - Persistence to disk
        - Batch operations
        - Incremental updates
    """

    def __init__(
        self,
        dimension: int = 512,
        index_type: str = "IVF",
        metric: str = "L2",
        gpu: bool = False,
        nlist: int = 100,
        nprobe: int = 10,
    ):
        """Initialize vector store.

        Args:
            dimension: Vector dimension
            index_type: Index type (Flat, IVF, HNSW)
            metric: Distance metric (L2, IP for inner product)
            gpu: Use GPU acceleration if available
            nlist: Number of clusters for IVF
            nprobe: Number of clusters to search
        """
        self.settings = get_settings()
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.use_gpu = gpu  # GPU support if requested and available
        self.nlist = nlist
        self.nprobe = nprobe

        # Initialize index
        self.index = self._create_index()
        self.id_counter = 0
        self.metadata: dict[int, VectorMetadata] = {}
        self.id_to_index: dict[str, int] = {}  # Map external IDs to FAISS indices

        logger.info(
            "vector_store_initialized",
            dimension=dimension,
            index_type=index_type,
            metric=metric,
            gpu=self.use_gpu,
        )

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on configuration."""
        try:
            # Select metric
            if self.metric == "L2":
                metric_type = faiss.METRIC_L2
            elif self.metric == "IP":
                metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

            # Create base index
            if self.index_type == "Flat":
                index = (
                    faiss.IndexFlatL2(self.dimension)
                    if self.metric == "L2"
                    else faiss.IndexFlatIP(self.dimension)
                )

            elif self.index_type == "IVF":
                quantizer = (
                    faiss.IndexFlatL2(self.dimension)
                    if self.metric == "L2"
                    else faiss.IndexFlatIP(self.dimension)
                )
                index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, metric_type)

            elif self.index_type == "HNSW":
                index = faiss.IndexHNSWFlat(self.dimension, 32, metric_type)

            else:
                raise ValueError(f"Unknown index type: {self.index_type}")

            # Move to GPU if requested
            if self.use_gpu and faiss.get_num_gpus() > 0:
                try:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    logger.debug("vector_index_on_gpu")
                except Exception as e:
                    logger.warning("gpu_index_failed", error=str(e), fallback="cpu")
                    self.use_gpu = False

            return index

        except Exception as e:
            raise VectorDatabaseException("create_index", str(e)) from e

    def add_vectors(
        self,
        vectors: np.ndarray[Any, Any],
        metadata: list[VectorMetadata] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add vectors to the index.

        Args:
            vectors: Vectors to add (N x D)
            metadata: Optional metadata for each vector
            ids: Optional external IDs

        Returns:
            List of assigned IDs
        """
        try:
            # Validate input
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)

            if vectors.shape[1] != self.dimension:
                raise ValueError(
                    f"Vector dimension mismatch: {vectors.shape[1]} != {self.dimension}"
                )

            n_vectors = len(vectors)

            # Generate IDs if not provided
            if ids is None:
                ids = [f"vec_{self.id_counter + i}" for i in range(n_vectors)]

            # Ensure metadata list matches vectors
            if metadata is None:
                metadata = [VectorMetadata(id=id_) for id_ in ids]
            elif len(metadata) != n_vectors:
                raise ValueError(f"Metadata count mismatch: {len(metadata)} != {n_vectors}")

            # Train index if needed (for IVF)
            if self.index_type == "IVF" and not self.index.is_trained:
                logger.debug("training_ivf_index")
                self.index.train(vectors)
                self.index.nprobe = self.nprobe

            # Add vectors
            start_idx = self.id_counter
            self.index.add(vectors)

            # Store metadata and mappings
            for i, (id_, meta) in enumerate(zip(ids, metadata, strict=False)):
                faiss_idx = start_idx + i
                self.metadata[faiss_idx] = meta
                self.id_to_index[id_] = faiss_idx

            self.id_counter += n_vectors

            logger.debug("vectors_added", count=n_vectors, total=self.index.ntotal)

            return ids

        except Exception as e:
            raise VectorDatabaseException("add_vectors", str(e)) from e

    def search(
        self, query_vectors: np.ndarray[Any, Any], k: int = 5, threshold: float | None = None
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], list[list[VectorMetadata | None]]]:
        """Search for similar vectors.

        Args:
            query_vectors: Query vectors (N x D)
            k: Number of nearest neighbors
            threshold: Optional distance threshold

        Returns:
            Tuple of (distances, indices, metadata)
        """
        try:
            # Validate input
            if query_vectors.ndim == 1:
                query_vectors = query_vectors.reshape(1, -1)

            if query_vectors.shape[1] != self.dimension:
                raise ValueError(
                    f"Query dimension mismatch: {query_vectors.shape[1]} != {self.dimension}"
                )

            # Ensure k doesn't exceed index size
            k = min(k, self.index.ntotal)
            if k == 0:
                return np.array([]), np.array([]), []

            # Search
            distances, indices = self.index.search(query_vectors, k)

            # Apply threshold if specified
            if threshold is not None:
                valid_mask = distances <= threshold
                distances = [d[m] for d, m in zip(distances, valid_mask, strict=False)]
                indices = [i[m] for i, m in zip(indices, valid_mask, strict=False)]

            # Retrieve metadata
            metadata_results: list[list[VectorMetadata | None]] = []
            for idx_list in indices:
                meta_list: list[VectorMetadata | None] = []
                for idx in idx_list:
                    if idx >= 0 and idx in self.metadata:
                        meta_list.append(self.metadata[idx])
                    else:
                        meta_list.append(None)
                metadata_results.append(meta_list)

            return distances, indices, metadata_results

        except Exception as e:
            raise VectorDatabaseException("search", str(e)) from e

    def search_by_id(
        self, id_: str, k: int = 5, include_self: bool = False
    ) -> tuple[np.ndarray[Any, Any], list[VectorMetadata]]:
        """Search for similar vectors to a stored vector.

        Args:
            id_: ID of stored vector
            k: Number of neighbors
            include_self: Include the query vector in results

        Returns:
            Tuple of (distances, metadata)
        """
        # Get FAISS index for ID
        if id_ not in self.id_to_index:
            raise ValueError(f"ID not found: {id_}")

        faiss_idx = self.id_to_index[id_]

        # Reconstruct vector
        query_vector = self.index.reconstruct(faiss_idx)

        # Search
        k_search = k + 1 if not include_self else k
        distances, indices, metadata = self.search(query_vector, k_search)

        # Remove self if needed
        if not include_self:
            # Filter out the query vector itself
            mask = indices[0] != faiss_idx
            distances = distances[0][mask][:k]
            metadata_filtered = [
                m for i, m in zip(indices[0], metadata[0], strict=False) if i != faiss_idx
            ][:k]
        else:
            distances = distances[0]
            metadata_filtered = metadata[0]

        # Filter out None values from metadata to match return type
        metadata_final: list[VectorMetadata] = [m for m in metadata_filtered if m is not None]

        return distances, metadata_final

    def update_vector(
        self, id_: str, new_vector: np.ndarray[Any, Any], new_metadata: VectorMetadata | None = None
    ):
        """Update an existing vector.

        Args:
            id_: ID of vector to update
            new_vector: New vector data
            new_metadata: Optional new metadata
        """
        # Note: FAISS doesn't support in-place updates
        # We need to remove and re-add

        # Get current metadata if not provided
        if id_ not in self.id_to_index:
            raise ValueError(f"ID not found: {id_}")

        faiss_idx = self.id_to_index[id_]

        if new_metadata is None:
            new_metadata = self.metadata[faiss_idx]

        # For simplicity, we'll log this limitation
        logger.warning(
            "vector_update_not_supported", id=id_, note="FAISS doesn't support in-place updates"
        )

        # In production, would need to rebuild index or use updateable index

    def remove_vector(self, id_: str):
        """Remove a vector from the index.

        Args:
            id_: ID of vector to remove
        """
        # Note: FAISS doesn't support removal without rebuilding
        logger.warning(
            "vector_removal_not_supported",
            id=id_,
            note="FAISS doesn't support removal without rebuilding",
        )

    def save(self, path: str | Path):
        """Save index and metadata to disk.

        Args:
            path: Directory to save to
        """
        try:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)

            # Save index
            index_path = path / "index.faiss"
            if self.use_gpu:
                # Transfer to CPU for saving
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(index_path))
            else:
                faiss.write_index(self.index, str(index_path))

            # Save metadata
            metadata_path = path / "metadata.pkl"
            with open(metadata_path, "wb") as f:
                pickle.dump(
                    {
                        "metadata": self.metadata,
                        "id_to_index": self.id_to_index,
                        "id_counter": self.id_counter,
                        "config": {
                            "dimension": self.dimension,
                            "index_type": self.index_type,
                            "metric": self.metric,
                            "nlist": self.nlist,
                            "nprobe": self.nprobe,
                        },
                    },
                    f,
                )

            logger.info("vector_store_saved", path=str(path), vectors=self.index.ntotal)

        except Exception as e:
            raise StorageWriteException(
                key=str(path), storage_type="vector_store", reason=str(e)
            ) from e

    def load(self, path: str | Path):
        """Load index and metadata from disk.

        Args:
            path: Directory to load from
        """
        try:
            path = Path(path)

            # Load index
            index_path = path / "index.faiss"
            if not index_path.exists():
                raise FileNotFoundError(f"Index file not found: {index_path}")

            self.index = faiss.read_index(str(index_path))

            # Move to GPU if configured
            if self.use_gpu and faiss.get_num_gpus() > 0:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                except Exception as e:
                    logger.warning("gpu_index_load_failed", error=str(e))

            # Load metadata
            metadata_path = path / "metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    data = pickle.load(f)
                    self.metadata = data["metadata"]
                    self.id_to_index = data["id_to_index"]
                    self.id_counter = data["id_counter"]

                    # Restore config
                    config = data.get("config", {})
                    self.dimension = config.get("dimension", self.dimension)
                    self.index_type = config.get("index_type", self.index_type)

            logger.info("vector_store_loaded", path=str(path), vectors=self.index.ntotal)

        except Exception as e:
            raise StorageReadException(
                key=str(path), storage_type="vector_store", reason=str(e)
            ) from e

    def clear(self):
        """Clear all vectors from the index."""
        self.index = self._create_index()
        self.metadata.clear()
        self.id_to_index.clear()
        self.id_counter = 0
        logger.debug("vector_store_cleared")

    def get_statistics(self) -> dict[str, Any]:
        """Get index statistics.

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "is_trained": self.index.is_trained if hasattr(self.index, "is_trained") else True,
            "gpu": self.use_gpu,
            "metadata_count": len(self.metadata),
        }

        # Add state distribution if available
        if self.metadata:
            state_counts: dict[str, int] = {}
            for meta in self.metadata.values():
                if meta.state_name:
                    state_counts[meta.state_name] = state_counts.get(meta.state_name, 0) + 1
            stats["state_distribution"] = state_counts

        return stats
