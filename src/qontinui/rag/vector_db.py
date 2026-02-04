"""
Qdrant local vector database wrapper for Qontinui RAG.

Provides file-based vector storage without requiring a Qdrant server.
"""

import logging
from pathlib import Path
from typing import Any, cast

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from .models import GUIElementChunk, SearchResult

logger = logging.getLogger(__name__)


class QdrantLocalDB:
    """
    Local Qdrant database using file-based storage.

    Uses Qdrant's embedded mode (no server required) with persistence to disk.
    """

    def __init__(self, db_path: Path) -> None:
        """
        Initialize local Qdrant database.

        Args:
            db_path: Path to .qvdb file or directory for storage
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize Qdrant client in local mode
        self.client = QdrantClient(path=str(self.db_path))
        logger.info(f"Initialized Qdrant local database at {self.db_path}")

    async def create_collection(
        self,
        name: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> None:
        """
        Create a collection if it doesn't exist.

        Args:
            name: Collection name
            vector_size: Size of the vector
            distance: Distance metric (Cosine, Euclidean, or Dot)

        Raises:
            ValueError: If distance metric is invalid
        """
        # Map string distance to Qdrant Distance enum
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclidean": Distance.EUCLID,
            "Dot": Distance.DOT,
        }

        if distance not in distance_map:
            raise ValueError(
                f"Invalid distance metric: {distance}. Must be one of {list(distance_map.keys())}"
            )

        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if name in collection_names:
            logger.debug(f"Collection '{name}' already exists")
            return

        # Create collection with single vector configuration
        self.client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance_map[distance],
            ),
        )
        logger.info(
            f"Created collection '{name}' with vector size {vector_size} and {distance} distance"
        )

    async def create_collection_multivector(
        self,
        name: str,
        vectors_config: dict[str, dict[str, Any]],
    ) -> None:
        """
        Create a collection with multiple named vectors.

        Args:
            name: Collection name
            vectors_config: Dictionary mapping vector names to config dicts.
                Each config must have 'size' and optionally 'distance' (default: Cosine)
                Example: {
                    'text_embedding': {'size': 384, 'distance': 'Cosine'},
                    'clip_embedding': {'size': 512, 'distance': 'Cosine'}
                }
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if name in collection_names:
            logger.debug(f"Collection '{name}' already exists")
            return

        # Build VectorsConfig with named vectors
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclidean": Distance.EUCLID,
            "Dot": Distance.DOT,
        }

        vector_params = {}
        for vector_name, config in vectors_config.items():
            distance_str = config.get("distance", "Cosine")
            if distance_str not in distance_map:
                raise ValueError(f"Invalid distance metric for {vector_name}: {distance_str}")

            vector_params[vector_name] = VectorParams(
                size=config["size"],
                distance=distance_map[distance_str],
            )

        # Create collection
        self.client.create_collection(
            collection_name=name,
            vectors_config=vector_params,
        )
        logger.info(
            f"Created multi-vector collection '{name}' with vectors: {list(vectors_config.keys())}"
        )

    async def upsert(self, collection: str, points: list[dict[str, Any]]) -> None:
        """
        Insert or update points in a collection.

        Args:
            collection: Collection name
            points: List of point dictionaries with 'id', 'vector', and 'payload'
        """
        if not points:
            logger.warning("No points to upsert")
            return

        # Convert dicts to PointStruct objects
        point_structs = [
            PointStruct(
                id=point["id"],
                vector=point["vector"],
                payload=point.get("payload", {}),
            )
            for point in points
        ]

        self.client.upsert(
            collection_name=collection,
            points=point_structs,
        )
        logger.info(f"Upserted {len(points)} points to collection '{collection}'")

    async def search(
        self,
        collection: str,
        vector: list[float],
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        vector_name: str | None = None,
    ) -> list[Any]:
        """
        Perform vector similarity search.

        Args:
            collection: Collection name
            vector: Query vector
            filter: Optional filter conditions (Qdrant filter format)
            limit: Maximum number of results
            vector_name: Name of vector to search (for multi-vector collections)

        Returns:
            List of ScoredPoint objects
        """
        # Build Qdrant filter if provided
        qdrant_filter = None
        if filter:
            qdrant_filter = self._build_filter(filter)

        results = self.client.search(
            collection_name=collection,
            query_vector=vector if vector_name is None else (vector_name, vector),
            query_filter=qdrant_filter,
            limit=limit,
        )

        logger.debug(f"Search in '{collection}' returned {len(results)} results (limit={limit})")
        return cast(list[Any], results)

    async def get(self, collection: str, id: str) -> Any | None:
        """
        Get a single point by ID.

        Args:
            collection: Collection name
            id: Point ID

        Returns:
            PointStruct or None if not found
        """
        try:
            results = self.client.retrieve(
                collection_name=collection,
                ids=[id],
            )
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Error retrieving point {id} from '{collection}': {e}")
            return None

    async def delete(self, collection: str, ids: list[str]) -> None:
        """
        Delete points by IDs.

        Args:
            collection: Collection name
            ids: List of point IDs to delete
        """
        if not ids:
            logger.warning("No IDs to delete")
            return

        self.client.delete(
            collection_name=collection,
            points_selector=ids,
        )
        logger.info(f"Deleted {len(ids)} points from collection '{collection}'")

    async def count(self, collection: str) -> int:
        """
        Count points in a collection.

        Args:
            collection: Collection name

        Returns:
            Number of points
        """
        result = self.client.count(collection_name=collection)
        return cast(int, result.count)

    def _build_filter(self, filter_dict: dict[str, Any]) -> Filter:
        """
        Build Qdrant filter from dictionary.

        Args:
            filter_dict: Dictionary with filter conditions
                Example: {'state_id': 'state123', 'element_type': 'button'}

        Returns:
            Qdrant Filter object
        """
        conditions = []
        for key, value in filter_dict.items():
            conditions.append(
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                )
            )

        return Filter(must=conditions) if conditions else None

    def close(self) -> None:
        """Close the database connection."""
        # Qdrant client doesn't need explicit close in local mode
        logger.info("Closed Qdrant local database")


class RAGIndex:
    """
    High-level interface for indexing and searching GUI elements.

    Manages the GUI element collection with multimodal embeddings.
    """

    COLLECTION_NAME = "gui_elements"

    # Vector configurations
    VECTOR_CONFIGS = {
        "text_embedding": {"size": 384, "distance": "Cosine"},
        "clip_embedding": {"size": 512, "distance": "Cosine"},
        "dinov2_embedding": {"size": 768, "distance": "Cosine"},
    }

    def __init__(self, db: QdrantLocalDB) -> None:
        """
        Initialize RAG index.

        Args:
            db: QdrantLocalDB instance
        """
        self.db = db
        logger.info("Initialized RAG index")

    async def initialize(self) -> None:
        """
        Create the GUI elements collection with proper schema.

        Creates a multi-vector collection supporting:
        - text_embedding: 384-dim (sentence-transformers)
        - clip_embedding: 512-dim (CLIP)
        - dinov2_embedding: 768-dim (DINOv2)
        """
        await self.db.create_collection_multivector(
            name=self.COLLECTION_NAME,
            vectors_config=self.VECTOR_CONFIGS,
        )
        logger.info("Initialized GUI elements collection")

    async def index_elements(self, elements: list[GUIElementChunk]) -> None:
        """
        Index multiple GUI elements.

        Args:
            elements: List of GUIElementChunk objects to index
        """
        if not elements:
            logger.warning("No elements to index")
            return

        # Convert elements to Qdrant points
        points = [element.to_qdrant_point() for element in elements]

        await self.db.upsert(self.COLLECTION_NAME, points)
        logger.info(f"Indexed {len(elements)} GUI elements")

    async def search_by_text(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Search GUI elements by text embedding.

        Args:
            query_embedding: Text embedding vector (384-dim)
            filters: Optional filters (e.g., {'state_id': 'state123'})
            limit: Maximum number of results

        Returns:
            List of SearchResult objects
        """
        results = await self.db.search(
            collection=self.COLLECTION_NAME,
            vector=query_embedding,
            filter=filters,
            limit=limit,
            vector_name="text_embedding",
        )

        return [
            SearchResult(
                element=GUIElementChunk.from_qdrant_point(point),
                score=point.score,
                search_type="text",
            )
            for point in results
        ]

    async def search_by_image(
        self,
        image_embedding: list[float],
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        use_clip: bool = True,
    ) -> list[SearchResult]:
        """
        Search GUI elements by image embedding.

        Args:
            image_embedding: Image embedding vector (512-dim for CLIP, 768-dim for DINOv2)
            filters: Optional filters
            limit: Maximum number of results
            use_clip: Use CLIP embedding (True) or DINOv2 (False)

        Returns:
            List of SearchResult objects
        """
        vector_name = "clip_embedding" if use_clip else "dinov2_embedding"

        results = await self.db.search(
            collection=self.COLLECTION_NAME,
            vector=image_embedding,
            filter=filters,
            limit=limit,
            vector_name=vector_name,
        )

        return [
            SearchResult(
                element=GUIElementChunk.from_qdrant_point(point),
                score=point.score,
                search_type="image_clip" if use_clip else "image_dinov2",
            )
            for point in results
        ]

    async def search_hybrid(
        self,
        text_embedding: list[float],
        image_embedding: list[float],
        filters: dict[str, Any] | None = None,
        text_weight: float = 0.6,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Perform hybrid search combining text and image embeddings.

        Note: This performs two separate searches and combines results.
        For true hybrid search, use Qdrant's query API (requires Qdrant 1.8+).

        Args:
            text_embedding: Text embedding vector (384-dim)
            image_embedding: Image embedding vector (512-dim CLIP)
            filters: Optional filters
            text_weight: Weight for text results (0.0-1.0), image weight = 1 - text_weight
            limit: Maximum number of results

        Returns:
            List of SearchResult objects sorted by combined score
        """
        if not 0.0 <= text_weight <= 1.0:
            raise ValueError("text_weight must be between 0.0 and 1.0")

        image_weight = 1.0 - text_weight

        # Perform both searches
        text_results = await self.db.search(
            collection=self.COLLECTION_NAME,
            vector=text_embedding,
            filter=filters,
            limit=limit * 2,  # Get more results for merging
            vector_name="text_embedding",
        )

        image_results = await self.db.search(
            collection=self.COLLECTION_NAME,
            vector=image_embedding,
            filter=filters,
            limit=limit * 2,
            vector_name="clip_embedding",
        )

        # Combine and re-rank results
        scores: dict[str, tuple[float, Any]] = {}

        for point in text_results:
            point_id = str(point.id)
            scores[point_id] = (point.score * text_weight, point)

        for point in image_results:
            point_id = str(point.id)
            if point_id in scores:
                # Combine scores
                existing_score, existing_point = scores[point_id]
                scores[point_id] = (
                    existing_score + point.score * image_weight,
                    existing_point,
                )
            else:
                scores[point_id] = (point.score * image_weight, point)

        # Sort by combined score and take top K
        sorted_results = sorted(
            scores.values(),
            key=lambda x: x[0],
            reverse=True,
        )[:limit]

        return [
            SearchResult(
                element=GUIElementChunk.from_qdrant_point(point),
                score=score,
                search_type="hybrid",
            )
            for score, point in sorted_results
        ]

    async def get_elements_by_state(self, state_id: str) -> list[GUIElementChunk]:
        """
        Get all GUI elements belonging to a specific state.

        Args:
            state_id: State ID to filter by

        Returns:
            List of GUIElementChunk objects
        """
        # Use scroll to get all points matching filter
        points, _ = self.db.client.scroll(
            collection_name=self.COLLECTION_NAME,
            scroll_filter=self.db._build_filter({"state_id": state_id}),
            limit=10000,  # Large limit to get all elements
        )

        elements = [GUIElementChunk.from_qdrant_point(point) for point in points]
        logger.info(f"Retrieved {len(elements)} elements for state '{state_id}'")
        return elements

    async def delete_elements(self, element_ids: list[str]) -> None:
        """
        Delete GUI elements by IDs.

        Args:
            element_ids: List of element IDs to delete
        """
        await self.db.delete(self.COLLECTION_NAME, element_ids)
        logger.info(f"Deleted {len(element_ids)} GUI elements")

    async def get_element_count(self) -> int:
        """
        Get total number of indexed elements.

        Returns:
            Number of elements in the collection
        """
        return await self.db.count(self.COLLECTION_NAME)

    async def list_all_elements(self, limit: int = 10000) -> list[GUIElementChunk]:
        """
        Get all GUI elements in the collection.

        Args:
            limit: Maximum number of elements to retrieve (default: 10000)

        Returns:
            List of all GUIElementChunk objects
        """
        # Use scroll to get all points without filter
        try:
            points, _ = self.db.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=None,
                limit=limit,
                with_vectors=True,  # Include vectors for matching
            )

            elements = [GUIElementChunk.from_qdrant_point(point) for point in points]
            logger.info(f"Retrieved {len(elements)} total elements from collection")
            return elements
        except Exception as e:
            logger.error(f"Error listing all elements: {e}")
            return []

    async def get_element(self, element_id: str) -> GUIElementChunk | None:
        """
        Get a single GUI element by ID.

        Args:
            element_id: Element ID to retrieve

        Returns:
            GUIElementChunk or None if not found
        """
        point = await self.db.get(self.COLLECTION_NAME, element_id)
        if point:
            return GUIElementChunk.from_qdrant_point(point)
        return None
