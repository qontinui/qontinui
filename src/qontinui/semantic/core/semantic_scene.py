"""SemanticScene - Orchestrator for semantic scene operations.

This class has been refactored to delegate responsibilities to focused components:
- SceneObjectStore: Object storage and indexing
- SceneQueryService: Query operations
- SceneAnalyzer: Analysis operations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from ...model.element.region import Region
from .pixel_location import Point
from .scene_analyzer import SceneAnalyzer
from .scene_object_store import SceneObjectStore
from .scene_query_service import SceneQueryService
from .semantic_object import ObjectType, SemanticObject


@dataclass
class SemanticScene:
    """Orchestrator for semantic scene operations.

    This class delegates to focused components for different responsibilities:
    - SceneObjectStore: Manages object storage and indexing
    - SceneQueryService: Handles all query operations
    - SceneAnalyzer: Performs analysis and comparison

    The scene acts as a facade providing a unified interface to these services.
    """

    source_image: np.ndarray[Any, Any] | None = field(default=None)
    """Source image this scene was extracted from."""

    timestamp: datetime = field(default_factory=datetime.now)
    """When this scene was created."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional scene metadata."""

    _store: SceneObjectStore = field(default_factory=SceneObjectStore, init=False)
    """Object storage and indexing."""

    _query: SceneQueryService = field(init=False)
    """Query service."""

    _analyzer: SceneAnalyzer = field(init=False)
    """Analysis service."""

    def __post_init__(self):
        """Initialize services."""
        self._query = SceneQueryService(self._store)
        self._analyzer = SceneAnalyzer(self._store)

    @property
    def objects(self) -> list[SemanticObject]:
        """Get all objects in the scene.

        Returns:
            List of all semantic objects
        """
        return self._store.get_all()

    def add_object(self, obj: SemanticObject) -> None:
        """Add an object to the scene.

        Args:
            obj: SemanticObject to add
        """
        self._store.add(obj)

    def remove_object(self, object_id: str) -> bool:
        """Remove an object from the scene.

        Args:
            object_id: ID of object to remove

        Returns:
            True if object was found and removed
        """
        return self._store.remove(object_id)

    def get_object_by_id(self, object_id: str) -> SemanticObject | None:
        """Get object by its ID.

        Args:
            object_id: Object ID

        Returns:
            SemanticObject or None if not found
        """
        return self._store.get_by_id(object_id)

    def find_by_description(
        self, pattern: str, case_sensitive: bool = False
    ) -> list[SemanticObject]:
        """Find objects by description pattern.

        Args:
            pattern: Regex pattern or substring to search for
            case_sensitive: Whether to use case-sensitive matching

        Returns:
            List of matching objects
        """
        return self._query.find_by_description(pattern, case_sensitive)

    def find_by_type(self, object_type: ObjectType | str) -> list[SemanticObject]:
        """Find objects by type.

        Args:
            object_type: ObjectType enum or string type name

        Returns:
            List of matching objects
        """
        return self._query.find_by_type(object_type)

    def find_in_region(self, region: Region) -> list[SemanticObject]:
        """Find objects within a region.

        Args:
            region: Region to search within

        Returns:
            List of objects whose bounding boxes overlap with the region
        """
        return self._query.find_in_region(region)

    def find_closest_to(self, point: Point | tuple[int, int]) -> SemanticObject | None:
        """Find the object closest to a point.

        Args:
            point: Point or (x, y) tuple

        Returns:
            Closest SemanticObject or None if scene is empty
        """
        return self._query.find_closest_to(point)

    def find_interactable(self) -> list[SemanticObject]:
        """Find all interactable objects.

        Returns:
            List of interactable objects
        """
        return self._query.find_interactable()

    def find_with_text(self) -> list[SemanticObject]:
        """Find all objects containing text.

        Returns:
            List of objects with extracted text
        """
        return self._query.find_with_text()

    def get_objects_above(self, reference: SemanticObject) -> list[SemanticObject]:
        """Get all objects above a reference object.

        Args:
            reference: Reference object

        Returns:
            List of objects above the reference
        """
        return self._query.get_objects_above(reference)

    def get_objects_below(self, reference: SemanticObject) -> list[SemanticObject]:
        """Get all objects below a reference object.

        Args:
            reference: Reference object

        Returns:
            List of objects below the reference
        """
        return self._query.get_objects_below(reference)

    def get_objects_left_of(self, reference: SemanticObject) -> list[SemanticObject]:
        """Get all objects to the left of a reference object.

        Args:
            reference: Reference object

        Returns:
            List of objects to the left of the reference
        """
        return self._query.get_objects_left_of(reference)

    def get_objects_right_of(self, reference: SemanticObject) -> list[SemanticObject]:
        """Get all objects to the right of a reference object.

        Args:
            reference: Reference object

        Returns:
            List of objects to the right of the reference
        """
        return self._query.get_objects_right_of(reference)

    def get_objects_near(
        self, reference: SemanticObject, max_distance: float
    ) -> list[SemanticObject]:
        """Get objects within a certain distance of a reference.

        Args:
            reference: Reference object
            max_distance: Maximum distance in pixels

        Returns:
            List of nearby objects
        """
        return self._query.get_objects_near(reference, max_distance)

    def generate_scene_description(self) -> str:
        """Generate natural language description of the scene.

        Returns:
            Text description of the scene
        """
        return self._analyzer.generate_description()

    def get_object_type_count(self) -> dict[ObjectType, int]:
        """Get count of each object type.

        Returns:
            Dictionary mapping ObjectType to count
        """
        return self._store.get_type_counts()

    def get_hierarchy(self) -> list[tuple[SemanticObject, list[SemanticObject]]]:
        """Analyze parent-child relationships based on containment.

        Returns:
            List of (parent, children) tuples
        """
        return self._analyzer.calculate_hierarchy()

    def similarity_to(self, other: SemanticScene) -> float:
        """Calculate similarity to another scene.

        Uses object type distribution and spatial layout similarity.

        Args:
            other: Other scene to compare

        Returns:
            Similarity score (0.0 to 1.0)
        """
        return self._analyzer.calculate_similarity(other._store)

    def find_differences(
        self, other: SemanticScene
    ) -> dict[str, list[SemanticObject | tuple[SemanticObject, SemanticObject]]]:
        """Find objects that differ between scenes.

        Args:
            other: Other scene to compare

        Returns:
            Dictionary with 'added', 'removed', and 'changed' objects
        """
        return self._analyzer.find_differences(other._store)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with scene properties
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "object_count": self._store.count(),
            "object_types": {
                k.value: v for k, v in self.get_object_type_count().items()
            },
            "objects": [obj.to_dict() for obj in self.objects],
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SemanticScene(objects={self._store.count()}, "
            f"timestamp={self.timestamp.strftime('%Y-%m-%d %H:%M:%S')})"
        )

    def __str__(self) -> str:
        """Human-readable string."""
        return self.generate_scene_description()
