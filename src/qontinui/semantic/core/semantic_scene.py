"""SemanticScene - Container for all semantic objects discovered in a screenshot."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast

import numpy as np

from ...model.element.region import Region
from .pixel_location import Point
from .semantic_object import ObjectType, SemanticObject


@dataclass
class SemanticScene:
    """Container for all semantic objects discovered in a screenshot.

    Provides methods for querying, analyzing, and comparing semantic scenes.
    Acts as the primary data structure for semantic understanding of GUI states.
    """

    source_image: np.ndarray[Any, Any] | None = field(default=None)
    """Source image this scene was extracted from."""

    objects: list[SemanticObject] = field(default_factory=list)
    """List of all semantic objects in the scene."""

    timestamp: datetime = field(default_factory=datetime.now)
    """When this scene was created."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional scene metadata."""

    _objects_by_id: dict[str, SemanticObject] = field(default_factory=dict, init=False)
    _objects_by_type: dict[ObjectType, list[SemanticObject]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self):
        """Index objects after initialization."""
        self._reindex_objects()

    def _reindex_objects(self):
        """Rebuild internal indices."""
        self._objects_by_id = {obj.id: obj for obj in self.objects}

        self._objects_by_type = {}
        for obj in self.objects:
            if obj.object_type not in self._objects_by_type:
                self._objects_by_type[obj.object_type] = []
            self._objects_by_type[obj.object_type].append(obj)

    def add_object(self, obj: SemanticObject) -> None:
        """Add an object to the scene.

        Args:
            obj: SemanticObject to add
        """
        self.objects.append(obj)
        self._objects_by_id[obj.id] = obj

        if obj.object_type not in self._objects_by_type:
            self._objects_by_type[obj.object_type] = []
        self._objects_by_type[obj.object_type].append(obj)

    def remove_object(self, object_id: str) -> bool:
        """Remove an object from the scene.

        Args:
            object_id: ID of object to remove

        Returns:
            True if object was found and removed
        """
        if object_id not in self._objects_by_id:
            return False

        obj = self._objects_by_id[object_id]
        self.objects.remove(obj)
        del self._objects_by_id[object_id]

        if obj.object_type in self._objects_by_type:
            self._objects_by_type[obj.object_type].remove(obj)
            if not self._objects_by_type[obj.object_type]:
                del self._objects_by_type[obj.object_type]

        return True

    def get_object_by_id(self, object_id: str) -> SemanticObject | None:
        """Get object by its ID.

        Args:
            object_id: Object ID

        Returns:
            SemanticObject or None if not found
        """
        return self._objects_by_id.get(object_id)

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
        flags = 0 if case_sensitive else re.IGNORECASE

        try:
            regex = re.compile(pattern, flags)
            return [obj for obj in self.objects if regex.search(obj.description)]
        except re.error:
            # Fall back to substring matching if regex is invalid
            if not case_sensitive:
                pattern = pattern.lower()
                return [obj for obj in self.objects if pattern in obj.description.lower()]
            return [obj for obj in self.objects if pattern in obj.description]

    def find_by_type(self, object_type: ObjectType | str) -> list[SemanticObject]:
        """Find objects by type.

        Args:
            object_type: ObjectType enum or string type name

        Returns:
            List of matching objects
        """
        if isinstance(object_type, str):
            try:
                object_type = ObjectType(object_type.lower())
            except ValueError:
                # Check custom types
                return [
                    obj for obj in self.objects if obj.get_attribute("custom_type") == object_type
                ]

        return self._objects_by_type.get(object_type, [])

    def find_in_region(self, region: Region) -> list[SemanticObject]:
        """Find objects within a region.

        Args:
            region: Region to search within

        Returns:
            List of objects whose bounding boxes overlap with the region
        """
        results = []
        for obj in self.objects:
            box = obj.get_bounding_box()
            if (
                box.x < region.x + region.width
                and box.x + box.width > region.x
                and box.y < region.y + region.height
                and box.y + box.height > region.y
            ):
                results.append(obj)
        return results

    def find_closest_to(self, point: Point | tuple[int, int]) -> SemanticObject | None:
        """Find the object closest to a point.

        Args:
            point: Point or (x, y) tuple

        Returns:
            Closest SemanticObject or None if scene is empty
        """
        if not self.objects:
            return None

        if isinstance(point, tuple):
            point = Point(point[0], point[1])

        min_distance = float("inf")
        closest = None

        for obj in self.objects:
            centroid = obj.location.get_centroid()
            distance = ((centroid.x - point.x) ** 2 + (centroid.y - point.y) ** 2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                closest = obj

        return closest

    def find_interactable(self) -> list[SemanticObject]:
        """Find all interactable objects.

        Returns:
            List of interactable objects
        """
        return [obj for obj in self.objects if obj.is_interactable()]

    def find_with_text(self) -> list[SemanticObject]:
        """Find all objects containing text.

        Returns:
            List of objects with extracted text
        """
        return [obj for obj in self.objects if obj.get_text()]

    def get_objects_above(self, reference: SemanticObject) -> list[SemanticObject]:
        """Get all objects above a reference object.

        Args:
            reference: Reference object

        Returns:
            List of objects above the reference
        """
        return [obj for obj in self.objects if obj.is_above(reference)]

    def get_objects_below(self, reference: SemanticObject) -> list[SemanticObject]:
        """Get all objects below a reference object.

        Args:
            reference: Reference object

        Returns:
            List of objects below the reference
        """
        return [obj for obj in self.objects if obj.is_below(reference)]

    def get_objects_left_of(self, reference: SemanticObject) -> list[SemanticObject]:
        """Get all objects to the left of a reference object.

        Args:
            reference: Reference object

        Returns:
            List of objects to the left of the reference
        """
        return [obj for obj in self.objects if obj.is_left_of(reference)]

    def get_objects_right_of(self, reference: SemanticObject) -> list[SemanticObject]:
        """Get all objects to the right of a reference object.

        Args:
            reference: Reference object

        Returns:
            List of objects to the right of the reference
        """
        return [obj for obj in self.objects if obj.is_right_of(reference)]

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
        nearby = []
        for obj in self.objects:
            if obj.id != reference.id:
                distance = reference.distance_to(obj)
                if distance <= max_distance:
                    nearby.append(obj)
        return nearby

    def generate_scene_description(self) -> str:
        """Generate natural language description of the scene.

        Returns:
            Text description of the scene
        """
        if not self.objects:
            return "Empty scene with no detected objects."

        type_counts = self.get_object_type_count()

        description = f"Scene contains {len(self.objects)} objects:\n"

        for obj_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            if count > 1:
                description += f"- {count} {obj_type.value}s\n"
            else:
                description += f"- {count} {obj_type.value}\n"

        # Add information about text content
        text_objects = self.find_with_text()
        if text_objects:
            description += f"\nText content found in {len(text_objects)} objects:\n"
            for obj in text_objects[:5]:  # Show first 5
                text = obj.get_text()
                if text and len(text) > 50:
                    text = text[:50] + "..."
                description += f"- {text}\n"
            if len(text_objects) > 5:
                description += f"... and {len(text_objects) - 5} more\n"

        return description

    def get_object_type_count(self) -> dict[ObjectType, int]:
        """Get count of each object type.

        Returns:
            Dictionary mapping ObjectType to count
        """
        return {obj_type: len(objs) for obj_type, objs in self._objects_by_type.items()}

    def get_hierarchy(self) -> list[tuple[SemanticObject, list[SemanticObject]]]:
        """Analyze parent-child relationships based on containment.

        Returns:
            List of (parent, children) tuples
        """
        hierarchy = []
        processed = set()

        # Sort by area (largest first) to find top-level containers
        sorted_objects = sorted(self.objects, key=lambda o: o.location.get_area(), reverse=True)

        for potential_parent in sorted_objects:
            if potential_parent.id in processed:
                continue

            children = []
            for other in self.objects:
                if other.id != potential_parent.id and potential_parent.contains(other):
                    children.append(other)
                    processed.add(other.id)

            if children:
                hierarchy.append((potential_parent, children))
                processed.add(potential_parent.id)

        return hierarchy

    def similarity_to(self, other: SemanticScene) -> float:
        """Calculate similarity to another scene.

        Uses object type distribution and spatial layout similarity.

        Args:
            other: Other scene to compare

        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not self.objects and not other.objects:
            return 1.0
        if not self.objects or not other.objects:
            return 0.0

        # Compare object type distributions
        self_types = self.get_object_type_count()
        other_types = other.get_object_type_count()

        all_types = set(self_types.keys()) | set(other_types.keys())
        if not all_types:
            return 0.0

        type_similarity = 0.0
        for obj_type in all_types:
            self_count = self_types.get(obj_type, 0)
            other_count = other_types.get(obj_type, 0)

            if self_count == other_count:
                type_similarity += 1.0
            elif self_count == 0 or other_count == 0:
                type_similarity += 0.0
            else:
                type_similarity += min(self_count, other_count) / max(self_count, other_count)

        type_similarity /= len(all_types)

        # Compare spatial distribution (using grid-based approach)
        grid_similarity = self._calculate_grid_similarity(other)

        # Weighted average
        return 0.6 * type_similarity + 0.4 * grid_similarity

    def _calculate_grid_similarity(self, other: SemanticScene, grid_size: int = 10) -> float:
        """Calculate spatial similarity using grid-based approach.

        Args:
            other: Other scene
            grid_size: Grid division size

        Returns:
            Grid similarity score (0.0 to 1.0)
        """
        # Assume 1920x1080 screen (could be parameterized)
        screen_width, screen_height = 1920, 1080
        cell_width = screen_width // grid_size
        cell_height = screen_height // grid_size

        def get_grid_distribution(scene: SemanticScene) -> np.ndarray[Any, Any]:
            grid = np.zeros((grid_size, grid_size))
            for obj in scene.objects:
                centroid = obj.location.get_centroid()
                x_cell = min(centroid.x // cell_width, grid_size - 1)
                y_cell = min(centroid.y // cell_height, grid_size - 1)
                grid[y_cell, x_cell] += 1
            return grid / (np.sum(grid) + 1e-10)  # Normalize

        self_grid = get_grid_distribution(self)
        other_grid = get_grid_distribution(other)

        # Calculate cosine similarity
        flat_self = self_grid.flatten()
        flat_other = other_grid.flatten()

        dot_product = np.dot(flat_self, flat_other)
        norm_self = np.linalg.norm(flat_self)
        norm_other = np.linalg.norm(flat_other)

        if norm_self == 0 or norm_other == 0:
            return 0.0

        return cast(float, dot_product / (norm_self * norm_other))

    def find_differences(self, other: SemanticScene) -> dict[str, list[SemanticObject]]:
        """Find objects that differ between scenes.

        Args:
            other: Other scene to compare

        Returns:
            Dictionary with 'added', 'removed', and 'changed' objects
        """
        differences: dict[str, list[Any]] = {"added": [], "removed": [], "changed": []}

        # Build lookup by approximate location and type
        def get_object_key(obj: SemanticObject) -> tuple[Any, ...]:
            centroid = obj.location.get_centroid()
            # Round to nearest 10 pixels for fuzzy matching
            return (centroid.x // 10, centroid.y // 10, obj.object_type)

        self_lookup = {get_object_key(obj): obj for obj in self.objects}
        other_lookup = {get_object_key(obj): obj for obj in other.objects}

        # Find removed objects
        for key, obj in self_lookup.items():
            if key not in other_lookup:
                differences["removed"].append(obj)

        # Find added objects
        for key, obj in other_lookup.items():
            if key not in self_lookup:
                differences["added"].append(obj)

        # Find changed objects
        for key in set(self_lookup.keys()) & set(other_lookup.keys()):
            self_obj = self_lookup[key]
            other_obj = other_lookup[key]

            # Check if attributes changed significantly
            if (
                self_obj.description != other_obj.description
                or abs(self_obj.confidence - other_obj.confidence) > 0.2
                or self_obj.get_text() != other_obj.get_text()
            ):
                differences["changed"].append((self_obj, other_obj))

        return differences

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with scene properties
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "object_count": len(self.objects),
            "object_types": {k.value: v for k, v in self.get_object_type_count().items()},
            "objects": [obj.to_dict() for obj in self.objects],
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SemanticScene(objects={len(self.objects)}, "
            f"timestamp={self.timestamp.strftime('%Y-%m-%d %H:%M:%S')})"
        )

    def __str__(self) -> str:
        """Human-readable string."""
        return self.generate_scene_description()
