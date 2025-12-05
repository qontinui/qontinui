"""SceneAnalyzer - Performs analysis operations on semantic scenes."""


from typing import Any, cast

import numpy as np

from .scene_object_store import SceneObjectStore
from .semantic_object import SemanticObject


class SceneAnalyzer:
    """Performs analysis operations on semantic scenes.

    Responsibilities:
    - Generate scene descriptions
    - Calculate similarity between scenes
    - Find differences between scenes
    - Analyze hierarchy relationships

    This class performs complex analysis operations on scene data.
    """

    def __init__(self, store: SceneObjectStore) -> None:
        """Initialize with an object store.

        Args:
            store: SceneObjectStore to analyze
        """
        self.store = store

    def generate_description(self) -> str:
        """Generate natural language description of the scene.

        Returns:
            Text description of the scene
        """
        objects = self.store.get_all()
        if not objects:
            return "Empty scene with no detected objects."

        type_counts = self.store.get_type_counts()

        description = f"Scene contains {len(objects)} objects:\n"

        for obj_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            if count > 1:
                description += f"- {count} {obj_type.value}s\n"
            else:
                description += f"- {count} {obj_type.value}\n"

        # Add information about text content
        text_objects = [obj for obj in objects if obj.get_text()]
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

    def calculate_hierarchy(self) -> list[tuple[SemanticObject, list[SemanticObject]]]:
        """Analyze parent-child relationships based on containment.

        Returns:
            List of (parent, children) tuples
        """
        objects = self.store.get_all()
        hierarchy = []
        processed = set()

        # Sort by area (largest first) to find top-level containers
        sorted_objects = sorted(objects, key=lambda o: o.location.get_area(), reverse=True)

        for potential_parent in sorted_objects:
            if potential_parent.id in processed:
                continue

            children = []
            for other in objects:
                if other.id != potential_parent.id and potential_parent.contains(other):
                    children.append(other)
                    processed.add(other.id)

            if children:
                hierarchy.append((potential_parent, children))
                processed.add(potential_parent.id)

        return hierarchy

    def calculate_similarity(self, other_store: SceneObjectStore) -> float:
        """Calculate similarity to another scene.

        Uses object type distribution and spatial layout similarity.

        Args:
            other_store: Other scene's object store to compare

        Returns:
            Similarity score (0.0 to 1.0)
        """
        self_objects = self.store.get_all()
        other_objects = other_store.get_all()

        if not self_objects and not other_objects:
            return 1.0
        if not self_objects or not other_objects:
            return 0.0

        # Compare object type distributions
        self_types = self.store.get_type_counts()
        other_types = other_store.get_type_counts()

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
        grid_similarity = self._calculate_grid_similarity(other_store)

        # Weighted average
        return 0.6 * type_similarity + 0.4 * grid_similarity

    def _calculate_grid_similarity(
        self, other_store: SceneObjectStore, grid_size: int = 10
    ) -> float:
        """Calculate spatial similarity using grid-based approach.

        Args:
            other_store: Other scene's object store
            grid_size: Grid division size

        Returns:
            Grid similarity score (0.0 to 1.0)
        """
        # Assume 1920x1080 screen (could be parameterized)
        screen_width, screen_height = 1920, 1080
        cell_width = screen_width // grid_size
        cell_height = screen_height // grid_size

        def get_grid_distribution(store: SceneObjectStore) -> np.ndarray[Any, Any]:
            grid = np.zeros((grid_size, grid_size))
            for obj in store.get_all():
                centroid = obj.location.get_centroid()
                x_cell = min(centroid.x // cell_width, grid_size - 1)
                y_cell = min(centroid.y // cell_height, grid_size - 1)
                grid[y_cell, x_cell] += 1
            return grid / (np.sum(grid) + 1e-10)  # Normalize

        self_grid = get_grid_distribution(self.store)
        other_grid = get_grid_distribution(other_store)

        # Calculate cosine similarity
        flat_self = self_grid.flatten()
        flat_other = other_grid.flatten()

        dot_product = np.dot(flat_self, flat_other)
        norm_self = np.linalg.norm(flat_self)
        norm_other = np.linalg.norm(flat_other)

        if norm_self == 0 or norm_other == 0:
            return 0.0

        return cast(float, dot_product / (norm_self * norm_other))

    def find_differences(
        self, other_store: SceneObjectStore
    ) -> dict[str, list[SemanticObject | tuple[SemanticObject, SemanticObject]]]:
        """Find objects that differ between scenes.

        Args:
            other_store: Other scene's object store to compare

        Returns:
            Dictionary with 'added', 'removed', and 'changed' objects
        """
        differences: dict[str, list[Any]] = {"added": [], "removed": [], "changed": []}

        # Build lookup by approximate location and type
        def get_object_key(obj: SemanticObject) -> tuple[Any, ...]:
            centroid = obj.location.get_centroid()
            # Round to nearest 10 pixels for fuzzy matching
            return (centroid.x // 10, centroid.y // 10, obj.object_type)

        self_lookup = {get_object_key(obj): obj for obj in self.store.get_all()}
        other_lookup = {get_object_key(obj): obj for obj in other_store.get_all()}

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
