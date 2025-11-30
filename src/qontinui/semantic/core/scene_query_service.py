"""SceneQueryService - Handles all query operations on semantic scenes."""

from __future__ import annotations

import re

from ...model.element.region import Region
from .pixel_location import Point
from .scene_object_store import SceneObjectStore
from .semantic_object import ObjectType, SemanticObject


class SceneQueryService:
    """Handles all query operations on semantic scenes.

    Responsibilities:
    - Find objects by description, type, or region
    - Spatial queries (closest, above, below, left, right, near)
    - Filter objects by properties (interactable, with text)

    This class is stateless and performs queries on a provided SceneObjectStore.
    """

    def __init__(self, store: SceneObjectStore) -> None:
        """Initialize with an object store.

        Args:
            store: SceneObjectStore to query
        """
        self.store = store

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
            return [obj for obj in self.store.get_all() if regex.search(obj.description)]
        except re.error:
            # Fall back to substring matching if regex is invalid
            if not case_sensitive:
                pattern = pattern.lower()
                return [obj for obj in self.store.get_all() if pattern in obj.description.lower()]
            return [obj for obj in self.store.get_all() if pattern in obj.description]

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
                    obj
                    for obj in self.store.get_all()
                    if obj.get_attribute("custom_type") == object_type
                ]

        return self.store.get_by_type(object_type)

    def find_in_region(self, region: Region) -> list[SemanticObject]:
        """Find objects within a region.

        Args:
            region: Region to search within

        Returns:
            List of objects whose bounding boxes overlap with the region
        """
        results = []
        for obj in self.store.get_all():
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
            Closest SemanticObject or None if no objects
        """
        objects = self.store.get_all()
        if not objects:
            return None

        if isinstance(point, tuple):
            point = Point(point[0], point[1])

        min_distance = float("inf")
        closest = None

        for obj in objects:
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
        return [obj for obj in self.store.get_all() if obj.is_interactable()]

    def find_with_text(self) -> list[SemanticObject]:
        """Find all objects containing text.

        Returns:
            List of objects with extracted text
        """
        return [obj for obj in self.store.get_all() if obj.get_text()]

    def get_objects_above(self, reference: SemanticObject) -> list[SemanticObject]:
        """Get all objects above a reference object.

        Args:
            reference: Reference object

        Returns:
            List of objects above the reference
        """
        return [obj for obj in self.store.get_all() if obj.is_above(reference)]

    def get_objects_below(self, reference: SemanticObject) -> list[SemanticObject]:
        """Get all objects below a reference object.

        Args:
            reference: Reference object

        Returns:
            List of objects below the reference
        """
        return [obj for obj in self.store.get_all() if obj.is_below(reference)]

    def get_objects_left_of(self, reference: SemanticObject) -> list[SemanticObject]:
        """Get all objects to the left of a reference object.

        Args:
            reference: Reference object

        Returns:
            List of objects to the left of the reference
        """
        return [obj for obj in self.store.get_all() if obj.is_left_of(reference)]

    def get_objects_right_of(self, reference: SemanticObject) -> list[SemanticObject]:
        """Get all objects to the right of a reference object.

        Args:
            reference: Reference object

        Returns:
            List of objects to the right of the reference
        """
        return [obj for obj in self.store.get_all() if obj.is_right_of(reference)]

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
        for obj in self.store.get_all():
            if obj.id != reference.id:
                distance = reference.distance_to(obj)
                if distance <= max_distance:
                    nearby.append(obj)
        return nearby
