"""SceneObjectStore - Manages storage and indexing of semantic objects."""

from dataclasses import dataclass, field

from .semantic_object import ObjectType, SemanticObject


@dataclass
class SceneObjectStore:
    """Manages storage and indexing of semantic objects.

    Responsibilities:
    - Store objects in a list
    - Maintain indices for fast lookup (by ID, by type)
    - Add and remove objects
    - Provide basic access methods

    This class focuses solely on storage and indexing, with no query or analysis logic.
    """

    objects: list[SemanticObject] = field(default_factory=list)
    """List of all semantic objects in storage."""

    _objects_by_id: dict[str, SemanticObject] = field(default_factory=dict, init=False)
    """Index for fast lookup by object ID."""

    _objects_by_type: dict[ObjectType, list[SemanticObject]] = field(
        default_factory=dict, init=False
    )
    """Index for fast lookup by object type."""

    def __post_init__(self):
        """Build indices after initialization."""
        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        """Rebuild all internal indices from the objects list."""
        self._objects_by_id = {obj.id: obj for obj in self.objects}

        self._objects_by_type = {}
        for obj in self.objects:
            if obj.object_type not in self._objects_by_type:
                self._objects_by_type[obj.object_type] = []
            self._objects_by_type[obj.object_type].append(obj)

    def add(self, obj: SemanticObject) -> None:
        """Add an object to storage.

        Args:
            obj: SemanticObject to add
        """
        self.objects.append(obj)
        self._objects_by_id[obj.id] = obj

        if obj.object_type not in self._objects_by_type:
            self._objects_by_type[obj.object_type] = []
        self._objects_by_type[obj.object_type].append(obj)

    def remove(self, object_id: str) -> bool:
        """Remove an object from storage.

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

    def get_by_id(self, object_id: str) -> SemanticObject | None:
        """Get object by ID.

        Args:
            object_id: Object ID to look up

        Returns:
            SemanticObject or None if not found
        """
        return self._objects_by_id.get(object_id)

    def get_by_type(self, object_type: ObjectType) -> list[SemanticObject]:
        """Get all objects of a specific type.

        Args:
            object_type: ObjectType to retrieve

        Returns:
            List of objects with that type (empty list if none)
        """
        return self._objects_by_type.get(object_type, [])

    def get_all(self) -> list[SemanticObject]:
        """Get all objects.

        Returns:
            List of all objects
        """
        return list(self.objects)

    def get_type_counts(self) -> dict[ObjectType, int]:
        """Get count of objects by type.

        Returns:
            Dictionary mapping ObjectType to count
        """
        return {obj_type: len(objs) for obj_type, objs in self._objects_by_type.items()}

    def count(self) -> int:
        """Get total object count.

        Returns:
            Number of objects in storage
        """
        return len(self.objects)

    def clear(self) -> None:
        """Clear all objects from storage."""
        self.objects.clear()
        self._objects_by_id.clear()
        self._objects_by_type.clear()

    def __len__(self) -> int:
        """Get object count."""
        return len(self.objects)

    def __repr__(self) -> str:
        """Developer representation."""
        return f"SceneObjectStore(objects={len(self.objects)})"
