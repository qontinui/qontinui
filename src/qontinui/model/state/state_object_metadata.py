"""StateObjectMetadata class - ported from Qontinui framework.

Lightweight reference to StateObject instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .state_object import StateObjectType

if TYPE_CHECKING:
    from .state_object import StateObject


@dataclass
class StateObjectMetadata:
    """Lightweight reference to StateObject instances.

    Port of StateObjectMetadata from Qontinui framework class.

    StateObjectMetadata provides a minimal, serializable representation of StateObject identity
    without the full object graph. This design pattern solves critical architectural challenges
    around circular dependencies and persistence while maintaining the ability to reference
    state objects throughout the framework.

    Key design benefits:
    - Prevents Circular Dependencies: Avoids infinite object graphs that would occur
      if Match objects contained full StateObject references
    - Persistence Friendly: Can be embedded in entities without complex mappings,
      as StateObjects are entities while this is embeddable
    - Lightweight References: Minimal memory footprint for object references
    - Repository Pattern Support: Contains sufficient data to retrieve full objects
      from their respective repositories

    Reference data captured:
    - Object Identity: Unique ID for repository lookup
    - Object Type: Specifies which repository to query (IMAGE, REGION, etc.)
    - Object Name: Human-readable identifier for debugging
    - Owner State: Both name and ID of the containing state

    Common usage patterns:
    - Stored in Match objects to track which StateObject was found
    - Used in action results to reference involved state objects
    - Enables lazy loading of full StateObjects when needed
    - Facilitates cross-reference tracking without object coupling
    """

    state_object_id: str = ""
    object_type: StateObjectType = StateObjectType.IMAGE
    state_object_name: str = ""
    owner_state_name: str = ""
    owner_state_id: int | None = None

    def __init__(self, state_object: StateObject | None = None) -> None:
        """Initialize metadata from StateObject or with defaults.

        Args:
            state_object: Optional StateObject to extract metadata from
        """
        if state_object:
            self.state_object_id = state_object.get_id_as_string()
            self.object_type = state_object.get_object_type()
            self.state_object_name = getattr(state_object, "name", "")
            self.owner_state_name = state_object.get_owner_state_name()
            self.owner_state_id = state_object.get_owner_state_id()
        else:
            self.state_object_id = ""
            self.object_type = StateObjectType.IMAGE
            self.state_object_name = ""
            self.owner_state_name = ""
            self.owner_state_id = None

    def is_valid(self) -> bool:
        """Check if metadata contains valid reference.

        Returns:
            True if has valid object ID
        """
        return bool(self.state_object_id)

    def is_image(self) -> bool:
        """Check if references an image object.

        Returns:
            True if type is IMAGE
        """
        return self.object_type == StateObjectType.IMAGE

    def is_region(self) -> bool:
        """Check if references a region object.

        Returns:
            True if type is REGION
        """
        return self.object_type == StateObjectType.REGION

    def is_location(self) -> bool:
        """Check if references a location object.

        Returns:
            True if type is LOCATION
        """
        return self.object_type == StateObjectType.LOCATION

    def is_string(self) -> bool:
        """Check if references a string object.

        Returns:
            True if type is STRING
        """
        return self.object_type == StateObjectType.STRING

    def matches_state(self, state_name: str) -> bool:
        """Check if object belongs to specified state.

        Args:
            state_name: Name of state to check

        Returns:
            True if object belongs to state
        """
        return self.owner_state_name == state_name

    def __str__(self) -> str:
        """String representation."""
        return (
            f"StateObject: {self.state_object_name}, {self.object_type.value}, "
            f"ownerState={self.owner_state_name}, id={self.state_object_id}, "
            f"owner state id={self.owner_state_id}"
        )

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"StateObjectMetadata(id='{self.state_object_id}', "
            f"type={self.object_type}, name='{self.state_object_name}', "
            f"owner='{self.owner_state_name}')"
        )

    @classmethod
    def empty(cls) -> StateObjectMetadata:
        """Create empty metadata.

        Returns:
            Empty StateObjectMetadata instance
        """
        return cls()
