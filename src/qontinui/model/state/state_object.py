"""StateObject interface and metadata - ported from Qontinui framework.

Core interface for all objects that belong to states in the framework.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class StateObjectType(Enum):
    """Types of state objects.

    Port of StateObject from Qontinui framework.Type enum.
    """

    IMAGE = "IMAGE"
    REGION = "REGION"
    LOCATION = "LOCATION"
    STRING = "STRING"
    TEXT = "TEXT"


@runtime_checkable
class StateObject(Protocol):
    """Core interface for all objects that belong to states.

    Port of StateObject from Qontinui framework interface.

    StateObject defines the contract for all elements that can be contained within a State,
    establishing a unified interface for diverse object types while maintaining their association
    with parent states. This abstraction is fundamental to the framework's ability to treat
    different GUI elements polymorphically while preserving their state context.

    Object type hierarchy:
    - IMAGE: Visual patterns for recognition (StateImage)
    - REGION: Defined screen areas (StateRegion)
    - LOCATION: Specific screen coordinates (StateLocation)
    - STRING: Text strings for typing or validation (StateString)
    - TEXT: Expected text patterns (StateText)

    Core responsibilities:
    - Identity: Provides unique identification as string for persistence
    - Type Declaration: Declares the specific object type for proper handling
    - Naming: Human-readable name for debugging and logging
    - State Association: Maintains reference to owning state by name and ID
    - Usage Tracking: Records how many times the object has been acted upon

    MatchHistory integration:
    - StateObjects maintain MatchHistory to record action results
    - Snapshots capture success/failure patterns for mock execution
    - Historical data enables learning and optimization
    - Usage statistics inform automation strategy adjustments

    In the model-based approach, StateObject enables the framework to handle diverse GUI
    elements uniformly while maintaining their state context. This polymorphic design allows
    actions to operate on different object types without knowing their specific implementation,
    while the state association ensures proper scoping and context awareness.
    """

    def get_id_as_string(self) -> str:
        """Get the unique ID as a string.

        Returns:
            String representation of the ID
        """
        ...

    def get_object_type(self) -> StateObjectType:
        """Get the object type.

        Returns:
            The type of this state object
        """
        ...

    def get_name(self) -> str:
        """Get the name of this object.

        Returns:
            Human-readable name
        """
        ...

    def get_owner_state_name(self) -> str:
        """Get the name of the owner state.

        Returns:
            Owner state name
        """
        ...

    def get_owner_state_id(self) -> int | None:
        """Get the ID of the owner state.

        Returns:
            Owner state ID or None
        """
        ...

    def add_times_acted_on(self) -> None:
        """Increment the times acted on counter."""
        ...

    def set_times_acted_on(self, times: int) -> None:
        """Set the times acted on counter.

        Args:
            times: Number of times acted on
        """
        ...


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

    In the model-based approach, StateObjectMetadata enables the framework to maintain rich
    cross-references between matches, actions, and state objects without the complexity and
    performance overhead of full object graphs. This is essential for scalable automation
    that can handle complex state structures with many interconnected elements.
    """

    state_object_id: str = ""
    """Unique ID for repository lookup."""

    object_type: StateObjectType = StateObjectType.IMAGE
    """Type of the state object."""

    state_object_name: str = ""
    """Human-readable name."""

    owner_state_name: str = ""
    """Name of the containing state."""

    owner_state_id: int | None = None
    """ID of the containing state."""

    @classmethod
    def from_state_object(cls, state_object: StateObject) -> "StateObjectMetadata":
        """Create metadata from a StateObject.

        Args:
            state_object: StateObject to extract metadata from

        Returns:
            StateObjectMetadata instance
        """
        return cls(
            state_object_id=state_object.get_id_as_string(),
            object_type=state_object.get_object_type(),
            state_object_name=state_object.get_name(),
            owner_state_name=state_object.get_owner_state_name(),
            owner_state_id=state_object.get_owner_state_id(),
        )

    def __str__(self) -> str:
        """String representation."""
        return (
            f"StateObject: {self.state_object_name}, {self.object_type}, "
            f"ownerState={self.owner_state_name}, id={self.state_object_id}, "
            f"owner state id={self.owner_state_id}"
        )
