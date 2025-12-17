"""Transition data model for video capture architecture.

This module defines the Transition model, which represents a state transition
triggered by a user action. It captures both the outgoing action (what the user
did) and the incoming recognition criteria (how to identify the target state).

Transitions form the edges in the state machine graph, connecting DetectedStates
and defining the actions needed to navigate between them.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Transition:
    """Represents a state transition triggered by a user action.

    A Transition captures the complete context of a state change, including:
    - What states were active before and after
    - What action triggered the transition (outgoing/action side)
    - How to recognize the target state (incoming/recognition side)
    - Timing and frame information

    This model supports multi-state scenarios where multiple states can be
    active simultaneously (e.g., multiple windows, overlays, etc.).

    Attributes:
        id: Unique identifier for this transition
        source_states: List of state IDs active before the action
        target_states: List of state IDs active after the action
        states_appeared: List of state IDs that appeared during transition
        states_disappeared: List of state IDs that disappeared during transition

        action_type: Type of action that triggered transition ("click", "key_press", "drag", etc.)
        action_target: Optional StateImage ID that was clicked/targeted
        action_location: Optional (x, y) coordinates where action occurred
        action_data: Dictionary containing action-specific data (key pressed, drag path, etc.)

        recognition_images: List of StateImage IDs used to identify the target state
        recognition_confidence: Confidence score for target state recognition (0.0 to 1.0)

        timestamp: Unix timestamp when transition occurred
        frame_before: Frame number immediately before the transition
        frame_after: Frame number immediately after the transition
        duration_ms: Duration of the transition animation/change in milliseconds

        metadata: Additional arbitrary data for extensibility
    """

    id: str
    source_states: list[str]
    target_states: list[str]
    states_appeared: list[str]
    states_disappeared: list[str]

    # Outgoing (Action)
    action_type: str
    action_target: str | None = None
    action_location: tuple[int, int] | None = None
    action_data: dict[str, Any] = field(default_factory=dict)

    # Incoming (Recognition)
    recognition_images: list[str] = field(default_factory=list)
    recognition_confidence: float = 0.0

    # Timing
    timestamp: float = 0.0
    frame_before: int = 0
    frame_after: int = 0
    duration_ms: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_state_change(self) -> bool:
        """Check if this transition resulted in a state change.

        Returns:
            True if any states appeared or disappeared, False otherwise
        """
        return len(self.states_appeared) > 0 or len(self.states_disappeared) > 0

    @property
    def is_simple_transition(self) -> bool:
        """Check if this is a simple one-to-one state transition.

        Returns:
            True if exactly one state disappeared and one appeared
        """
        return len(self.states_appeared) == 1 and len(self.states_disappeared) == 1

    @property
    def frame_count(self) -> int:
        """Calculate the number of frames in this transition.

        Returns:
            Number of frames (after - before)
        """
        return self.frame_after - self.frame_before

    @property
    def has_action_target(self) -> bool:
        """Check if this transition has a specific action target.

        Returns:
            True if action_target or action_location is specified
        """
        return self.action_target is not None or self.action_location is not None

    def get_primary_source_state(self) -> str | None:
        """Get the primary source state ID.

        For simple transitions, returns the single source state.
        For multi-state, returns the first state that disappeared.

        Returns:
            State ID or None if no states disappeared
        """
        if self.states_disappeared:
            return self.states_disappeared[0]
        return None

    def get_primary_target_state(self) -> str | None:
        """Get the primary target state ID.

        For simple transitions, returns the single target state.
        For multi-state, returns the first state that appeared.

        Returns:
            State ID or None if no states appeared
        """
        if self.states_appeared:
            return self.states_appeared[0]
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary containing all serializable fields
        """
        return {
            "id": self.id,
            "source_states": self.source_states,
            "target_states": self.target_states,
            "states_appeared": self.states_appeared,
            "states_disappeared": self.states_disappeared,
            "action_type": self.action_type,
            "action_target": self.action_target,
            "action_location": (list(self.action_location) if self.action_location else None),
            "action_data": self.action_data,
            "recognition_images": self.recognition_images,
            "recognition_confidence": self.recognition_confidence,
            "timestamp": self.timestamp,
            "frame_before": self.frame_before,
            "frame_after": self.frame_after,
            "duration_ms": self.duration_ms,
            "is_state_change": self.is_state_change,
            "is_simple_transition": self.is_simple_transition,
            "frame_count": self.frame_count,
            "has_action_target": self.has_action_target,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transition":
        """Create Transition from dictionary.

        Args:
            data: Dictionary containing transition data

        Returns:
            New Transition instance
        """
        # Convert action_location list to tuple if present
        action_location = data.get("action_location")
        if action_location is not None and not isinstance(action_location, tuple):
            action_location = tuple(action_location)

        # Remove computed properties that shouldn't be in constructor
        data_copy = {
            k: v
            for k, v in data.items()
            if k
            not in [
                "is_state_change",
                "is_simple_transition",
                "frame_count",
                "has_action_target",
            ]
        }

        return cls(
            id=data_copy["id"],
            source_states=data_copy["source_states"],
            target_states=data_copy["target_states"],
            states_appeared=data_copy["states_appeared"],
            states_disappeared=data_copy["states_disappeared"],
            action_type=data_copy["action_type"],
            action_target=data_copy.get("action_target"),
            action_location=action_location,
            action_data=data_copy.get("action_data", {}),
            recognition_images=data_copy.get("recognition_images", []),
            recognition_confidence=data_copy.get("recognition_confidence", 0.0),
            timestamp=data_copy.get("timestamp", 0.0),
            frame_before=data_copy.get("frame_before", 0),
            frame_after=data_copy.get("frame_after", 0),
            duration_ms=data_copy.get("duration_ms", 0),
            metadata=data_copy.get("metadata", {}),
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"Transition(id={self.id}, "
            f"action={self.action_type}, "
            f"from={self.get_primary_source_state()}, "
            f"to={self.get_primary_target_state()}, "
            f"frames=[{self.frame_before}, {self.frame_after}])"
        )


@dataclass
class StateChangePoint:
    """Represents a point where the active states change.

    This is used during transition analysis to identify when the GUI moves
    from one state to another.

    Attributes:
        frame_number: Frame index where the state change occurs
        timestamp: Timestamp in seconds
        states_before: Set of state IDs active before the change
        states_after: Set of state IDs active after the change
        states_appeared: Set of state IDs that appeared (new states)
        states_disappeared: Set of state IDs that disappeared (old states)
    """

    frame_number: int
    timestamp: float
    states_before: set = field(default_factory=set)
    states_after: set = field(default_factory=set)
    states_appeared: set = field(default_factory=set)
    states_disappeared: set = field(default_factory=set)

    @property
    def is_state_change(self) -> bool:
        """Check if there was actually a state change."""
        return len(self.states_appeared) > 0 or len(self.states_disappeared) > 0

    @property
    def change_magnitude(self) -> int:
        """Get the magnitude of the state change (number of states changed)."""
        return len(self.states_appeared) + len(self.states_disappeared)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "states_before": sorted(self.states_before),
            "states_after": sorted(self.states_after),
            "states_appeared": sorted(self.states_appeared),
            "states_disappeared": sorted(self.states_disappeared),
            "is_state_change": self.is_state_change,
            "change_magnitude": self.change_magnitude,
        }
