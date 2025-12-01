"""State transition manager - manages transitions between states.

Handles outgoing and incoming transitions, tracking navigation paths
and state relationships.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...transition.state_transition import StateTransition


@dataclass
class StateTransitionManager:
    """Manages transitions from and to a state.

    Responsible for storing and querying outgoing and incoming transitions,
    enabling state navigation and path finding.
    """

    transitions: list[StateTransition] = field(default_factory=list)
    """List of outgoing transitions from this state."""

    incoming_transitions: list[StateTransition] = field(default_factory=list)
    """List of incoming transitions to this state (verification workflows executed when entering)."""

    def add(self, transition: StateTransition) -> None:
        """Add an outgoing transition from this state.

        Args:
            transition: StateTransition to add
        """
        self.transitions.append(transition)

    def add_incoming(self, transition: StateTransition) -> None:
        """Add an incoming transition to this state.

        Args:
            transition: StateTransition to add
        """
        self.incoming_transitions.append(transition)

    def get_transitions_to(self, target_state: str) -> list[StateTransition]:
        """Get transitions to a specific target state.

        Args:
            target_state: Target state name

        Returns:
            List of transitions to the target state
        """
        return [t for t in self.transitions if t.to_state == target_state]

    def get_possible_next_states(self) -> list[str]:
        """Get list of possible next states from this state.

        Returns:
            List of state names that can be reached from this state
        """
        next_states = []
        for transition in self.transitions:
            if transition.to_state:
                next_states.append(transition.to_state)
        return list(set(next_states))  # Remove duplicates

    def get_outgoing_transitions(self) -> list[StateTransition]:
        """Get all outgoing transitions.

        Returns:
            List of outgoing transitions
        """
        return self.transitions

    def get_incoming_transitions(self) -> list[StateTransition]:
        """Get all incoming transitions.

        Returns:
            List of incoming transitions
        """
        return self.incoming_transitions

    def has_transition_to(self, target_state: str) -> bool:
        """Check if there is a transition to the target state.

        Args:
            target_state: Target state name

        Returns:
            True if transition exists, False otherwise
        """
        return any(t.to_state == target_state for t in self.transitions)

    def remove_transition(self, transition: StateTransition) -> bool:
        """Remove an outgoing transition.

        Args:
            transition: Transition to remove

        Returns:
            True if removed, False if not found
        """
        try:
            self.transitions.remove(transition)
            return True
        except ValueError:
            return False

    def clear_transitions(self) -> None:
        """Clear all outgoing transitions."""
        self.transitions.clear()

    def clear_incoming_transitions(self) -> None:
        """Clear all incoming transitions."""
        self.incoming_transitions.clear()

    def __str__(self) -> str:
        """String representation."""
        parts = []
        parts.append(f"Outgoing transitions: {len(self.transitions)}")
        for t in self.transitions:
            parts.append(f"  -> {t.to_state}")
        parts.append(f"Incoming transitions: {len(self.incoming_transitions)}")
        for t in self.incoming_transitions:
            parts.append(
                f"  <- {t.from_state if hasattr(t, 'from_state') else 'unknown'}"
            )
        return "\n".join(parts)
