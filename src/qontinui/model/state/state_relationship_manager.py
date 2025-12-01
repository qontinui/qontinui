"""StateRelationshipManager - Manages state relationships and transitions.

Handles parent-child hierarchies and state transitions.
"""

import logging
import threading

from ..transition.state_transition import StateTransition

logger = logging.getLogger(__name__)


class StateRelationshipManager:
    """Manages relationships between states.

    Single responsibility: Track and manage parent-child relationships and transitions.
    """

    def __init__(self) -> None:
        """Initialize the relationship manager."""
        self._transitions: dict[str, list[StateTransition]] = {}
        self._parent_states: dict[str, str] = {}  # child -> parent mapping
        self._child_states: dict[str, set[str]] = {}  # parent -> children mapping
        self._lock = threading.RLock()

    def set_parent(self, child: str, parent: str) -> None:
        """Set parent-child relationship.

        Args:
            child: Child state name
            parent: Parent state name
        """
        with self._lock:
            # Remove old parent relationship
            if child in self._parent_states:
                old_parent = self._parent_states[child]
                if old_parent in self._child_states:
                    self._child_states[old_parent].discard(child)

            # Set new parent
            self._parent_states[child] = parent
            if parent not in self._child_states:
                self._child_states[parent] = set()
            self._child_states[parent].add(child)

            logger.debug(f"Set parent of '{child}' to '{parent}'")

    def get_parent(self, child: str) -> str | None:
        """Get parent of a state.

        Args:
            child: Child state name

        Returns:
            Parent state name or None
        """
        with self._lock:
            return self._parent_states.get(child)

    def get_children(self, parent: str) -> list[str]:
        """Get child states of a parent.

        Args:
            parent: Parent state name

        Returns:
            List of child state names
        """
        with self._lock:
            return list(self._child_states.get(parent, set()))

    def has_children(self, parent: str) -> bool:
        """Check if a state has children.

        Args:
            parent: Parent state name

        Returns:
            True if state has children
        """
        with self._lock:
            return parent in self._child_states and len(self._child_states[parent]) > 0

    def add_transition(self, from_state: str, to_state: str, transition: StateTransition) -> bool:
        """Add a transition between states.

        Args:
            from_state: Source state name
            to_state: Target state name
            transition: Transition object

        Returns:
            True if added successfully
        """
        with self._lock:
            if from_state not in self._transitions:
                self._transitions[from_state] = []

            self._transitions[from_state].append(transition)
            logger.debug(f"Added transition from '{from_state}' to '{to_state}'")
            return True

    def get_transitions(self, from_state: str) -> list[StateTransition]:
        """Get transitions from a state.

        Args:
            from_state: Source state name

        Returns:
            List of transitions from the state
        """
        with self._lock:
            return self._transitions.get(from_state, []).copy()

    def has_transition(self, from_state: str, to_state: str) -> bool:
        """Check if a transition exists between states.

        Args:
            from_state: Source state name
            to_state: Target state name

        Returns:
            True if transition exists
        """
        with self._lock:
            if from_state not in self._transitions:
                return False
            return any(t.to_state == to_state for t in self._transitions[from_state])

    def remove_state_relationships(self, name: str) -> None:
        """Remove all relationships for a state.

        Args:
            name: State name
        """
        with self._lock:
            # Remove as parent
            if name in self._child_states:
                for child in self._child_states[name]:
                    if child in self._parent_states:
                        del self._parent_states[child]
                del self._child_states[name]

            # Remove as child
            if name in self._parent_states:
                parent = self._parent_states[name]
                if parent in self._child_states:
                    self._child_states[parent].discard(name)
                del self._parent_states[name]

            # Remove transitions from this state
            if name in self._transitions:
                del self._transitions[name]

            # Remove transitions to this state
            for from_state in list(self._transitions.keys()):
                self._transitions[from_state] = [
                    t for t in self._transitions[from_state] if t.to_state != name
                ]
                if not self._transitions[from_state]:
                    del self._transitions[from_state]

            logger.debug(f"Removed all relationships for '{name}'")

    def get_transition_count(self) -> int:
        """Get the total number of transitions.

        Returns:
            Number of transitions
        """
        with self._lock:
            return sum(len(transitions) for transitions in self._transitions.values())

    def validate_relationships(self, valid_states: set[str]) -> list[str]:
        """Validate that all relationships reference valid states.

        Args:
            valid_states: Set of valid state names

        Returns:
            List of validation errors
        """
        errors = []
        with self._lock:
            # Check transitions reference valid states
            for from_state in self._transitions.keys():
                if from_state not in valid_states:
                    errors.append(f"Transition source '{from_state}' not in store")

            # Check parent-child consistency
            for child, parent in self._parent_states.items():
                if parent not in valid_states:
                    errors.append(f"Parent state '{parent}' not in store")
                if child not in self._child_states.get(parent, set()):
                    errors.append(f"Parent-child mismatch for '{child}'-'{parent}'")

        return errors

    def clear(self) -> None:
        """Clear all relationships."""
        with self._lock:
            self._transitions.clear()
            self._parent_states.clear()
            self._child_states.clear()
            logger.debug("Relationship manager cleared")
