"""StateStore class - ported from Qontinui framework.

Centralized storage and management of states.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

from ..transition.state_transition import StateTransition
from .state import State
from .state_enum import StateEnum

logger = logging.getLogger(__name__)


class StateStatus(Enum):
    """Status of a state in the store."""

    ACTIVE = auto()  # Currently active/loaded
    INACTIVE = auto()  # Not currently active
    TRANSITIONING = auto()  # In process of transition
    ERROR = auto()  # State has errors
    UNKNOWN = auto()  # Status unknown


@dataclass
class StateMetadata:
    """Metadata about a stored state."""

    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    transition_count: int = 0
    average_duration: float = 0.0
    last_error: str | None = None
    tags: set[str] = field(default_factory=set)
    custom_data: dict[str, Any] = field(default_factory=dict)


class StateStore:
    """Centralized storage and management of states.

    Port of StateStore from Qontinui framework class.

    StateStore provides a centralized repository for all application states,
    managing their lifecycle, relationships, and metadata. It ensures consistent
    state management across the application and provides utilities for state
    discovery, validation, and optimization.

    Key responsibilities:
    - Store and retrieve states by name or enum
    - Track state relationships and transitions
    - Manage state lifecycle (creation, activation, deletion)
    - Collect and provide state usage statistics
    - Ensure thread-safe state access
    - Cache frequently used states
    - Validate state consistency

    Thread safety:
    - All public methods are thread-safe
    - Uses ReentrantLock for complex operations
    - Supports concurrent reads, exclusive writes

    Example:
        # Initialize store
        store = StateStore()

        # Register states
        store.register(login_state)
        store.register(dashboard_state)

        # Add transition
        store.add_transition(login_state, dashboard_state, transition_func)

        # Get state
        state = store.get("LoginState")

        # Get active states
        active = store.get_active_states()
    """

    def __init__(self, max_cache_size: int = 100):
        """Initialize StateStore.

        Args:
            max_cache_size: Maximum number of states to cache
        """
        # State storage
        self._states: dict[str, State] = {}
        self._states_by_enum: dict[StateEnum, State] = {}

        # State metadata
        self._metadata: dict[str, StateMetadata] = {}

        # State relationships
        self._transitions: dict[str, list[StateTransition]] = {}
        self._parent_states: dict[str, str] = {}  # child -> parent mapping
        self._child_states: dict[str, set[str]] = {}  # parent -> children mapping

        # Active states tracking
        self._active_states: set[str] = set()
        self._current_state: str | None = None

        # State status
        self._status: dict[str, StateStatus] = {}

        # Cache configuration
        self._max_cache_size = max_cache_size
        self._cache_order: list[str] = []  # LRU cache order

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self._total_transitions = 0
        self._failed_transitions = 0

        logger.info(f"StateStore initialized with cache size {max_cache_size}")

    def register(self, state: State, parent: str | None = None) -> bool:
        """Register a state in the store.

        Args:
            state: State to register
            parent: Optional parent state name

        Returns:
            True if registered successfully
        """
        with self._lock:
            try:
                name = state.name

                # Check if already registered
                if name in self._states:
                    logger.warning(f"State '{name}' already registered, updating")

                # Store state
                self._states[name] = state
                if state.state_enum:
                    self._states_by_enum[state.state_enum] = state

                # Initialize metadata
                if name not in self._metadata:
                    self._metadata[name] = StateMetadata()

                # Set initial status
                self._status[name] = StateStatus.INACTIVE

                # Handle parent relationship
                if parent:
                    self._set_parent(name, parent)

                # Update cache
                self._update_cache(name)

                logger.debug(f"Registered state '{name}'")
                return True

            except Exception as e:
                logger.error(f"Failed to register state: {e}")
                return False

    def unregister(self, name: str) -> bool:
        """Unregister a state from the store.

        Args:
            name: Name of state to unregister

        Returns:
            True if unregistered successfully
        """
        with self._lock:
            if name not in self._states:
                return False

            try:
                # Remove from active states
                self._active_states.discard(name)
                if self._current_state == name:
                    self._current_state = None

                # Remove state
                state = self._states.pop(name)
                if state.state_enum in self._states_by_enum:
                    del self._states_by_enum[state.state_enum]

                # Remove metadata
                if name in self._metadata:
                    del self._metadata[name]

                # Remove status
                if name in self._status:
                    del self._status[name]

                # Remove from cache
                if name in self._cache_order:
                    self._cache_order.remove(name)

                # Remove relationships
                self._remove_relationships(name)

                logger.debug(f"Unregistered state '{name}'")
                return True

            except Exception as e:
                logger.error(f"Failed to unregister state '{name}': {e}")
                return False

    def get(self, identifier: str | StateEnum) -> State | None:
        """Get a state by name or enum.

        Args:
            identifier: State name or enum

        Returns:
            State or None if not found
        """
        with self._lock:
            if isinstance(identifier, str):
                state = self._states.get(identifier)
                if state:
                    self._access_state(identifier)
                return state
            elif isinstance(identifier, StateEnum):
                return self._states_by_enum.get(identifier)
            return None

    def get_all(self) -> list[State]:
        """Get all registered states.

        Returns:
            List of all states
        """
        with self._lock:
            return list(self._states.values())

    def get_active_states(self) -> list[State]:
        """Get all active states.

        Returns:
            List of active states
        """
        with self._lock:
            return [self._states[name] for name in self._active_states if name in self._states]

    def get_current_state(self) -> State | None:
        """Get the current state.

        Returns:
            Current state or None
        """
        with self._lock:
            if self._current_state:
                return self._states.get(self._current_state)
            return None

    def set_current_state(self, name: str) -> bool:
        """Set the current state.

        Args:
            name: Name of state to make current

        Returns:
            True if set successfully
        """
        with self._lock:
            if name not in self._states:
                logger.error(f"Cannot set current state: '{name}' not found")
                return False

            # Deactivate previous current state
            if self._current_state:
                self._status[self._current_state] = StateStatus.INACTIVE
                self._active_states.discard(self._current_state)

            # Activate new current state
            self._current_state = name
            self._active_states.add(name)
            self._status[name] = StateStatus.ACTIVE
            self._access_state(name)

            logger.info(f"Current state set to '{name}'")
            return True

    def activate(self, name: str) -> bool:
        """Activate a state.

        Args:
            name: Name of state to activate

        Returns:
            True if activated successfully
        """
        with self._lock:
            if name not in self._states:
                return False

            self._active_states.add(name)
            self._status[name] = StateStatus.ACTIVE
            self._access_state(name)

            logger.debug(f"Activated state '{name}'")
            return True

    def deactivate(self, name: str) -> bool:
        """Deactivate a state.

        Args:
            name: Name of state to deactivate

        Returns:
            True if deactivated successfully
        """
        with self._lock:
            if name not in self._states:
                return False

            self._active_states.discard(name)
            if self._current_state != name:
                self._status[name] = StateStatus.INACTIVE

            logger.debug(f"Deactivated state '{name}'")
            return True

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
            if from_state not in self._states or to_state not in self._states:
                logger.error("Cannot add transition: states not found")
                return False

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

    def get_metadata(self, name: str) -> StateMetadata | None:
        """Get metadata for a state.

        Args:
            name: State name

        Returns:
            StateMetadata or None
        """
        with self._lock:
            return self._metadata.get(name)

    def get_status(self, name: str) -> StateStatus:
        """Get status of a state.

        Args:
            name: State name

        Returns:
            State status
        """
        with self._lock:
            return self._status.get(name, StateStatus.UNKNOWN)

    def get_children(self, parent: str) -> list[str]:
        """Get child states of a parent.

        Args:
            parent: Parent state name

        Returns:
            List of child state names
        """
        with self._lock:
            return list(self._child_states.get(parent, set()))

    def get_parent(self, child: str) -> str | None:
        """Get parent of a state.

        Args:
            child: Child state name

        Returns:
            Parent state name or None
        """
        with self._lock:
            return self._parent_states.get(child)

    def find_by_tag(self, tag: str) -> list[State]:
        """Find states by tag.

        Args:
            tag: Tag to search for

        Returns:
            List of states with the tag
        """
        with self._lock:
            results = []
            for name, metadata in self._metadata.items():
                if tag in metadata.tags:
                    if name in self._states:
                        results.append(self._states[name])
            return results

    def add_tag(self, name: str, tag: str) -> bool:
        """Add a tag to a state.

        Args:
            name: State name
            tag: Tag to add

        Returns:
            True if added successfully
        """
        with self._lock:
            if name in self._metadata:
                self._metadata[name].tags.add(tag)
                return True
            return False

    def remove_tag(self, name: str, tag: str) -> bool:
        """Remove a tag from a state.

        Args:
            name: State name
            tag: Tag to remove

        Returns:
            True if removed successfully
        """
        with self._lock:
            if name in self._metadata:
                self._metadata[name].tags.discard(tag)
                return True
            return False

    def clear(self):
        """Clear all states from the store."""
        with self._lock:
            self._states.clear()
            self._states_by_enum.clear()
            self._metadata.clear()
            self._transitions.clear()
            self._parent_states.clear()
            self._child_states.clear()
            self._active_states.clear()
            self._current_state = None
            self._status.clear()
            self._cache_order.clear()
            self._total_transitions = 0
            self._failed_transitions = 0
            logger.info("StateStore cleared")

    def get_statistics(self) -> dict[str, Any]:
        """Get store statistics.

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            return {
                "total_states": len(self._states),
                "active_states": len(self._active_states),
                "cached_states": len(self._cache_order),
                "total_transitions": self._total_transitions,
                "failed_transitions": self._failed_transitions,
                "success_rate": (self._total_transitions - self._failed_transitions)
                / max(1, self._total_transitions)
                * 100,
            }

    def validate(self) -> list[str]:
        """Validate store consistency.

        Returns:
            List of validation errors
        """
        errors = []
        with self._lock:
            # Check current state exists
            if self._current_state and self._current_state not in self._states:
                errors.append(f"Current state '{self._current_state}' not in store")

            # Check active states exist
            for name in self._active_states:
                if name not in self._states:
                    errors.append(f"Active state '{name}' not in store")

            # Check transitions reference valid states
            for from_state, _transitions in self._transitions.items():
                if from_state not in self._states:
                    errors.append(f"Transition source '{from_state}' not in store")

            # Check parent-child consistency
            for child, parent in self._parent_states.items():
                if parent not in self._states:
                    errors.append(f"Parent state '{parent}' not in store")
                if child not in self._child_states.get(parent, set()):
                    errors.append(f"Parent-child mismatch for '{child}'-'{parent}'")

        return errors

    def _access_state(self, name: str):
        """Update access tracking for a state.

        Args:
            name: State name
        """
        if name in self._metadata:
            self._metadata[name].last_accessed = datetime.now()
            self._metadata[name].access_count += 1
            self._update_cache(name)

    def _update_cache(self, name: str):
        """Update LRU cache for a state.

        Args:
            name: State name
        """
        if name in self._cache_order:
            self._cache_order.remove(name)
        self._cache_order.append(name)

        # Trim cache if needed
        while len(self._cache_order) > self._max_cache_size:
            evicted = self._cache_order.pop(0)
            if evicted in self._active_states:
                # Don't evict active states
                self._cache_order.append(evicted)
                break

    def _set_parent(self, child: str, parent: str):
        """Set parent-child relationship.

        Args:
            child: Child state name
            parent: Parent state name
        """
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

    def _remove_relationships(self, name: str):
        """Remove all relationships for a state.

        Args:
            name: State name
        """
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

        # Remove transitions
        if name in self._transitions:
            del self._transitions[name]

        # Remove transitions to this state
        for from_state in list(self._transitions.keys()):
            self._transitions[from_state] = [
                t for t in self._transitions[from_state] if t.to_state != name
            ]
            if not self._transitions[from_state]:
                del self._transitions[from_state]

    def __str__(self) -> str:
        """String representation."""
        with self._lock:
            return (
                f"StateStore({len(self._states)} states, "
                f"{len(self._active_states)} active, "
                f"current='{self._current_state}')"
            )

    def __repr__(self) -> str:
        """Developer representation."""
        with self._lock:
            return (
                f"StateStore(states={len(self._states)}, "
                f"active={len(self._active_states)}, "
                f"transitions={len(self._transitions)}, "
                f"cache_size={self._max_cache_size})"
            )
