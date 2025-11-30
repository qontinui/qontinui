"""StateStore - Centralized storage and management of states.

Orchestrates state management using focused component classes.
"""

import logging
import threading
from typing import Any

from ..transition.state_transition import StateTransition
from .state import State
from .state_cache_manager import StateCacheManager
from .state_enum import StateEnum
from .state_lifecycle_manager import StateLifecycleManager, StateStatus
from .state_metadata_tracker import StateMetadata, StateMetadataTracker
from .state_relationship_manager import StateRelationshipManager
from .state_repository import StateRepository
from .state_validator import StateValidator

logger = logging.getLogger(__name__)


class StateStore:
    """Centralized storage and management of states.

    Orchestrates multiple focused components to provide comprehensive state management:
    - StateRepository: Core state storage and lookup
    - StateLifecycleManager: State activation and lifecycle
    - StateRelationshipManager: Parent-child and transition relationships
    - StateMetadataTracker: Access counts, tags, and metadata
    - StateCacheManager: LRU caching for performance
    - StateValidator: Consistency validation

    Thread safety:
    - All public methods are thread-safe
    - Uses ReentrantLock for complex operations

    Example:
        # Initialize store
        store = StateStore()

        # Register states
        store.register(login_state)
        store.register(dashboard_state)

        # Add transition
        store.add_transition(login_state.name, dashboard_state.name, transition_func)

        # Get state
        state = store.get("LoginState")

        # Get active states
        active = store.get_active_states()
    """

    def __init__(self, max_cache_size: int = 100) -> None:
        """Initialize StateStore with component managers.

        Args:
            max_cache_size: Maximum number of states to cache
        """
        # Core components
        self._repository = StateRepository()
        self._lifecycle = StateLifecycleManager()
        self._relationships = StateRelationshipManager()
        self._metadata = StateMetadataTracker()
        self._cache = StateCacheManager(max_cache_size)

        # Validator with callbacks to other components
        self._validator = StateValidator(
            repository_contains_func=self._repository.contains,
            lifecycle_current_state_func=self._lifecycle.get_current_state,
            lifecycle_active_states_func=self._lifecycle.get_active_states,
            relationship_validate_func=lambda states: self._relationships.validate_relationships(
                states
            ),
        )

        # Statistics
        self._total_transitions = 0
        self._failed_transitions = 0

        # Thread safety for statistics
        self._lock = threading.RLock()

        logger.info(f"StateStore initialized with cache size {max_cache_size}")

    def register(self, state: State, parent: str | None = None) -> bool:
        """Register a state in the store.

        Args:
            state: State to register
            parent: Optional parent state name

        Returns:
            True if registered successfully
        """
        try:
            name = state.name

            # Check if already registered
            if self._repository.contains(name):
                logger.warning(f"State '{name}' already registered, updating")

            # Register in all components
            if not self._repository.add(state):
                return False

            self._lifecycle.register_state(name)
            self._metadata.register_state(name)

            # Handle parent relationship
            if parent:
                if not self._repository.contains(parent):
                    logger.error(f"Parent state '{parent}' not found")
                    return False
                self._relationships.set_parent(name, parent)

            # Update cache
            self._cache.access(name, is_active=False)

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
        if not self._repository.contains(name):
            return False

        try:
            # Unregister from all components
            self._repository.remove(name)
            self._lifecycle.unregister_state(name)
            self._metadata.unregister_state(name)
            self._relationships.remove_state_relationships(name)
            self._cache.remove(name)

            logger.debug(f"Unregistered state '{name}'")
            return True

        except Exception as e:
            logger.error(f"Failed to unregister state '{name}': {e}")
            return False

    def get(self, identifier: int | str | StateEnum) -> State | None:
        """Get a state by ID, name, or enum.

        Args:
            identifier: State ID (int), name (str), or enum (StateEnum)

        Returns:
            State or None if not found
        """
        state = self._repository.get(identifier)
        if state:
            # Record access
            is_active = self._lifecycle.is_active(state.name)
            self._metadata.record_access(state.name)
            self._cache.access(state.name, is_active=is_active)
        return state

    def get_all(self) -> list[State]:
        """Get all registered states.

        Returns:
            List of all states
        """
        return self._repository.get_all()

    def get_active_states(self) -> list[State]:
        """Get all active states.

        Returns:
            List of active states
        """
        active_names = self._lifecycle.get_active_states()
        states = []
        for name in active_names:
            state = self._repository.get(name)
            if state:
                states.append(state)
        return states

    def get_current_state(self) -> State | None:
        """Get the current state.

        Returns:
            Current state or None
        """
        current_name = self._lifecycle.get_current_state()
        if current_name:
            return self._repository.get(current_name)
        return None

    def set_current_state(self, name: str) -> bool:
        """Set the current state.

        Args:
            name: Name of state to make current

        Returns:
            True if set successfully
        """
        if not self._repository.contains(name):
            logger.error(f"Cannot set current state: '{name}' not found")
            return False

        success = self._lifecycle.set_current_state(name)
        if success:
            self._metadata.record_access(name)
            self._cache.access(name, is_active=True)
        return success

    def activate(self, name: str) -> bool:
        """Activate a state.

        Args:
            name: Name of state to activate

        Returns:
            True if activated successfully
        """
        if not self._repository.contains(name):
            return False

        success = self._lifecycle.activate(name)
        if success:
            self._metadata.record_access(name)
            self._cache.access(name, is_active=True)
        return success

    def deactivate(self, name: str) -> bool:
        """Deactivate a state.

        Args:
            name: Name of state to deactivate

        Returns:
            True if deactivated successfully
        """
        if not self._repository.contains(name):
            return False

        return self._lifecycle.deactivate(name)

    def add_transition(self, from_state: str, to_state: str, transition: StateTransition) -> bool:
        """Add a transition between states.

        Args:
            from_state: Source state name
            to_state: Target state name
            transition: Transition object

        Returns:
            True if added successfully
        """
        if not self._repository.contains(from_state):
            logger.error(f"Cannot add transition: source state '{from_state}' not found")
            return False
        if not self._repository.contains(to_state):
            logger.error(f"Cannot add transition: target state '{to_state}' not found")
            return False

        return self._relationships.add_transition(from_state, to_state, transition)

    def get_transitions(self, from_state: str) -> list[StateTransition]:
        """Get transitions from a state.

        Args:
            from_state: Source state name

        Returns:
            List of transitions from the state
        """
        return self._relationships.get_transitions(from_state)

    def get_metadata(self, name: str) -> StateMetadata | None:
        """Get metadata for a state.

        Args:
            name: State name

        Returns:
            StateMetadata or None
        """
        return self._metadata.get_metadata(name)

    def get_status(self, name: str) -> StateStatus:
        """Get status of a state.

        Args:
            name: State name

        Returns:
            State status
        """
        return self._lifecycle.get_status(name)

    def get_children(self, parent: str) -> list[str]:
        """Get child states of a parent.

        Args:
            parent: Parent state name

        Returns:
            List of child state names
        """
        return self._relationships.get_children(parent)

    def get_parent(self, child: str) -> str | None:
        """Get parent of a state.

        Args:
            child: Child state name

        Returns:
            Parent state name or None
        """
        return self._relationships.get_parent(child)

    def find_by_tag(self, tag: str) -> list[State]:
        """Find states by tag.

        Args:
            tag: Tag to search for

        Returns:
            List of states with the tag
        """
        state_names = self._metadata.find_by_tag(tag)
        states = []
        for name in state_names:
            state = self._repository.get(name)
            if state:
                states.append(state)
        return states

    def add_tag(self, name: str, tag: str) -> bool:
        """Add a tag to a state.

        Args:
            name: State name
            tag: Tag to add

        Returns:
            True if added successfully
        """
        return self._metadata.add_tag(name, tag)

    def remove_tag(self, name: str, tag: str) -> bool:
        """Remove a tag from a state.

        Args:
            name: State name
            tag: Tag to remove

        Returns:
            True if removed successfully
        """
        return self._metadata.remove_tag(name, tag)

    def record_transition(self, name: str, duration: float | None = None) -> None:
        """Record a transition for statistics.

        Args:
            name: State name
            duration: Optional transition duration in seconds
        """
        with self._lock:
            self._total_transitions += 1
        self._metadata.record_transition(name, duration)

    def record_failed_transition(self, name: str, error: str) -> None:
        """Record a failed transition.

        Args:
            name: State name
            error: Error message
        """
        with self._lock:
            self._failed_transitions += 1
        self._metadata.record_error(name, error)

    def clear(self) -> None:
        """Clear all states from the store."""
        self._repository.clear()
        self._lifecycle.clear()
        self._relationships.clear()
        self._metadata.clear()
        self._cache.clear()

        with self._lock:
            self._total_transitions = 0
            self._failed_transitions = 0

        logger.info("StateStore cleared")

    def get_statistics(self) -> dict[str, Any]:
        """Get store statistics.

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            stats = {
                "total_states": self._repository.count(),
                "active_states": self._lifecycle.active_count(),
                "cached_states": self._cache.cache_size(),
                "total_transitions": self._total_transitions,
                "failed_transitions": self._failed_transitions,
                "success_rate": (
                    (self._total_transitions - self._failed_transitions)
                    / max(1, self._total_transitions)
                    * 100
                ),
            }

        # Add cache statistics
        stats.update(self._cache.get_statistics())

        return stats

    def validate(self) -> list[str]:
        """Validate store consistency.

        Returns:
            List of validation errors
        """
        # Get valid states for relationship validation
        valid_states = set(self._repository.get_names())

        # Update validator with current valid states
        errors = []

        # Check current state exists
        current_state = self._lifecycle.get_current_state()
        if current_state and not self._repository.contains(current_state):
            errors.append(f"Current state '{current_state}' not in repository")

        # Check active states exist
        active_states = self._lifecycle.get_active_states()
        for name in active_states:
            if not self._repository.contains(name):
                errors.append(f"Active state '{name}' not in repository")

        # Validate relationships
        relationship_errors = self._relationships.validate_relationships(valid_states)
        errors.extend(relationship_errors)

        return errors

    def __str__(self) -> str:
        """String representation."""
        current = self._lifecycle.get_current_state()
        return (
            f"StateStore({self._repository.count()} states, "
            f"{self._lifecycle.active_count()} active, "
            f"current='{current}')"
        )

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"StateStore(states={self._repository.count()}, "
            f"active={self._lifecycle.active_count()}, "
            f"transitions={self._relationships.get_transition_count()}, "
            f"cache_size={self._cache._max_cache_size})"
        )
