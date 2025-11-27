"""StateRepository - Core state storage and lookup.

Responsible for storing states and providing efficient lookup by different identifiers.
"""

import logging
import threading

from .state import State
from .state_enum import StateEnum

logger = logging.getLogger(__name__)


class StateRepository:
    """Core repository for state storage and retrieval.

    Provides efficient lookup by name, ID, and enum with thread-safe operations.
    Single responsibility: Store and retrieve states.
    """

    def __init__(self) -> None:
        """Initialize the state repository."""
        self._states: dict[str, State] = {}
        self._states_by_enum: dict[StateEnum, State] = {}
        self._states_by_id: dict[int, State] = {}
        self._lock = threading.RLock()

    def add(self, state: State) -> bool:
        """Add a state to the repository.

        Args:
            state: State to add

        Returns:
            True if added successfully
        """
        with self._lock:
            try:
                name = state.name

                # Store state by name
                self._states[name] = state

                # Store by enum if available
                if state.state_enum:
                    self._states_by_enum[state.state_enum] = state

                # Store by ID if available
                if state.id is not None:
                    self._states_by_id[state.id] = state

                logger.debug(f"Added state '{name}' to repository")
                return True

            except Exception as e:
                logger.error(f"Failed to add state to repository: {e}")
                return False

    def remove(self, name: str) -> bool:
        """Remove a state from the repository.

        Args:
            name: Name of state to remove

        Returns:
            True if removed successfully
        """
        with self._lock:
            if name not in self._states:
                return False

            try:
                state = self._states.pop(name)

                # Remove from enum lookup
                if state.state_enum in self._states_by_enum:
                    del self._states_by_enum[state.state_enum]

                # Remove from ID lookup
                if state.id is not None and state.id in self._states_by_id:
                    del self._states_by_id[state.id]

                logger.debug(f"Removed state '{name}' from repository")
                return True

            except Exception as e:
                logger.error(f"Failed to remove state '{name}': {e}")
                return False

    def get(self, identifier: int | str | StateEnum) -> State | None:
        """Get a state by ID, name, or enum.

        Args:
            identifier: State ID (int), name (str), or enum (StateEnum)

        Returns:
            State or None if not found
        """
        with self._lock:
            if isinstance(identifier, int):
                return self._states_by_id.get(identifier)
            elif isinstance(identifier, str):
                return self._states.get(identifier)
            elif isinstance(identifier, StateEnum):
                return self._states_by_enum.get(identifier)
            return None

    def get_all(self) -> list[State]:
        """Get all states in the repository.

        Returns:
            List of all states
        """
        with self._lock:
            return list(self._states.values())

    def contains(self, name: str) -> bool:
        """Check if a state exists in the repository.

        Args:
            name: State name to check

        Returns:
            True if state exists
        """
        with self._lock:
            return name in self._states

    def get_names(self) -> list[str]:
        """Get all state names.

        Returns:
            List of state names
        """
        with self._lock:
            return list(self._states.keys())

    def count(self) -> int:
        """Get the number of states in the repository.

        Returns:
            Number of states
        """
        with self._lock:
            return len(self._states)

    def clear(self) -> None:
        """Clear all states from the repository."""
        with self._lock:
            self._states.clear()
            self._states_by_enum.clear()
            self._states_by_id.clear()
            logger.debug("Repository cleared")

    def __len__(self) -> int:
        """Get the number of states."""
        return self.count()

    def __contains__(self, name: str) -> bool:
        """Check if a state exists."""
        return self.contains(name)
