"""StateService - manages all State objects in the automation framework.

This module provides a centralized service for managing State objects with
bidirectional ID mapping between string IDs (from JSON config) and integer IDs
(used internally by the library).

This is part of Phases 1 & 6 of the State/Transition Loading Implementation Plan.
"""

import logging
from typing import Any

from .state import State

logger = logging.getLogger(__name__)


class StateService:
    """Service for managing all State objects in the automation framework.

    This service provides centralized management of State objects with support for:
    - Storage and retrieval by integer ID
    - Storage and retrieval by state name
    - Bidirectional ID mapping between string IDs (from JSON config) and integer IDs
    - Auto-incrementing ID generation

    The StateService is the foundation of the state management system, enabling
    the rest of the framework to work with States through a consistent interface.

    Attributes:
        states_by_id: Map of integer IDs to State objects
        states_by_name: Map of state names to State objects
        string_id_to_int_id: Map of string IDs (from config) to integer IDs
        int_id_to_string_id: Map of integer IDs to string IDs (from config)
        next_id: Auto-incrementing counter for generating new integer IDs
    """

    def __init__(self) -> None:
        """Initialize the StateService with empty state storage."""
        self.states_by_id: dict[int, State] = {}
        self.states_by_name: dict[str, State] = {}
        self.string_id_to_int_id: dict[str, int] = {}
        self.int_id_to_string_id: dict[int, str] = {}
        self.next_id: int = 1

        logger.info("StateService initialized")

    def add_state(self, state: State) -> None:
        """Add a state to the service.

        The state must have either an ID already set, or the service will
        generate one automatically. The state is indexed by both ID and name.

        Args:
            state: State object to add

        Raises:
            ValueError: If a state with the same name already exists
        """
        # Check for duplicate names
        if state.name in self.states_by_name:
            logger.warning(f"State with name '{state.name}' already exists, replacing")

        # Assign ID if not set
        if state.id is None:
            state.id = self.next_id
            self.next_id += 1

        # Ensure next_id stays ahead
        if state.id >= self.next_id:
            self.next_id = state.id + 1

        # Store in both indexes
        self.states_by_id[state.id] = state
        self.states_by_name[state.name] = state

        logger.debug(f"Added state: id={state.id}, name='{state.name}'")

    def get_state(self, state_id: int) -> State | None:
        """Get a state by its integer ID.

        Args:
            state_id: Integer ID of the state

        Returns:
            State object if found, None otherwise
        """
        return self.states_by_id.get(state_id)

    def get_state_by_name(self, name: str) -> State | None:
        """Get a state by its name.

        Args:
            name: Name of the state

        Returns:
            State object if found, None otherwise
        """
        return self.states_by_name.get(name)

    def get_state_by_string_id(self, string_id: str) -> State | None:
        """Get a state by its string ID from configuration.

        This looks up the state using the string ID (e.g., "state-123")
        that was assigned in the JSON configuration.

        Args:
            string_id: String ID from configuration

        Returns:
            State object if found, None otherwise
        """
        int_id = self.string_id_to_int_id.get(string_id)
        if int_id is None:
            return None
        return self.states_by_id.get(int_id)

    def get_state_by_identifier(self, identifier: str) -> State | None:
        """Get a state by either its name or string ID.

        This method first tries to look up by name, then by string ID.
        This is useful when processing transitions that might reference
        states by either their human-readable name or their config ID.

        Args:
            identifier: Either state name or string ID from configuration

        Returns:
            State object if found, None otherwise
        """
        # First try by name
        state = self.states_by_name.get(identifier)
        if state is not None:
            return state

        # Then try by string ID
        return self.get_state_by_string_id(identifier)

    def get_all_states(self) -> list[State]:
        """Get all states managed by this service.

        Returns:
            List of all State objects, in no particular order
        """
        return list(self.states_by_id.values())

    def remove_state(self, state_id: int) -> None:
        """Remove a state from the service.

        Removes the state from all indexes and cleans up any ID mappings.

        Args:
            state_id: Integer ID of the state to remove
        """
        state = self.states_by_id.get(state_id)
        if state is None:
            logger.warning(f"Cannot remove state: ID {state_id} not found")
            return

        # Remove from all indexes
        del self.states_by_id[state_id]
        if state.name in self.states_by_name:
            del self.states_by_name[state.name]

        # Clean up ID mappings
        if state_id in self.int_id_to_string_id:
            string_id = self.int_id_to_string_id[state_id]
            del self.int_id_to_string_id[state_id]
            if string_id in self.string_id_to_int_id:
                del self.string_id_to_int_id[string_id]

        logger.debug(f"Removed state: id={state_id}, name='{state.name}'")

    def generate_id_for_string_id(self, string_id: str) -> int:
        """Generate or retrieve an integer ID for a string ID.

        This method maintains bidirectional mapping between string IDs (from JSON
        config) and integer IDs (used internally). If the string ID has already
        been mapped, the existing integer ID is returned. Otherwise, a new integer
        ID is generated and the mapping is stored.

        Args:
            string_id: String ID from configuration (e.g., "state-start")

        Returns:
            Integer ID corresponding to the string ID
        """
        # Return existing mapping if available
        if string_id in self.string_id_to_int_id:
            return self.string_id_to_int_id[string_id]

        # Generate new ID and create bidirectional mapping
        new_id = self.next_id
        self.next_id += 1

        self.string_id_to_int_id[string_id] = new_id
        self.int_id_to_string_id[new_id] = string_id

        logger.debug(f"Generated ID mapping: '{string_id}' -> {new_id}")

        return new_id

    def get_string_id(self, int_id: int) -> str | None:
        """Get the string ID corresponding to an integer ID.

        Args:
            int_id: Integer ID to look up

        Returns:
            String ID if mapping exists, None otherwise
        """
        return self.int_id_to_string_id.get(int_id)

    def get_int_id(self, string_id: str) -> int | None:
        """Get the integer ID corresponding to a string ID.

        This method only performs lookup - it does not generate new IDs.
        Use generate_id_for_string_id() to create new mappings.

        Args:
            string_id: String ID to look up

        Returns:
            Integer ID if mapping exists, None otherwise
        """
        return self.string_id_to_int_id.get(string_id)

    def clear(self) -> None:
        """Clear all states and ID mappings.

        Resets the service to its initial empty state. The ID counter
        is also reset to 1.
        """
        self.states_by_id.clear()
        self.states_by_name.clear()
        self.string_id_to_int_id.clear()
        self.int_id_to_string_id.clear()
        self.next_id = 1

        logger.info("StateService cleared")

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the managed states.

        Returns:
            Dictionary containing state count and ID mapping count
        """
        return {
            "total_states": len(self.states_by_id),
            "states_by_name": len(self.states_by_name),
            "string_to_int_mappings": len(self.string_id_to_int_id),
            "int_to_string_mappings": len(self.int_id_to_string_id),
            "next_id": self.next_id,
        }

    def __str__(self) -> str:
        """String representation of the StateService.

        Returns:
            Human-readable string describing the service state
        """
        return (
            f"StateService(states={len(self.states_by_id)}, "
            f"mappings={len(self.string_id_to_int_id)}, "
            f"next_id={self.next_id})"
        )

    def __repr__(self) -> str:
        """Detailed representation of the StateService.

        Returns:
            String representation suitable for debugging
        """
        return (
            f"StateService(states_by_id={len(self.states_by_id)}, "
            f"states_by_name={len(self.states_by_name)}, "
            f"string_id_to_int_id={len(self.string_id_to_int_id)}, "
            f"int_id_to_string_id={len(self.int_id_to_string_id)}, "
            f"next_id={self.next_id})"
        )
