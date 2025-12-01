"""StateAutomator class - Framework-level state automation functionality.

This class provides high-level state management capabilities including:
- Automatic registration of @state decorated classes
- Collection and registration of @transition decorated methods
- Transition listing and execution
- State lifecycle management
"""

import logging
from typing import Any

from ..actions import Actions
from ..annotations.state import get_state_metadata, is_state
from ..annotations.transition_method import collect_transitions, create_code_transition
from ..model.state import State
from ..model.state.state_store import StateStore

logger = logging.getLogger(__name__)


class StateAutomator:
    """High-level state automation framework.

    Provides framework-level functionality for managing states and transitions,
    including automatic registration, transition execution, and state lifecycle.

    This class acts as a facade over StateStore, providing convenience methods
    for common automation patterns.

    Example:
        # Initialize automator
        automator = StateAutomator(actions)

        # Register state instances
        automator.register_state_instances([prompt_state, working_state])

        # Collect and register transitions
        automator.collect_and_register_transitions([prompt_state, working_state])

        # List available transitions
        automator.list_available_transitions()

        # Execute a transition
        automator.execute_transition("submit_prompt")
    """

    def __init__(
        self, actions: Actions | None = None, state_store: StateStore | None = None
    ) -> None:
        """Initialize StateAutomator.

        Args:
            actions: Actions instance for performing UI operations
            state_store: Optional StateStore instance (creates new one if not provided)
        """
        self.state_store = state_store or StateStore()
        self.actions = actions or Actions()

        # Track registered states and transitions
        self._registered_states: set[str] = set()
        self._registered_transitions: dict[str, list[dict[str, Any]]] = {}

        logger.info("StateAutomator initialized")

    def register_state_instances(self, state_instances: list[Any]) -> int:
        """Register multiple state instances with the StateStore.

        Automatically detects @state decorated classes and registers them.
        Sets the initial state if one is marked with initial=True.

        Args:
            state_instances: List of state class instances to register

        Returns:
            Number of states successfully registered
        """
        registered_count = 0
        initial_state_set = False

        for state_obj in state_instances:
            try:
                # Check if instance has a state attribute (the State object)
                if hasattr(state_obj, "state") and isinstance(state_obj.state, State):
                    # Register the state
                    if self.state_store.register(state_obj.state):
                        self._registered_states.add(state_obj.state.name)
                        registered_count += 1
                        logger.debug(f"Registered state: {state_obj.state.name}")

                    # Check if it's the initial state
                    if is_state(state_obj.__class__) and not initial_state_set:
                        metadata = get_state_metadata(state_obj.__class__)
                        if metadata and metadata.get("initial", False):
                            self.state_store.set_current_state(state_obj.state.name)
                            initial_state_set = True
                            logger.info(f"Set {state_obj.state.name} as initial state")
                else:
                    logger.warning(
                        f"State instance {state_obj} does not have a valid 'state' attribute"
                    )

            except Exception as e:
                logger.error(f"Failed to register state {state_obj}: {e}")

        logger.info(f"Registered {registered_count} states with StateStore")
        return registered_count

    def collect_and_register_transitions(
        self, objects_with_transitions: list[Any]
    ) -> int:
        """Collect @transition decorated methods and register them with StateStore.

        Scans provided objects for @transition decorated methods and automatically
        registers them as state transitions.

        Args:
            objects_with_transitions: List of objects that may have @transition methods

        Returns:
            Number of transitions successfully registered
        """
        all_transitions = []

        # Collect transitions from each object
        for obj in objects_with_transitions:
            try:
                transitions = collect_transitions(obj)
                for trans_meta in transitions:
                    # Store the transition metadata
                    from_state = trans_meta.get("from_state") or (
                        obj.name if hasattr(obj, "name") else None
                    )

                    if from_state:
                        trans_meta["from_state"] = from_state
                        all_transitions.append(trans_meta)

                        # Track transitions by source state
                        if from_state not in self._registered_transitions:
                            self._registered_transitions[from_state] = []
                        self._registered_transitions[from_state].append(trans_meta)

                        logger.debug(
                            f"Found transition: {trans_meta['name']} "
                            f"({from_state} -> {trans_meta['to_state']})"
                        )

            except Exception as e:
                logger.error(f"Failed to collect transitions from {obj}: {e}")

        # Register transitions with StateStore
        registered_count = 0
        for trans_meta in all_transitions:
            try:
                if trans_meta["from_state"] and trans_meta["to_state"]:
                    # Create CodeStateTransition from metadata
                    code_transition = create_code_transition(trans_meta)

                    # Register with StateStore
                    if self.state_store.add_transition(
                        trans_meta["from_state"],
                        trans_meta["to_state"],
                        code_transition,
                    ):
                        registered_count += 1

            except Exception as e:
                logger.error(f"Failed to register transition {trans_meta['name']}: {e}")

        logger.info(f"Registered {registered_count} transitions with StateStore")
        return registered_count

    def list_available_transitions(self) -> list[dict[str, Any]]:
        """List all available transitions from the current state.

        Returns:
            List of transition information dictionaries
        """
        current = self.state_store.get_current_state()
        if not current:
            logger.warning("No current state set")
            return []

        transitions = self.state_store.get_transitions(current.name)
        transition_info = []

        logger.info(f"Current state: {current.name}")
        logger.info(f"Available transitions ({len(transitions)}):")

        for trans in transitions:
            info = {
                "name": trans.name,
                "to_state": trans.to_state,
                "priority": trans.priority,
                "description": getattr(trans, "description", ""),
                "from_state": current.name,
            }
            transition_info.append(info)
            logger.info(
                f"  - {trans.name} -> {trans.to_state} (priority: {trans.priority})"
            )

        return transition_info

    def execute_transition(self, transition_name: str) -> bool:
        """Execute a specific transition by name.

        Executes the transition function and updates the state store accordingly,
        managing state activation/deactivation and current state changes.

        Args:
            transition_name: Name of the transition to execute

        Returns:
            True if transition succeeded, False otherwise
        """
        current = self.state_store.get_current_state()
        if not current:
            logger.error("No current state - cannot execute transition")
            return False

        transitions = self.state_store.get_transitions(current.name)

        for trans in transitions:
            if trans.name == transition_name:
                logger.info(
                    f"Executing transition: {transition_name} "
                    f"({current.name} -> {trans.to_state})"
                )

                try:
                    # Execute the transition function if it exists
                    if (
                        hasattr(trans, "transition_function")
                        and trans.transition_function
                    ):
                        success = trans.transition_function()
                    else:
                        # No function means automatic success
                        success = True

                    if success:
                        # Update state store based on transition metadata

                        # Handle exit states
                        if hasattr(trans, "exit_names"):
                            for state_name in trans.exit_names:
                                self.state_store.deactivate(state_name)
                                logger.debug(f"Deactivated state: {state_name}")

                        # Handle activate states
                        if hasattr(trans, "activate_names"):
                            for state_name in trans.activate_names:
                                self.state_store.activate(state_name)
                                logger.debug(f"Activated state: {state_name}")

                        # Update current state if specified
                        if trans.to_state:
                            self.state_store.set_current_state(trans.to_state)
                            logger.info(f"Transitioned to state: {trans.to_state}")

                        # Update statistics
                        self.state_store.record_transition(current.name)

                        logger.info(f"✓ Transition '{transition_name}' succeeded")
                        return True
                    else:
                        # Update failure statistics
                        self.state_store.record_failed_transition(
                            current.name, "Transition function returned False"
                        )
                        logger.error(f"✗ Transition '{transition_name}' failed")
                        return False

                except Exception as e:
                    self.state_store.record_failed_transition(current.name, str(e))
                    logger.error(
                        f"Exception during transition '{transition_name}': {e}"
                    )
                    return False

        logger.error(f"Transition not found: {transition_name}")
        return False

    def get_current_state_name(self) -> str | None:
        """Get the name of the current state.

        Returns:
            Current state name or None
        """
        current = self.state_store.get_current_state()
        return current.name if current else None

    def get_active_state_names(self) -> list[str]:
        """Get names of all active states.

        Returns:
            List of active state names
        """
        return [state.name for state in self.state_store.get_active_states()]

    def get_registered_states(self) -> list[str]:
        """Get names of all registered states.

        Returns:
            List of registered state names
        """
        return list(self._registered_states)

    def get_transition_history(
        self, state_name: str | None = None
    ) -> list[dict[str, Any]]:
        """Get transition history for a state or all states.

        Args:
            state_name: Optional state name to filter by

        Returns:
            List of transition metadata dictionaries
        """
        if state_name:
            return self._registered_transitions.get(state_name, [])

        # Return all transitions
        all_trans = []
        for transitions in self._registered_transitions.values():
            all_trans.extend(transitions)
        return all_trans

    def reset(self):
        """Reset the automator to initial state.

        Clears all registered states and transitions, resetting to a clean state.
        """
        self.state_store.clear()
        self._registered_states.clear()
        self._registered_transitions.clear()
        logger.info("StateAutomator reset")

    def get_statistics(self) -> dict[str, Any]:
        """Get automator statistics.

        Returns:
            Dictionary of statistics including state and transition counts
        """
        stats = self.state_store.get_statistics()
        stats.update(
            {
                "registered_states": len(self._registered_states),
                "total_transition_definitions": sum(
                    len(trans) for trans in self._registered_transitions.values()
                ),
            }
        )
        return stats

    def validate(self) -> list[str]:
        """Validate automator configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = self.state_store.validate()

        # Check that all registered transitions have valid states
        for from_state, transitions in self._registered_transitions.items():
            if from_state not in self._registered_states:
                errors.append(f"Transition source state '{from_state}' not registered")

            for trans in transitions:
                to_state = trans.get("to_state")
                if to_state and to_state not in self._registered_states:
                    errors.append(
                        f"Transition target state '{to_state}' not registered"
                    )

        return errors

    def __str__(self) -> str:
        """String representation."""
        return (
            f"StateAutomator({len(self._registered_states)} states, "
            f"{sum(len(t) for t in self._registered_transitions.values())} transitions)"
        )

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"StateAutomator(states={self._registered_states}, "
            f"current={self.get_current_state_name()})"
        )
