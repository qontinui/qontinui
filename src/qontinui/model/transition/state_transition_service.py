"""State transition service module - manages state transitions in the automation framework.

This module is part of the Brobot-to-Qontinui migration and provides
services for managing state transitions within the state machine.
"""

import logging
from dataclasses import dataclass

from .state_transition import StateTransition
from .state_transitions import StateTransitions

logger = logging.getLogger(__name__)


@dataclass
class TransitionRecord:
    """Record of a state transition execution."""

    from_state: str
    to_state: str
    success: bool
    timestamp: float
    duration: float = 0.0
    error: str | None = None


class StateTransitionService:
    """Service for managing state transitions in the automation framework.

    This service provides:
    - Registration and management of state transitions
    - Validation of transition paths
    - Execution history tracking
    - Transition availability checking

    Based on Brobot's StateTransitionService pattern.
    """

    def __init__(self):
        """Initialize the state transition service."""
        # Map of state name to available transitions
        self._transitions: dict[str, list[StateTransition]] = {}

        # Transition execution history
        self._history: list[TransitionRecord] = []

        # Currently registered states
        self._registered_states: set[str] = set()

        # Transition validation rules
        self._validation_rules: list[callable] = []

        logger.info("StateTransitionService initialized")

    def register_transition(self, transition: StateTransition) -> None:
        """Register a state transition.

        Args:
            transition: The state transition to register
        """
        # TODO: StateTransition doesn't have from_state/to_state attributes
        # Need to determine how to properly register transitions
        # For now, just add to the general list
        logger.debug(
            "Transition registration needs to be updated for new StateTransition structure"
        )

    def register_transitions(self, transitions: StateTransitions) -> None:
        """Register multiple state transitions.

        Args:
            transitions: Container of state transitions to register
        """
        state_name = transitions.get_state_name()

        if state_name not in self._transitions:
            self._transitions[state_name] = []

        # Add all transitions from the container
        for transition in transitions.get_transitions():
            self._transitions[state_name].append(transition)
            self._registered_states.add(state_name)

            # Track destination states if specified
            if hasattr(transition, "activate_names"):
                for dest_state in transition.activate_names:
                    self._registered_states.add(dest_state)

        logger.debug(
            f"Registered {len(transitions.get_transitions())} transitions for state: {state_name}"
        )

    def get_transitions_from(self, state_name: str) -> list[StateTransition]:
        """Get all transitions available from a given state.

        Args:
            state_name: Name of the source state

        Returns:
            List of available transitions from the state
        """
        return self._transitions.get(state_name, [])

    def get_transition(self, from_state: str, to_state: str) -> StateTransition | None:
        """Get a specific transition between two states.

        Args:
            from_state: Source state name
            to_state: Target state name

        Returns:
            The transition if found, None otherwise
        """
        transitions = self._transitions.get(from_state, [])

        for transition in transitions:
            # Check if this transition leads to the target state
            if hasattr(transition, "to_state") and transition.to_state == to_state:
                return transition
            elif hasattr(transition, "activate_names") and to_state in transition.activate_names:
                return transition

        return None

    def can_transition(self, from_state: str, to_state: str) -> bool:
        """Check if a transition is possible between two states.

        Args:
            from_state: Source state name
            to_state: Target state name

        Returns:
            True if transition is possible, False otherwise
        """
        transition = self.get_transition(from_state, to_state)

        if transition is None:
            return False

        # Apply validation rules
        for rule in self._validation_rules:
            if not rule(from_state, to_state, transition):
                return False

        return True

    def execute_transition(self, from_state: str, to_state: str) -> bool:
        """Execute a transition between states.

        This method finds and executes the appropriate transition,
        recording the result in the history.

        Args:
            from_state: Source state name
            to_state: Target state name

        Returns:
            True if transition was successful, False otherwise
        """
        import time

        start_time = time.time()

        transition = self.get_transition(from_state, to_state)

        if transition is None:
            logger.warning(f"No transition found from {from_state} to {to_state}")
            self._record_transition(
                from_state, to_state, False, start_time, error="No transition found"
            )
            return False

        try:
            # Execute the transition
            logger.info(f"Executing transition: {from_state} -> {to_state}")

            # For CodeStateTransition objects
            if hasattr(transition, "transition_function"):
                success = transition.transition_function()
            # For other transition types
            elif hasattr(transition, "execute"):
                success = transition.execute()
            else:
                logger.error(f"Transition has no executable method: {transition}")
                success = False

            self._record_transition(from_state, to_state, success, start_time)

            if success:
                logger.info(f"Transition successful: {from_state} -> {to_state}")
            else:
                logger.warning(f"Transition failed: {from_state} -> {to_state}")

            return success

        except Exception as e:
            logger.error(f"Error executing transition from {from_state} to {to_state}", exc_info=e)
            self._record_transition(from_state, to_state, False, start_time, error=str(e))
            return False

    def _record_transition(
        self,
        from_state: str,
        to_state: str,
        success: bool,
        start_time: float,
        error: str | None = None,
    ) -> None:
        """Record a transition execution in the history.

        Args:
            from_state: Source state
            to_state: Target state
            success: Whether the transition was successful
            start_time: When the transition started
            error: Error message if applicable
        """
        import time

        record = TransitionRecord(
            from_state=from_state,
            to_state=to_state,
            success=success,
            timestamp=start_time,
            duration=time.time() - start_time,
            error=error,
        )
        self._history.append(record)

    def get_history(self) -> list[TransitionRecord]:
        """Get the transition execution history.

        Returns:
            List of transition records
        """
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear the transition execution history."""
        self._history.clear()
        logger.debug("Transition history cleared")

    def add_validation_rule(self, rule: callable) -> None:
        """Add a validation rule for transitions.

        The rule should be a callable that takes (from_state, to_state, transition)
        and returns True if the transition is valid.

        Args:
            rule: Validation function
        """
        self._validation_rules.append(rule)
        logger.debug(f"Added transition validation rule: {rule.__name__}")

    def get_registered_states(self) -> set[str]:
        """Get all registered state names.

        Returns:
            Set of state names
        """
        return self._registered_states.copy()

    def get_transition_graph(self) -> dict[str, list[str]]:
        """Get a graph representation of all transitions.

        Returns:
            Dictionary mapping source states to list of reachable states
        """
        graph = {}

        for from_state, transitions in self._transitions.items():
            destinations = set()

            for transition in transitions:
                if hasattr(transition, "to_state") and transition.to_state:
                    destinations.add(transition.to_state)
                elif hasattr(transition, "activate_names"):
                    destinations.update(transition.activate_names)

            graph[from_state] = list(destinations)

        return graph

    def find_path(self, from_state: str, to_state: str) -> list[str] | None:
        """Find a path between two states if one exists.

        Uses breadth-first search to find the shortest path.

        Args:
            from_state: Starting state
            to_state: Target state

        Returns:
            List of states forming the path, or None if no path exists
        """
        if from_state == to_state:
            return [from_state]

        if from_state not in self._registered_states:
            logger.warning(f"Source state not registered: {from_state}")
            return None

        if to_state not in self._registered_states:
            logger.warning(f"Target state not registered: {to_state}")
            return None

        # BFS to find shortest path
        from collections import deque

        queue = deque([(from_state, [from_state])])
        visited = {from_state}

        while queue:
            current_state, path = queue.popleft()

            # Get all reachable states from current
            transitions = self._transitions.get(current_state, [])

            for transition in transitions:
                next_states = set()

                if hasattr(transition, "to_state") and transition.to_state:
                    next_states.add(transition.to_state)
                elif hasattr(transition, "activate_names"):
                    next_states.update(transition.activate_names)

                for next_state in next_states:
                    if next_state == to_state:
                        return path + [next_state]

                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append((next_state, path + [next_state]))

        return None

    def reset(self) -> None:
        """Reset the service to initial state."""
        self._transitions.clear()
        self._history.clear()
        self._registered_states.clear()
        self._validation_rules.clear()
        logger.info("StateTransitionService reset")
