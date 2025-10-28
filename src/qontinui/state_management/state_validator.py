"""State and transition validation.

This module handles validation of states and transitions, ensuring they exist
in the configuration before execution.

Architecture:
    - StateValidator: Validates states and transitions
    - Supports lookup by ID or name
    - Provides clear error messages for missing items

Key Features:
    1. Transition validation and lookup
    2. State validation and lookup
    3. Available transitions query
    4. Configuration-based validation

Example:
    >>> validator = StateValidator(config)
    >>> transition = validator.validate_transition("login")
    >>> if transition:
    ...     print(f"Found: {transition.name}")
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class StateValidator:
    """Validates states and transitions against configuration.

    Provides validation and lookup operations for states and transitions,
    ensuring they exist in the configuration before execution.

    Example:
        >>> validator = StateValidator(config)
        >>> transition = validator.find_transition("login")
        >>> if transition:
        ...     print(f"Valid transition: {transition.name}")
    """

    def __init__(self, config: Any) -> None:
        """Initialize StateValidator.

        Args:
            config: QontinuiConfig with states and transitions
        """
        self.config = config

    def validate_transition(self, transition_id: str) -> Optional[Any]:
        """Validate that transition exists in configuration.

        Args:
            transition_id: ID of transition to validate

        Returns:
            Transition object if found, None otherwise
        """
        transition = self.find_transition(transition_id)
        if not transition:
            logger.error(f"Transition '{transition_id}' not found in configuration")
        return transition

    def find_transition(self, transition_id: str) -> Optional[Any]:
        """Find transition by ID in config.

        Args:
            transition_id: Transition ID to find

        Returns:
            Transition object or None if not found
        """
        for transition in self.config.transitions:
            if getattr(transition, "id", None) == transition_id:
                return transition
        return None

    def get_available_transitions(self, active_states: set[int]) -> list[Any]:
        """Get transitions available from current active states.

        Returns transitions that can be executed from the given active states.

        Args:
            active_states: Set of currently active state IDs

        Returns:
            List of available transition objects

        Example:
            >>> validator = StateValidator(config)
            >>> active = {1, 2, 3}
            >>> transitions = validator.get_available_transitions(active)
            >>> for t in transitions:
            ...     print(f"Available: {t.name}")
        """
        available = []
        for transition in self.config.transitions:
            if hasattr(transition, "from_state") and transition.from_state in active_states:
                available.append(transition)
            elif hasattr(transition, "from_states") and any(
                s in active_states for s in transition.from_states
            ):
                available.append(transition)

        logger.debug(
            f"Found {len(available)} available transitions from {len(active_states)} active states"
        )

        return available

    def validate_state(self, state_id: int) -> Optional[Any]:
        """Validate that state exists in configuration.

        Args:
            state_id: ID of state to validate

        Returns:
            State object if found, None otherwise
        """
        state = self.find_state(state_id)
        if not state:
            logger.error(f"State '{state_id}' not found in configuration")
        return state

    def find_state(self, state_id: int) -> Optional[Any]:
        """Find state by ID in config.

        Args:
            state_id: State ID to find

        Returns:
            State object or None if not found
        """
        for state in self.config.states:
            if getattr(state, "id", None) == state_id:
                return state
        return None
