"""State query operations.

This module handles state query operations such as getting active states,
available transitions, and statistics.

Architecture:
    - StateQuery: Queries state information
    - Delegates to StateMemory and StateValidator
    - Provides statistics aggregation

Key Features:
    1. Get active states from StateMemory
    2. Get available transitions via StateValidator
    3. Aggregate statistics from components
    4. Clean query interface

Example:
    >>> query = StateQuery(state_memory, validator, config, transition_executor, navigator)
    >>> active = query.get_active_states()
    >>> transitions = query.get_available_transitions()
    >>> stats = query.get_statistics()
"""

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)


class StateQuery:
    """Handles state query operations.

    Provides query interface for active states, available transitions,
    and execution statistics.

    Example:
        >>> query = StateQuery(state_memory, validator, config, transition_executor, navigator)
        >>> active = query.get_active_states()
        >>> print(f"Active states: {active}")
        >>> transitions = query.get_available_transitions()
        >>> print(f"Available: {[t.name for t in transitions]}")
    """

    def __init__(
        self,
        state_memory: Any,
        validator: Any,
        config: Any,
        transition_executor: Any,
        navigator: Any,
    ) -> None:
        """Initialize StateQuery.

        Args:
            state_memory: EnhancedStateMemory for state tracking
            validator: StateValidator for transition lookup
            config: QontinuiConfig with states and transitions
            transition_executor: EnhancedTransitionExecutor for statistics
            navigator: PathfindingNavigator for statistics
        """
        self.state_memory = state_memory
        self.validator = validator
        self.config = config
        self.transition_executor = transition_executor
        self.navigator = navigator

    def get_active_states(self) -> set[int]:
        """Get currently active state IDs.

        Returns:
            Set of active state IDs

        Example:
            >>> active = query.get_active_states()
            >>> print(f"Active states: {active}")
        """
        return set(self.state_memory.active_states)

    def get_available_transitions(self) -> list[Any]:
        """Get transitions available from current states.

        Returns transitions that can be executed from the current active states.

        Returns:
            List of available transition objects

        Example:
            >>> transitions = query.get_available_transitions()
            >>> for t in transitions:
            ...     print(f"Available: {t.name}")
        """
        active_states = self.state_memory.active_states
        return cast(list[Any], self.validator.get_available_transitions(active_states))

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about state execution.

        Returns:
            Dictionary with execution statistics

        Example:
            >>> stats = query.get_statistics()
            >>> print(f"Total transitions: {stats['total_transitions']}")
            >>> print(f"Active states: {stats['active_states']}")
        """
        stats = {
            "active_states": len(self.state_memory.active_states),
            "total_states": len(self.config.states),
            "total_transitions": len(self.config.transitions),
        }

        # Add memory statistics
        if hasattr(self.state_memory, "get_statistics"):
            memory_stats = self.state_memory.get_statistics()
            stats.update({"memory": memory_stats})

        # Add executor statistics
        if hasattr(self.transition_executor, "get_execution_statistics"):
            executor_stats = self.transition_executor.get_execution_statistics()
            stats.update({"transitions": executor_stats})

        # Add navigator statistics
        if hasattr(self.navigator, "get_navigation_statistics"):
            navigator_stats = self.navigator.get_navigation_statistics()
            stats.update({"navigation": navigator_stats})

        return stats
