"""State Operations Facade - Delegates state operations to StateExecutionAPI.

This module provides the StateOperationsFacade class that handles:
- Delegating transition execution to StateExecutionAPI
- Delegating state navigation to StateExecutionAPI
- Delegating state queries to StateExecutionAPI
- Converting API results to execution manager format

ExecutionManager NEVER touches state directly - all state operations
are delegated to StateExecutionAPI through this facade.
"""

import logging
from typing import Any

from ..state_execution_api import StateExecutionAPI

logger = logging.getLogger(__name__)


class StateOperationsFacade:
    """Facade for state operations via StateExecutionAPI.

    The StateOperationsFacade provides:
    - Clean delegation to StateExecutionAPI
    - Result format conversion for ExecutionManager
    - Consistent error handling for state operations

    ExecutionManager NEVER touches state directly - all state operations
    are delegated through this facade to StateExecutionAPI.
    """

    def __init__(self, state_apis: dict[str, StateExecutionAPI]) -> None:
        """Initialize state operations facade.

        Args:
            state_apis: Dictionary mapping execution IDs to StateExecutionAPI instances
        """
        self.state_apis = state_apis

        logger.info("StateOperationsFacade initialized")

    def execute_transition(self, execution_id: str, transition_id: str) -> dict[str, Any]:
        """Execute a transition via StateExecutionAPI.

        Args:
            execution_id: Execution identifier
            transition_id: Transition identifier to execute

        Returns:
            Dictionary with transition execution result:
            - success: Whether transition succeeded
            - transition_id: Transition that was executed
            - activated_states: List of states activated
            - deactivated_states: List of states deactivated
            - error: Error message if failed

        Raises:
            ValueError: If execution not found
        """
        state_api = self._get_state_api(execution_id)

        # Delegate to StateExecutionAPI (library handles ALL state management)
        result = state_api.execute_transition(transition_id)

        return {
            "success": result.success,
            "transition_id": result.transition_id,
            "activated_states": result.activated_states,
            "deactivated_states": result.deactivated_states,
            "error": result.error,  # type: ignore[attr-defined]
        }

    def navigate_to_states(self, execution_id: str, target_state_ids: list[str]) -> dict[str, Any]:
        """Navigate to target states via StateExecutionAPI.

        Args:
            execution_id: Execution identifier
            target_state_ids: List of target state IDs

        Returns:
            Dictionary with navigation result:
            - success: Whether navigation succeeded
            - path: List of transition IDs executed
            - active_states: Currently active states
            - error: Error message if failed

        Raises:
            ValueError: If execution not found
        """
        state_api = self._get_state_api(execution_id)

        # Delegate to StateExecutionAPI (library handles ALL state management)
        result = state_api.navigate_to_states(target_state_ids)  # type: ignore[arg-type]

        return {
            "success": result.success,
            "path": result.path,
            "active_states": result.active_states,  # type: ignore[attr-defined]
            "error": result.error,  # type: ignore[attr-defined]
        }

    def get_active_states(self, execution_id: str) -> list[str]:
        """Get active states via StateExecutionAPI.

        Args:
            execution_id: Execution identifier

        Returns:
            List of currently active state IDs

        Raises:
            ValueError: If execution not found
        """
        state_api = self._get_state_api(execution_id)

        # Delegate to StateExecutionAPI (library handles ALL state management)
        return state_api.get_active_states()  # type: ignore[return-value]

    def get_available_transitions(self, execution_id: str) -> list[dict[str, Any]]:
        """Get available transitions via StateExecutionAPI.

        Args:
            execution_id: Execution identifier

        Returns:
            List of available transition information dictionaries

        Raises:
            ValueError: If execution not found
        """
        state_api = self._get_state_api(execution_id)

        # Delegate to StateExecutionAPI (library handles ALL state management)
        return state_api.get_available_transitions()

    def _get_state_api(self, execution_id: str) -> StateExecutionAPI:
        """Get StateExecutionAPI for an execution.

        Args:
            execution_id: Execution identifier

        Returns:
            StateExecutionAPI for the execution

        Raises:
            ValueError: If execution not found
        """
        state_api = self.state_apis.get(execution_id)
        if not state_api:
            raise ValueError(f"Execution {execution_id} not found")
        return state_api
