"""State management endpoints.

This router provides statechart management endpoints:
- Execute transition
- Navigate to states
- Get active states
- Get available transitions

All state management is delegated to ExecutionManager, which delegates to StateExecutionAPI.
REST layer never touches state directly.

NO backward compatibility - clean FastAPI code.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..execution_manager import ExecutionManager
from ..models import (
    ActiveStatesResponse,
    AvailableTransitionsResponse,
    StateNavigationRequest,
    StateNavigationResponse,
    TransitionExecutionRequest,
    TransitionExecutionResponse,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Dependency Injection
# ============================================================================


def get_manager() -> ExecutionManager:
    """Get execution manager from app state.

    This dependency is injected by FastAPI and retrieves the manager
    from the application state.

    Yields:
        ExecutionManager instance
    """
    # This will be overridden by app.dependency_overrides in create_app
    raise NotImplementedError("Manager dependency not configured")


# ============================================================================
# Router
# ============================================================================

router = APIRouter(prefix="/api", tags=["state"])


@router.post("/transitions/execute", response_model=TransitionExecutionResponse)
async def execute_transition(
    request: TransitionExecutionRequest,
    manager: ExecutionManager = Depends(get_manager),
):
    """Execute a transition.

    This endpoint delegates all state management to ExecutionManager,
    which delegates to StateExecutionAPI. REST layer never touches state directly.

    Args:
        request: Transition execution request with execution_id and transition_id
        manager: Execution manager instance

    Returns:
        TransitionExecutionResponse with success status and state changes

    Raises:
        HTTPException: If execution not found or transition execution fails
    """
    try:
        # Delegate to ExecutionManager (which delegates to StateExecutionAPI)
        result = manager.execute_transition(
            execution_id=request.execution_id, transition_id=request.transition_id
        )

        return TransitionExecutionResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to execute transition: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/states/navigate", response_model=StateNavigationResponse)
async def navigate_to_states(
    request: StateNavigationRequest, manager: ExecutionManager = Depends(get_manager)
):
    """Navigate to target states using pathfinding.

    This endpoint delegates all state management to ExecutionManager,
    which delegates to StateExecutionAPI. REST layer never touches state directly.

    Args:
        request: Navigation request with execution_id and target_state_ids
        manager: Execution manager instance

    Returns:
        StateNavigationResponse with path and active states

    Raises:
        HTTPException: If execution not found or navigation fails
    """
    try:
        # Delegate to ExecutionManager (which delegates to StateExecutionAPI)
        result = manager.navigate_to_states(
            execution_id=request.execution_id, target_state_ids=request.target_state_ids
        )

        return StateNavigationResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to navigate to states: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/states/active", response_model=ActiveStatesResponse)
async def get_active_states(
    execution_id: str = "default", manager: ExecutionManager = Depends(get_manager)
):
    """Get currently active states.

    This endpoint delegates all state management to ExecutionManager,
    which delegates to StateExecutionAPI. REST layer never touches state directly.

    Args:
        execution_id: Execution identifier (default: "default")
        manager: Execution manager instance

    Returns:
        ActiveStatesResponse with list of active state IDs

    Raises:
        HTTPException: If execution not found
    """
    try:
        # Delegate to ExecutionManager (which delegates to StateExecutionAPI)
        active_states = manager.get_active_states(execution_id=execution_id)

        return ActiveStatesResponse(execution_id=execution_id, active_states=active_states)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to get active states: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/transitions/available", response_model=AvailableTransitionsResponse)
async def get_available_transitions(
    execution_id: str = "default", manager: ExecutionManager = Depends(get_manager)
):
    """Get available transitions from current active states.

    This endpoint delegates all state management to ExecutionManager,
    which delegates to StateExecutionAPI. REST layer never touches state directly.

    Args:
        execution_id: Execution identifier (default: "default")
        manager: Execution manager instance

    Returns:
        AvailableTransitionsResponse with list of available transitions

    Raises:
        HTTPException: If execution not found
    """
    try:
        # Delegate to ExecutionManager (which delegates to StateExecutionAPI)
        transitions = manager.get_available_transitions(execution_id=execution_id)

        return AvailableTransitionsResponse(execution_id=execution_id, transitions=transitions)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to get available transitions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
