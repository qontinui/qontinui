"""Execution control endpoints.

This router provides workflow execution control endpoints:
- Start execution
- Pause execution
- Resume execution
- Cancel execution
- Step execution

NO backward compatibility - clean FastAPI code.
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...config import Workflow
from ..execution_manager import ExecutionManager, ExecutionOptions

logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================


class ExecutionOptionsRequest(BaseModel):
    """Execution options request model."""

    initial_variables: dict[str, Any] = Field(default_factory=dict)
    debug_mode: bool = False
    breakpoints: list[str] = Field(default_factory=list)
    step_mode: bool = False
    timeout: int = 0
    continue_on_error: bool = False


class WorkflowExecutionRequest(BaseModel):
    """Workflow execution request model."""

    workflow: dict[str, Any]  # Workflow JSON
    options: ExecutionOptionsRequest = Field(default_factory=ExecutionOptionsRequest)


class ExecutionHandleResponse(BaseModel):
    """Execution handle response model."""

    execution_id: str
    workflow_id: str
    workflow_name: str
    start_time: str
    status: str
    stream_url: str


class ExecutionStatusResponse(BaseModel):
    """Execution status response model."""

    execution_id: str
    workflow_id: str
    status: str
    start_time: str
    end_time: str | None = None
    current_action: str | None = None
    progress: float
    total_actions: int
    completed_actions: int
    failed_actions: int
    skipped_actions: int
    action_states: dict[str, str]
    error: dict[str, Any] | None = None
    variables: dict[str, Any] | None = None


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

router = APIRouter(prefix="/api", tags=["execution"])


@router.post("/execute", response_model=ExecutionHandleResponse)
async def execute_workflow(
    request: WorkflowExecutionRequest, manager: ExecutionManager = Depends(get_manager)
):
    """Start workflow execution.

    Args:
        request: Workflow execution request
        manager: Execution manager instance

    Returns:
        Execution handle with execution ID and stream URL

    Raises:
        HTTPException: If execution fails to start
    """
    try:
        # Parse workflow
        workflow_data = request.workflow
        workflow = Workflow(**workflow_data)

        # Convert options
        options = ExecutionOptions(
            initial_variables=request.options.initial_variables,
            debug_mode=request.options.debug_mode,
            breakpoints=request.options.breakpoints,
            step_mode=request.options.step_mode,
            timeout=request.options.timeout,
            continue_on_error=request.options.continue_on_error,
        )

        # Start execution
        execution_id = await manager.start_execution(workflow, options)

        # Get status
        status = manager.get_status(execution_id)

        return {
            "execution_id": execution_id,
            "workflow_id": workflow.id,
            "workflow_name": workflow.name,
            "start_time": status["start_time"],
            "status": status["status"],
            "stream_url": f"/api/execution/{execution_id}/stream",
        }

    except Exception as e:
        logger.error(f"Failed to start execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/execution/{execution_id}/status", response_model=ExecutionStatusResponse)
async def get_execution_status(execution_id: str, manager: ExecutionManager = Depends(get_manager)):
    """Get execution status.

    Args:
        execution_id: Execution ID
        manager: Execution manager instance

    Returns:
        Execution status

    Raises:
        HTTPException: If execution not found
    """
    try:
        status = manager.get_status(execution_id)
        return ExecutionStatusResponse(**status)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to get execution status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/execution/{execution_id}/pause")
async def pause_execution(execution_id: str, manager: ExecutionManager = Depends(get_manager)):
    """Pause execution.

    Args:
        execution_id: Execution ID
        manager: Execution manager instance

    Returns:
        Success message

    Raises:
        HTTPException: If execution not found or cannot be paused
    """
    try:
        await manager.pause_execution(execution_id)
        return {"message": "Execution paused"}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to pause execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/execution/{execution_id}/resume")
async def resume_execution(execution_id: str, manager: ExecutionManager = Depends(get_manager)):
    """Resume execution.

    Args:
        execution_id: Execution ID
        manager: Execution manager instance

    Returns:
        Success message

    Raises:
        HTTPException: If execution not found or cannot be resumed
    """
    try:
        await manager.resume_execution(execution_id)
        return {"message": "Execution resumed"}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to resume execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/execution/{execution_id}/step")
async def step_execution(execution_id: str, manager: ExecutionManager = Depends(get_manager)):
    """Step execution (execute next action).

    Args:
        execution_id: Execution ID
        manager: Execution manager instance

    Returns:
        Success message

    Raises:
        HTTPException: If execution not found or not in step mode
    """
    try:
        await manager.step_execution(execution_id)
        return {"message": "Execution stepped"}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to step execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/execution/{execution_id}/cancel")
async def cancel_execution(execution_id: str, manager: ExecutionManager = Depends(get_manager)):
    """Cancel execution.

    Args:
        execution_id: Execution ID
        manager: Execution manager instance

    Returns:
        Success message

    Raises:
        HTTPException: If execution not found
    """
    try:
        await manager.cancel_execution(execution_id)
        return {"message": "Execution cancelled"}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to cancel execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
