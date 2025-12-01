"""Execution history endpoints.

This router provides endpoints for querying workflow execution history:
- Get workflow execution history
- Get all executions

NO backward compatibility - clean FastAPI code.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ..execution_manager import ExecutionManager

logger = logging.getLogger(__name__)


# ============================================================================
# Response Models
# ============================================================================


class ExecutionRecordResponse(BaseModel):
    """Execution record response model."""

    execution_id: str
    workflow_id: str
    workflow_name: str
    start_time: str
    end_time: str
    status: str
    duration: int
    total_actions: int
    completed_actions: int
    failed_actions: int
    error: str | None = None


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

router = APIRouter(prefix="/api", tags=["history"])


@router.get(
    "/workflow/{workflow_id}/history", response_model=list[ExecutionRecordResponse]
)
async def get_workflow_history(
    workflow_id: str,
    limit: int | None = Query(None, ge=1, le=1000),
    manager: ExecutionManager = Depends(get_manager),
):
    """Get execution history for a workflow.

    Args:
        workflow_id: Workflow ID
        limit: Maximum number of records to return
        manager: Execution manager instance

    Returns:
        List of execution records
    """
    try:
        history = await manager.get_execution_history(
            workflow_id=workflow_id, limit=limit
        )
        return [ExecutionRecordResponse(**record) for record in history]

    except Exception as e:
        logger.error(f"Failed to get execution history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/executions", response_model=list[ExecutionRecordResponse])
async def get_all_executions(
    limit: int | None = Query(None, ge=1, le=1000),
    manager: ExecutionManager = Depends(get_manager),
):
    """Get all execution history.

    Args:
        limit: Maximum number of records to return
        manager: Execution manager instance

    Returns:
        List of execution records
    """
    try:
        history = await manager.get_execution_history(limit=limit)
        return [ExecutionRecordResponse(**record) for record in history]

    except Exception as e:
        logger.error(f"Failed to get executions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
