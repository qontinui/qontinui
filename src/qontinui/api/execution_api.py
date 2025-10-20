"""FastAPI Execution API - REST and WebSocket endpoints for workflow execution.

This module provides:
- REST API endpoints for execution control
- WebSocket endpoint for event streaming
- Request/response models
- Error handling
- CORS configuration
"""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import Workflow
from .execution_manager import (
    ExecutionEvent,
    ExecutionManager,
    ExecutionOptions,
)

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


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


class VersionResponse(BaseModel):
    """Version information response."""

    version: str
    api_version: str
    python_version: str


class ErrorResponse(BaseModel):
    """Error response model."""

    message: str
    code: str | None = None
    details: dict[str, Any] | None = None


# ============================================================================
# FastAPI Application
# ============================================================================


def create_app(
    execution_manager: ExecutionManager | None = None,
    enable_cors: bool = True,
    cors_origins: list[str] = None,
) -> FastAPI:
    """Create FastAPI application.

    Args:
        execution_manager: Execution manager instance (creates new if None)
        enable_cors: Enable CORS middleware
        cors_origins: List of allowed CORS origins

    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="QontinUI Execution API",
        description="REST and WebSocket API for workflow execution",
        version="1.0.0",
    )

    # CORS configuration
    if enable_cors:
        if cors_origins is None:
            cors_origins = [
                "http://localhost:3000",
                "http://localhost:8000",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8000",
            ]

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Create execution manager
    if execution_manager is None:
        execution_manager = ExecutionManager()

    # Store manager in app state
    app.state.execution_manager = execution_manager

    # ========================================================================
    # Health and Version Endpoints
    # ========================================================================

    @app.get("/api/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": "1.0.0",
        }

    @app.get("/api/version", response_model=VersionResponse)
    async def get_version():
        """Get version information."""
        import sys

        return {
            "version": "1.0.0",
            "api_version": "1.0",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }

    # ========================================================================
    # Execution Control Endpoints
    # ========================================================================

    @app.post("/api/execute", response_model=ExecutionHandleResponse)
    async def execute_workflow(request: WorkflowExecutionRequest):
        """Start workflow execution.

        Args:
            request: Workflow execution request

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
            manager: ExecutionManager = app.state.execution_manager
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

    @app.get("/api/execution/{execution_id}/status", response_model=ExecutionStatusResponse)
    async def get_execution_status(execution_id: str):
        """Get execution status.

        Args:
            execution_id: Execution ID

        Returns:
            Execution status

        Raises:
            HTTPException: If execution not found
        """
        try:
            manager: ExecutionManager = app.state.execution_manager
            status = manager.get_status(execution_id)

            return ExecutionStatusResponse(**status)

        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:
            logger.error(f"Failed to get execution status: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/execution/{execution_id}/pause")
    async def pause_execution(execution_id: str):
        """Pause execution.

        Args:
            execution_id: Execution ID

        Returns:
            Success message

        Raises:
            HTTPException: If execution not found or cannot be paused
        """
        try:
            manager: ExecutionManager = app.state.execution_manager
            await manager.pause_execution(execution_id)

            return {"message": "Execution paused"}

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error(f"Failed to pause execution: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/execution/{execution_id}/resume")
    async def resume_execution(execution_id: str):
        """Resume execution.

        Args:
            execution_id: Execution ID

        Returns:
            Success message

        Raises:
            HTTPException: If execution not found or cannot be resumed
        """
        try:
            manager: ExecutionManager = app.state.execution_manager
            await manager.resume_execution(execution_id)

            return {"message": "Execution resumed"}

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error(f"Failed to resume execution: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/execution/{execution_id}/step")
    async def step_execution(execution_id: str):
        """Step execution (execute next action).

        Args:
            execution_id: Execution ID

        Returns:
            Success message

        Raises:
            HTTPException: If execution not found or not in step mode
        """
        try:
            manager: ExecutionManager = app.state.execution_manager
            await manager.step_execution(execution_id)

            return {"message": "Execution stepped"}

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error(f"Failed to step execution: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.post("/api/execution/{execution_id}/cancel")
    async def cancel_execution(execution_id: str):
        """Cancel execution.

        Args:
            execution_id: Execution ID

        Returns:
            Success message

        Raises:
            HTTPException: If execution not found
        """
        try:
            manager: ExecutionManager = app.state.execution_manager
            await manager.cancel_execution(execution_id)

            return {"message": "Execution cancelled"}

        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:
            logger.error(f"Failed to cancel execution: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    # ========================================================================
    # History Endpoints
    # ========================================================================

    @app.get("/api/workflow/{workflow_id}/history", response_model=list[ExecutionRecordResponse])
    async def get_workflow_history(
        workflow_id: str, limit: int | None = Query(None, ge=1, le=1000)
    ):
        """Get execution history for a workflow.

        Args:
            workflow_id: Workflow ID
            limit: Maximum number of records to return

        Returns:
            List of execution records
        """
        try:
            manager: ExecutionManager = app.state.execution_manager
            history = manager.get_execution_history(workflow_id=workflow_id, limit=limit)

            return [ExecutionRecordResponse(**record) for record in history]

        except Exception as e:
            logger.error(f"Failed to get execution history: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get("/api/executions", response_model=list[ExecutionRecordResponse])
    async def get_all_executions(limit: int | None = Query(None, ge=1, le=1000)):
        """Get all execution history.

        Args:
            limit: Maximum number of records to return

        Returns:
            List of execution records
        """
        try:
            manager: ExecutionManager = app.state.execution_manager
            history = manager.get_execution_history(limit=limit)

            return [ExecutionRecordResponse(**record) for record in history]

        except Exception as e:
            logger.error(f"Failed to get executions: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    # ========================================================================
    # WebSocket Endpoint
    # ========================================================================

    @app.websocket("/api/execution/{execution_id}/stream")
    async def stream_execution_events(websocket: WebSocket, execution_id: str):
        """Stream execution events via WebSocket.

        Args:
            websocket: WebSocket connection
            execution_id: Execution ID

        Raises:
            WebSocketDisconnect: When client disconnects
        """
        await websocket.accept()
        logger.info(f"WebSocket connected: {execution_id}")

        manager: ExecutionManager = app.state.execution_manager

        # Check if execution exists
        try:
            manager.get_status(execution_id)
        except ValueError:
            await websocket.close(code=1003, reason="Execution not found")
            return

        # Event callback
        async def send_event(event: ExecutionEvent):
            """Send event to WebSocket client."""
            try:
                await websocket.send_json(event.to_dict())
            except Exception as e:
                logger.error(f"Failed to send event: {e}")

        # Subscribe to events
        manager.subscribe_to_events(execution_id, send_event)

        try:
            # Keep connection alive and handle ping/pong
            while True:
                data = await websocket.receive_json()

                # Handle ping
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {execution_id}")

        finally:
            # Unsubscribe from events
            manager.unsubscribe_from_events(execution_id, send_event)

    return app


# ============================================================================
# Main Entry Point (for testing)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Create app
    app = create_app()

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
