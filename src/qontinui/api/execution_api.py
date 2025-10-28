"""FastAPI Execution API - REST and WebSocket endpoints for workflow execution.

This module provides:
- Router composition for modular endpoint organization
- CORS configuration
- Dependency injection for ExecutionManager
- Error handling

NO backward compatibility - clean FastAPI code.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .execution_manager import ExecutionManager
from .routers import (
    execution_router,
    health_router,
    history_router,
    state_router,
    websocket_router,
)

logger = logging.getLogger(__name__)


# ============================================================================
# FastAPI Application
# ============================================================================


def create_app(
    execution_manager: ExecutionManager | None = None,
    enable_cors: bool = True,
    cors_origins: list[str] = None,
) -> FastAPI:
    """Create FastAPI application with router composition.

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
    # Dependency Injection Override
    # ========================================================================

    def get_manager_override() -> ExecutionManager:
        """Get execution manager from app state.

        Returns:
            ExecutionManager instance
        """
        return app.state.execution_manager

    # Override the get_manager dependency in all routers
    app.dependency_overrides[execution_router.get_manager] = get_manager_override
    app.dependency_overrides[history_router.get_manager] = get_manager_override
    app.dependency_overrides[state_router.get_manager] = get_manager_override

    # ========================================================================
    # Include Routers
    # ========================================================================

    app.include_router(health_router.router)
    app.include_router(execution_router.router)
    app.include_router(history_router.router)
    app.include_router(state_router.router)
    app.include_router(websocket_router.router)

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
