"""FastAPI routers for modular endpoint organization.

This package contains focused router modules for different API endpoint groups:
- health_router: Health check and version endpoints
- execution_router: Execution control endpoints (start, pause, resume, cancel, step)
- history_router: Execution history endpoints
- state_router: State management endpoints
- websocket_router: WebSocket streaming endpoint

NO backward compatibility - clean FastAPI router pattern.
"""

from . import (
    execution_router,
    health_router,
    history_router,
    state_router,
    websocket_router,
)

__all__ = [
    "health_router",
    "execution_router",
    "history_router",
    "state_router",
    "websocket_router",
]
