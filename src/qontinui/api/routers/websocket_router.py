"""WebSocket streaming endpoint.

This router provides WebSocket endpoint for real-time execution event streaming.

NO backward compatibility - clean FastAPI code.
"""

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..execution_manager import ExecutionEvent, ExecutionManager

logger = logging.getLogger(__name__)


# ============================================================================
# Router
# ============================================================================

router = APIRouter(prefix="/api", tags=["websocket"])


def get_manager_from_websocket(websocket: WebSocket) -> ExecutionManager:
    """Get execution manager from WebSocket app state.

    Args:
        websocket: WebSocket connection

    Returns:
        ExecutionManager instance from app state
    """
    return websocket.app.state.execution_manager


@router.websocket("/execution/{execution_id}/stream")
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

    manager = get_manager_from_websocket(websocket)

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
    await manager.subscribe_to_events(execution_id, send_event)

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
        await manager.unsubscribe_from_events(execution_id, send_event)
