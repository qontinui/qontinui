"""Optional event reporting system for qontinui library.

This module provides a lightweight, zero-overhead event system that allows external
consumers (like qontinui-runner or qontinui-web) to receive diagnostic information
from the library without impacting performance when not in use.

Example:
    # Register a callback to receive events
    from qontinui.reporting import register_callback, EventType

    def on_match_attempt(event):
        print(f"Match confidence: {event.data['confidence']}")

    register_callback(EventType.MATCH_ATTEMPTED, on_match_attempt)
"""

from .events import (
    Event,
    EventCallback,
    EventCollector,
    EventRegistry,
    EventType,
    emit_event,
    get_event_registry,
    register_callback,
    unregister_callback,
)
from .schemas import (
    ActionCompletedData,
    ActionFailedData,
    ActionStartedData,
    EventData,
    EventDataType,
    MatchAttemptedData,
)

__all__ = [
    "Event",
    "EventCallback",
    "EventCollector",
    "EventRegistry",
    "EventType",
    "emit_event",
    "get_event_registry",
    "register_callback",
    "unregister_callback",
    # Schemas
    "EventData",
    "MatchAttemptedData",
    "ActionStartedData",
    "ActionCompletedData",
    "ActionFailedData",
    "EventDataType",
]
