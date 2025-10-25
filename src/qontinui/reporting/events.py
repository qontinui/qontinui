"""Core event system implementation for qontinui reporting.

Zero-overhead callback system for external consumers to monitor library events.
Thread-safe, optional, and designed for minimal performance impact.
"""

import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class EventType(Enum):
    """Types of events emitted by qontinui."""

    # Match events (image recognition)
    MATCH_ATTEMPTED = "match.attempted"
    MATCH_FOUND = "match.found"
    MATCH_NOT_FOUND = "match.not_found"

    # Action events
    ACTION_STARTED = "action.started"
    ACTION_COMPLETED = "action.completed"
    ACTION_FAILED = "action.failed"

    # Keyboard events
    TEXT_TYPED = "keyboard.text_typed"

    # Mouse events
    MOUSE_CLICKED = "mouse.clicked"
    MOUSE_MOVED = "mouse.moved"
    MOUSE_DRAGGED = "mouse.dragged"

    # State events
    STATE_DETECTED = "state.detected"
    STATE_TRANSITION_STARTED = "state.transition.started"
    STATE_TRANSITION_COMPLETED = "state.transition.completed"


@dataclass
class Event:
    """Event data structure emitted by qontinui library."""

    type: EventType
    """Type of event."""

    data: dict[str, Any] = field(default_factory=dict)
    """Event payload data."""

    timestamp: float = field(default_factory=time.time)
    """Event timestamp."""

    thread_id: int = field(default_factory=threading.get_ident)
    """Thread that emitted the event."""

    context: dict[str, Any] = field(default_factory=dict)
    """Additional context metadata."""


# Type alias for callback functions
EventCallback = Callable[[Event], None]


class EventRegistry:
    """Thread-safe global registry for event callbacks.

    Features:
    - Zero overhead when no callbacks registered (single bool check)
    - Thread-safe registration/unregistration
    - Support for wildcard subscriptions
    - Automatic error isolation (callback failures don't affect library)
    """

    def __init__(self):
        """Initialize event registry."""
        self._callbacks: dict[EventType, list[EventCallback]] = defaultdict(list)
        self._wildcard_callbacks: list[EventCallback] = []
        self._lock = RLock()
        self._enabled = True

    @property
    def has_listeners(self) -> bool:
        """Fast check if any listeners are registered.

        This is the critical fast path - optimized for performance.

        Returns:
            True if any callbacks registered
        """
        # Single bool check + len check - very fast (~2-5 nanoseconds)
        return self._enabled and (bool(self._callbacks) or bool(self._wildcard_callbacks))

    def register(
        self,
        event_type: EventType | None,
        callback: EventCallback,
    ) -> None:
        """Register a callback for specific event type(s).

        Args:
            event_type: Type to listen for (None for all events)
            callback: Function to call when event occurs

        Example:
            >>> def my_handler(event: Event):
            ...     print(f"Match confidence: {event.data['confidence']}")
            >>>
            >>> registry.register(EventType.MATCH_ATTEMPTED, my_handler)
        """
        with self._lock:
            if event_type is None:
                # Wildcard subscription
                if callback not in self._wildcard_callbacks:
                    self._wildcard_callbacks.append(callback)
            else:
                if callback not in self._callbacks[event_type]:
                    self._callbacks[event_type].append(callback)

    def unregister(
        self,
        event_type: EventType | None,
        callback: EventCallback,
    ) -> None:
        """Unregister a callback.

        Args:
            event_type: Event type (None for wildcard)
            callback: Callback to remove
        """
        with self._lock:
            if event_type is None:
                if callback in self._wildcard_callbacks:
                    self._wildcard_callbacks.remove(callback)
            else:
                if callback in self._callbacks[event_type]:
                    self._callbacks[event_type].remove(callback)

    def emit(self, event: Event) -> None:
        """Emit an event to all registered callbacks.

        Args:
            event: Event to emit

        Note:
            Callbacks are executed synchronously but errors are isolated.
            Failed callbacks don't affect other callbacks or library operation.
        """
        # Fast path: no listeners (critical for performance)
        if not self.has_listeners:
            return

        # Get snapshot of callbacks to avoid holding lock during execution
        with self._lock:
            callbacks = list(self._callbacks.get(event.type, []))
            callbacks.extend(self._wildcard_callbacks)

        # Execute callbacks outside of lock
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                # Isolate callback failures - never let them break library
                # Log exception to help debug event system issues
                import logging
                logger = logging.getLogger(__name__)
                logger.error(
                    f"Event callback failed for {event.type.value}: {e}",
                    exc_info=True,
                    extra={"event_type": event.type.value, "callback": callback.__name__}
                )
                # Still continue to other callbacks

    def clear(self) -> None:
        """Clear all registered callbacks."""
        with self._lock:
            self._callbacks.clear()
            self._wildcard_callbacks.clear()

    def enable(self) -> None:
        """Enable event emission."""
        self._enabled = True

    def disable(self) -> None:
        """Disable event emission (for performance)."""
        self._enabled = False


# Global singleton registry
_event_registry = EventRegistry()


def get_event_registry() -> EventRegistry:
    """Get the global event registry.

    Returns:
        Global event registry instance
    """
    return _event_registry


# Convenience functions for common usage patterns


def register_callback(
    event_type: EventType | None,
    callback: EventCallback,
) -> None:
    """Register an event callback.

    Args:
        event_type: Event type to listen for (None for all)
        callback: Callback function

    Example:
        >>> from qontinui.reporting import register_callback, EventType
        >>>
        >>> def on_match(event):
        ...     print(f"Match confidence: {event.data['confidence']:.2%}")
        >>>
        >>> register_callback(EventType.MATCH_ATTEMPTED, on_match)
    """
    _event_registry.register(event_type, callback)


def unregister_callback(
    event_type: EventType | None,
    callback: EventCallback,
) -> None:
    """Unregister an event callback.

    Args:
        event_type: Event type
        callback: Callback to remove
    """
    _event_registry.unregister(event_type, callback)


def emit_event(event_type: EventType, data: dict[str, Any] | None = None, **kwargs) -> None:
    """Emit an event (internal library use).

    Args:
        event_type: Type of event
        data: Event data payload
        **kwargs: Additional context

    Example (internal library usage):
        >>> # In match code
        >>> emit_event(
        ...     EventType.MATCH_ATTEMPTED,
        ...     data={
        ...         "confidence": 0.92,
        ...         "threshold": 0.90,
        ...         "passed": True,
        ...     },
        ... )
    """
    # Fast path: no listeners registered (critical for performance)
    if not _event_registry.has_listeners:
        return

    event = Event(
        type=event_type,
        data=data or {},
        context=kwargs,
    )
    _event_registry.emit(event)


class EventCollector:
    """Thread-local event collector for capturing events in a context.

    Useful for testing or collecting events within a specific scope without
    affecting global state.

    Example:
        >>> collector = EventCollector([EventType.MATCH_ATTEMPTED])
        >>> with collector:
        ...     # Perform actions
        ...     Find(image).execute()
        >>>
        >>> events = collector.get_events()
        >>> print(f"Captured {len(events)} match attempts")
    """

    def __init__(self, event_types: list[EventType] | None = None):
        """Initialize collector.

        Args:
            event_types: Specific event types to collect (None for all)
        """
        self._event_types = set(event_types) if event_types else None
        self._events: list[Event] = []
        self._collecting = False

    def _collect(self, event: Event) -> None:
        """Internal callback to collect events.

        Args:
            event: Event to collect
        """
        if self._event_types is None or event.type in self._event_types:
            self._events.append(event)

    def __enter__(self) -> "EventCollector":
        """Enter context and start collecting."""
        self._events.clear()
        self._collecting = True

        # Register collector callback
        if self._event_types is None:
            _event_registry.register(None, self._collect)
        else:
            for event_type in self._event_types:
                _event_registry.register(event_type, self._collect)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and stop collecting."""
        self._collecting = False

        # Unregister collector callback
        if self._event_types is None:
            _event_registry.unregister(None, self._collect)
        else:
            for event_type in self._event_types:
                _event_registry.unregister(event_type, self._collect)

    def get_events(self, event_type: EventType | None = None) -> list[Event]:
        """Get collected events.

        Args:
            event_type: Filter by type (None for all)

        Returns:
            List of collected events
        """
        if event_type is None:
            return list(self._events)
        return [e for e in self._events if e.type == event_type]

    def clear(self) -> None:
        """Clear collected events."""
        self._events.clear()

    def count(self, event_type: EventType | None = None) -> int:
        """Count events.

        Args:
            event_type: Type to count (None for all)

        Returns:
            Number of events
        """
        return len(self.get_events(event_type))
