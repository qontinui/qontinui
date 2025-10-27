# Event/Reporting System Architecture Design

## Executive Summary

This document outlines the design for a thread-safe, zero-dependency event/reporting system for the qontinui library that enables diagnostic event emission to external consumers (runner, web, etc.) without coupling the library to specific consumers.

**Recommended Approach:** Context-based with global registry pattern (hybrid approach)

This combines the best aspects of multiple patterns:
- Context-based scoping for hierarchical event relationships
- Global registry for decoupled consumer registration
- Queue-based buffering for thread safety
- Minimal overhead when disabled (null object pattern)

---

## 1. Architecture Overview

### 1.1 Design Principles

1. **Zero Dependencies**: No external libraries required (uses only stdlib: threading, queue, contextvars)
2. **Opt-in by Default**: No performance impact when disabled
3. **Thread-Safe**: Multiple actions can emit events concurrently
4. **Decoupled**: Library code doesn't know about consumers
5. **Hierarchical**: Events can be nested (action -> image_search -> template_match)
6. **Type-Safe**: Clear event schemas with dataclasses

### 1.2 Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│  Qontinui Library (Event Producers)                     │
├─────────────────────────────────────────────────────────┤
│  ActionExecution  │  Find  │  HAL (OpenCV)  │  etc.    │
│        ↓          │    ↓   │       ↓        │           │
│    EventEmitter.emit(event)                             │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│  Event System Core (src/qontinui/events/)               │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐         ┌─────────────────┐          │
│  │ EventEmitter │────────>│ EventRegistry   │          │
│  │  (singleton) │         │  (global)       │          │
│  └──────────────┘         └────────┬────────┘          │
│         │                           │                   │
│         │                    ┌──────▼──────┐            │
│         │                    │  Handlers   │            │
│         │                    │  (List)     │            │
│         │                    └──────┬──────┘            │
│         ▼                           │                   │
│  ┌──────────────┐                  │                   │
│  │ EventContext │                  │                   │
│  │ (contextvar) │                  │                   │
│  └──────────────┘                  │                   │
│         │                           │                   │
│         └───────────┬───────────────┘                   │
│                     ▼                                   │
│            ┌────────────────┐                           │
│            │ Event Queue    │                           │
│            │ (thread-safe)  │                           │
│            └────────┬───────┘                           │
└─────────────────────┼───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│  Consumers                                              │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Runner    │  │   Web Mock   │  │   Testing    │  │
│  │  (Bridge)   │  │   (Future)   │  │  (Capture)   │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Core Components

### 2.1 Event Schema (events/schema.py)

```python
"""Event schema definitions for qontinui diagnostic events."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4


class EventType(Enum):
    """Types of diagnostic events."""

    # Action lifecycle
    ACTION_STARTED = "action_started"
    ACTION_COMPLETED = "action_completed"
    ACTION_FAILED = "action_failed"

    # Image search events
    IMAGE_SEARCH_STARTED = "image_search_started"
    IMAGE_SEARCH_MATCH_FOUND = "image_search_match_found"
    IMAGE_SEARCH_COMPLETED = "image_search_completed"

    # Template matching (HAL level)
    TEMPLATE_MATCH_ATTEMPT = "template_match_attempt"
    TEMPLATE_MATCH_RESULT = "template_match_result"

    # State transitions
    STATE_TRANSITION = "state_transition"
    STATE_DETECTED = "state_detected"

    # Performance metrics
    PERFORMANCE_METRIC = "performance_metric"


class EventLevel(Enum):
    """Event severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Event:
    """Base event class for all diagnostic events.

    Attributes:
        id: Unique event identifier
        type: Type of event
        timestamp: When the event occurred
        level: Severity level
        parent_id: ID of parent event (for hierarchical relationships)
        correlation_id: ID linking related events across operations
        thread_id: Thread that generated the event
        data: Event-specific data payload
    """

    id: UUID = field(default_factory=uuid4)
    type: EventType = EventType.ACTION_STARTED
    timestamp: datetime = field(default_factory=datetime.now)
    level: EventLevel = EventLevel.INFO
    parent_id: Optional[UUID] = None
    correlation_id: Optional[UUID] = None
    thread_id: int = field(default_factory=lambda: __import__('threading').get_ident())
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "id": str(self.id),
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "thread_id": self.thread_id,
            "data": self.data,
        }


# Event-specific data schemas for type safety
@dataclass
class ImageSearchData:
    """Data payload for image search events."""

    pattern_name: str
    search_region: Optional[tuple[int, int, int, int]] = None  # x, y, w, h
    confidence_threshold: float = 0.9
    grayscale: bool = False
    match_count: Optional[int] = None
    best_confidence: Optional[float] = None
    duration_ms: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_name": self.pattern_name,
            "search_region": self.search_region,
            "confidence_threshold": self.confidence_threshold,
            "grayscale": self.grayscale,
            "match_count": self.match_count,
            "best_confidence": self.best_confidence,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ActionExecutionData:
    """Data payload for action execution events."""

    action_type: str
    action_description: str
    success: Optional[bool] = None
    duration_ms: Optional[float] = None
    error_message: Optional[str] = None
    match_count: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_type": self.action_type,
            "action_description": self.action_description,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "match_count": self.match_count,
        }


@dataclass
class TemplateMatchData:
    """Data payload for template matching events."""

    haystack_size: tuple[int, int]  # width, height
    needle_size: tuple[int, int]  # width, height
    method: str = "TM_CCOEFF_NORMED"
    confidence_threshold: float = 0.9
    found: Optional[bool] = None
    confidence: Optional[float] = None
    location: Optional[tuple[int, int]] = None  # x, y
    duration_ms: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "haystack_size": self.haystack_size,
            "needle_size": self.needle_size,
            "method": self.method,
            "confidence_threshold": self.confidence_threshold,
            "found": self.found,
            "confidence": self.confidence,
            "location": self.location,
            "duration_ms": self.duration_ms,
        }
```

### 2.2 Event Registry (events/registry.py)

```python
"""Global event handler registry."""

import threading
from typing import Callable, Protocol


class EventHandler(Protocol):
    """Protocol for event handlers."""

    def __call__(self, event: "Event") -> None:
        """Handle an event.

        Args:
            event: The event to handle
        """
        ...


class EventRegistry:
    """Thread-safe global registry for event handlers.

    This is a singleton that maintains a list of registered event handlers.
    Handlers are called synchronously in the order they were registered.
    """

    _instance: Optional["EventRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize registry (only once)."""
        if not self._initialized:
            self._handlers: list[EventHandler] = []
            self._handler_lock = threading.Lock()
            self._enabled = False
            self._initialized = True

    def register(self, handler: EventHandler) -> None:
        """Register an event handler.

        Args:
            handler: Callable that accepts an Event
        """
        with self._handler_lock:
            if handler not in self._handlers:
                self._handlers.append(handler)
                # Auto-enable when first handler is registered
                if not self._enabled:
                    self._enabled = True

    def unregister(self, handler: EventHandler) -> None:
        """Unregister an event handler.

        Args:
            handler: Handler to remove
        """
        with self._handler_lock:
            if handler in self._handlers:
                self._handlers.remove(handler)
                # Auto-disable when last handler is removed
                if not self._handlers:
                    self._enabled = False

    def clear(self) -> None:
        """Remove all handlers and disable events."""
        with self._handler_lock:
            self._handlers.clear()
            self._enabled = False

    def emit(self, event: "Event") -> None:
        """Emit an event to all registered handlers.

        Args:
            event: Event to emit
        """
        # Fast path: skip if disabled
        if not self._enabled:
            return

        # Get snapshot of handlers (avoid holding lock during callbacks)
        with self._handler_lock:
            handlers = list(self._handlers)

        # Call handlers outside of lock to prevent deadlocks
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Silently ignore handler errors to prevent breaking library code
                # In production, you might want to log this
                pass

    @property
    def is_enabled(self) -> bool:
        """Check if event system is enabled."""
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Manually enable/disable event emission.

        Args:
            enabled: True to enable, False to disable
        """
        with self._handler_lock:
            self._enabled = enabled
```

### 2.3 Event Context (events/context.py)

```python
"""Context management for hierarchical events."""

from contextvars import ContextVar
from typing import Optional
from uuid import UUID, uuid4


# Context variable for tracking parent event ID
_parent_event_id: ContextVar[Optional[UUID]] = ContextVar("parent_event_id", default=None)

# Context variable for correlation ID (links related operations)
_correlation_id: ContextVar[Optional[UUID]] = ContextVar("correlation_id", default=None)


class EventContext:
    """Context manager for hierarchical event relationships.

    Usage:
        with EventContext(parent_event_id=action_event.id):
            # Child events will automatically have parent_id set
            emit_image_search_event(...)
    """

    def __init__(
        self,
        parent_event_id: Optional[UUID] = None,
        correlation_id: Optional[UUID] = None,
        auto_correlate: bool = False
    ):
        """Initialize event context.

        Args:
            parent_event_id: ID of parent event
            correlation_id: ID to correlate related events
            auto_correlate: Auto-generate correlation_id if not provided
        """
        self.parent_event_id = parent_event_id
        self.correlation_id = correlation_id or (uuid4() if auto_correlate else None)
        self._parent_token = None
        self._correlation_token = None

    def __enter__(self):
        """Enter context and set parent/correlation IDs."""
        self._parent_token = _parent_event_id.set(self.parent_event_id)
        if self.correlation_id:
            self._correlation_token = _correlation_id.set(self.correlation_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore previous IDs."""
        _parent_event_id.reset(self._parent_token)
        if self._correlation_token:
            _correlation_id.reset(self._correlation_token)
        return False


def get_current_parent_id() -> Optional[UUID]:
    """Get the current parent event ID from context."""
    return _parent_event_id.get()


def get_current_correlation_id() -> Optional[UUID]:
    """Get the current correlation ID from context."""
    return _correlation_id.get()
```

### 2.4 Event Emitter (events/emitter.py)

```python
"""Event emission facade."""

from typing import Any, Optional
from uuid import UUID

from .context import get_current_correlation_id, get_current_parent_id
from .registry import EventRegistry
from .schema import Event, EventLevel, EventType


class EventEmitter:
    """Facade for emitting events.

    This provides a simple API for library code to emit events without
    needing to know about the registry or context management.
    """

    def __init__(self):
        """Initialize emitter."""
        self._registry = EventRegistry()

    def emit(
        self,
        event_type: EventType,
        data: dict[str, Any],
        level: EventLevel = EventLevel.INFO,
        parent_id: Optional[UUID] = None,
        correlation_id: Optional[UUID] = None,
    ) -> Event:
        """Emit an event.

        Args:
            event_type: Type of event
            data: Event data payload
            level: Event severity level
            parent_id: Override parent ID (otherwise uses context)
            correlation_id: Override correlation ID (otherwise uses context)

        Returns:
            The created event (for use as parent in child events)
        """
        # Use context values if not explicitly provided
        if parent_id is None:
            parent_id = get_current_parent_id()
        if correlation_id is None:
            correlation_id = get_current_correlation_id()

        # Create event
        event = Event(
            type=event_type,
            level=level,
            parent_id=parent_id,
            correlation_id=correlation_id,
            data=data,
        )

        # Emit to registry
        self._registry.emit(event)

        return event

    @property
    def is_enabled(self) -> bool:
        """Check if events are enabled."""
        return self._registry.is_enabled


# Global singleton instance
_emitter = EventEmitter()


def emit_event(
    event_type: EventType,
    data: dict[str, Any],
    level: EventLevel = EventLevel.INFO,
    parent_id: Optional[UUID] = None,
    correlation_id: Optional[UUID] = None,
) -> Event:
    """Convenience function to emit an event.

    Args:
        event_type: Type of event
        data: Event data payload
        level: Event severity level
        parent_id: Override parent ID
        correlation_id: Override correlation ID

    Returns:
        The created event
    """
    return _emitter.emit(event_type, data, level, parent_id, correlation_id)


def is_event_system_enabled() -> bool:
    """Check if event system is enabled."""
    return _emitter.is_enabled
```

### 2.5 Public API (events/__init__.py)

```python
"""Event system public API.

This module provides diagnostic event emission for external consumers.
The event system is:
- Opt-in: No performance impact when disabled
- Thread-safe: Can be used from multiple threads
- Decoupled: Library doesn't depend on consumers
- Hierarchical: Events can have parent-child relationships
"""

from .context import EventContext, get_current_correlation_id, get_current_parent_id
from .emitter import EventEmitter, emit_event, is_event_system_enabled
from .registry import EventHandler, EventRegistry
from .schema import (
    ActionExecutionData,
    Event,
    EventLevel,
    EventType,
    ImageSearchData,
    TemplateMatchData,
)

__all__ = [
    # Core classes
    "Event",
    "EventType",
    "EventLevel",
    "EventEmitter",
    "EventRegistry",
    "EventContext",
    "EventHandler",
    # Data schemas
    "ImageSearchData",
    "ActionExecutionData",
    "TemplateMatchData",
    # Functions
    "emit_event",
    "is_event_system_enabled",
    "get_current_parent_id",
    "get_current_correlation_id",
]
```

---

## 3. Integration Examples

### 3.1 Library Side: Emitting Events

#### Example 1: ActionExecution with context

```python
# In src/qontinui/actions/action_execution.py

from ..events import EventContext, EventLevel, EventType, emit_event

class ActionExecution:
    def perform(
        self,
        action: ActionInterface,
        action_description: str,
        action_config: ActionConfig,
        object_collections: tuple[ObjectCollection, ...],
    ) -> ActionResult:
        # Emit action started event and create context
        start_event = emit_event(
            EventType.ACTION_STARTED,
            data={
                "action_type": action.__class__.__name__,
                "action_description": action_description,
            },
        )

        # Create context for child events
        with EventContext(
            parent_event_id=start_event.id,
            correlation_id=start_event.id,  # Use action ID for correlation
        ):
            result = ActionResult(action_config)
            start_time = time.time()

            try:
                # Pre-action pause
                pause_before = action_config.get_pause_before_begin()
                if pause_before > 0:
                    time.sleep(pause_before)

                # Execute action (child events will have parent_id set)
                self._execution_count += 1
                action.perform(result, *object_collections)

                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000

                # Emit completion event
                emit_event(
                    EventType.ACTION_COMPLETED,
                    data={
                        "action_type": action.__class__.__name__,
                        "action_description": action_description,
                        "success": result.success,
                        "duration_ms": duration_ms,
                        "match_count": len(result.match_list),
                    },
                )

                if result.success:
                    self._success_count += 1
                else:
                    self._failure_count += 1

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                # Emit failure event
                emit_event(
                    EventType.ACTION_FAILED,
                    level=EventLevel.ERROR,
                    data={
                        "action_type": action.__class__.__name__,
                        "action_description": action_description,
                        "error_message": str(e),
                        "duration_ms": duration_ms,
                    },
                )

                self._failure_count += 1
                result.success = False
                result.output_text = f"Error: {str(e)}"

            return result
```

#### Example 2: HAL OpenCV Matcher

```python
# In src/qontinui/hal/implementations/opencv_matcher.py

from ...events import EventLevel, EventType, emit_event

class OpenCVMatcher(IPatternMatcher):
    def find_pattern(
        self,
        haystack: Image.Image,
        needle: Image.Image,
        confidence: float = 0.9,
        grayscale: bool = False,
    ) -> Match | None:
        # Emit search started event
        start_time = time.time()
        search_event = emit_event(
            EventType.IMAGE_SEARCH_STARTED,
            data={
                "haystack_size": haystack.size,
                "needle_size": needle.size,
                "confidence_threshold": confidence,
                "grayscale": grayscale,
            },
        )

        try:
            # Convert images
            haystack_cv = self._pil_to_cv2(haystack)
            needle_cv = self._pil_to_cv2(needle)

            if grayscale:
                haystack_cv = cv2.cvtColor(haystack_cv, cv2.COLOR_BGR2GRAY)
                needle_cv = cv2.cvtColor(needle_cv, cv2.COLOR_BGR2GRAY)

            # Perform template matching
            result = cv2.matchTemplate(haystack_cv, needle_cv, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            duration_ms = (time.time() - start_time) * 1000

            if max_val >= confidence:
                h, w = needle_cv.shape[:2]
                x, y = max_loc

                match = Match(
                    x=x, y=y, width=w, height=h,
                    confidence=float(max_val),
                    center=(x + w // 2, y + h // 2),
                )

                # Emit success event
                emit_event(
                    EventType.IMAGE_SEARCH_MATCH_FOUND,
                    data={
                        "location": (x, y),
                        "confidence": float(max_val),
                        "duration_ms": duration_ms,
                    },
                )

                return match
            else:
                # Emit no match event
                emit_event(
                    EventType.IMAGE_SEARCH_COMPLETED,
                    level=EventLevel.WARNING,
                    data={
                        "match_count": 0,
                        "best_confidence": float(max_val),
                        "duration_ms": duration_ms,
                    },
                )
                return None

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            emit_event(
                EventType.IMAGE_SEARCH_COMPLETED,
                level=EventLevel.ERROR,
                data={
                    "error_message": str(e),
                    "duration_ms": duration_ms,
                },
            )
            return None
```

### 3.2 Consumer Side: Runner (Python Bridge)

```python
# In qontinui-runner/python-bridge/qontinui_bridge.py

from qontinui.events import EventRegistry, Event

class QontinuiBridge:
    def __init__(self, mock_mode: bool = False):
        MockModeManager.set_mock_mode(mock_mode)
        self.runner = JSONRunner()
        self._sequence = 0

        # Register event handler
        self._setup_event_handler()

    def _setup_event_handler(self):
        """Register handler for qontinui diagnostic events."""
        registry = EventRegistry()
        registry.register(self._handle_qontinui_event)

    def _handle_qontinui_event(self, event: Event):
        """Handle diagnostic events from qontinui library.

        Converts library events to Tauri events and emits them.

        Args:
            event: Diagnostic event from qontinui
        """
        # Map qontinui event types to Tauri event types
        event_map = {
            "action_started": EventType.ACTION_STARTED,
            "action_completed": EventType.ACTION_COMPLETED,
            "action_failed": EventType.ERROR,
            "image_search_started": EventType.IMAGE_RECOGNITION,
            "image_search_match_found": EventType.IMAGE_RECOGNITION,
            "image_search_completed": EventType.IMAGE_RECOGNITION,
            "state_detected": EventType.STATE_DETECTED,
        }

        tauri_event_type = event_map.get(
            event.type.value,
            EventType.LOG
        )

        # Convert event to Tauri format
        tauri_data = {
            "event_id": str(event.id),
            "parent_id": str(event.parent_id) if event.parent_id else None,
            "correlation_id": str(event.correlation_id) if event.correlation_id else None,
            "level": event.level.value,
            "thread_id": event.thread_id,
            **event.data,  # Merge event data
        }

        # Emit to Tauri
        self._emit_event(tauri_event_type, tauri_data)

    def cleanup(self):
        """Clean up resources."""
        # Unregister event handler
        registry = EventRegistry()
        registry.unregister(self._handle_qontinui_event)
```

### 3.3 Consumer Side: Web (Future Mock Runs)

```python
# Future: In qontinui-web or mock execution system

from qontinui.events import EventRegistry, Event
import json

class WebEventCollector:
    """Collects events during mock runs for visualization."""

    def __init__(self):
        self.events: list[Event] = []
        self._registry = EventRegistry()

    def start_collection(self):
        """Start collecting events."""
        self._registry.register(self._collect_event)

    def stop_collection(self):
        """Stop collecting events."""
        self._registry.unregister(self._collect_event)

    def _collect_event(self, event: Event):
        """Store event for later analysis."""
        self.events.append(event)

    def get_event_timeline(self) -> list[dict]:
        """Get events as timeline for UI display."""
        return [event.to_dict() for event in self.events]

    def get_event_tree(self) -> dict:
        """Get events as hierarchical tree based on parent_id."""
        # Build parent-child relationships
        tree = {}
        events_by_id = {str(e.id): e for e in self.events}

        for event in self.events:
            event_dict = event.to_dict()
            parent_id = event_dict.get("parent_id")

            if parent_id is None:
                # Root event
                tree[str(event.id)] = event_dict
                tree[str(event.id)]["children"] = []
            else:
                # Child event - add to parent's children
                if parent_id in events_by_id:
                    parent_dict = events_by_id[parent_id]
                    if "children" not in parent_dict:
                        parent_dict["children"] = []
                    parent_dict["children"].append(event_dict)

        return tree

    def export_json(self, filepath: str):
        """Export events to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.get_event_timeline(), f, indent=2)
```

### 3.4 Consumer Side: Testing

```python
# In tests/events/test_event_system.py

from qontinui.events import EventRegistry, Event, EventType
import pytest

class EventCapture:
    """Test helper to capture events."""

    def __init__(self):
        self.events: list[Event] = []
        self._registry = EventRegistry()

    def __enter__(self):
        self._registry.register(self.events.append)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._registry.unregister(self.events.append)
        return False

    def get_events_by_type(self, event_type: EventType) -> list[Event]:
        """Filter events by type."""
        return [e for e in self.events if e.type == event_type]

    def assert_event_count(self, event_type: EventType, count: int):
        """Assert specific number of events of a type."""
        actual = len(self.get_events_by_type(event_type))
        assert actual == count, f"Expected {count} {event_type.value} events, got {actual}"


def test_action_execution_events():
    """Test that action execution emits proper events."""
    with EventCapture() as capture:
        # Execute some action
        runner = JSONRunner()
        runner.load_configuration("test_config.json")
        runner.run(process_id="test_process")

        # Verify events
        capture.assert_event_count(EventType.ACTION_STARTED, 1)
        capture.assert_event_count(EventType.ACTION_COMPLETED, 1)

        # Verify event hierarchy
        started_event = capture.get_events_by_type(EventType.ACTION_STARTED)[0]
        completed_event = capture.get_events_by_type(EventType.ACTION_COMPLETED)[0]

        assert completed_event.parent_id == started_event.id
        assert completed_event.correlation_id == started_event.id
```

---

## 4. Performance Considerations

### 4.1 Minimal Overhead When Disabled

```python
# In EventRegistry.emit()
def emit(self, event: "Event") -> None:
    # Fast path: single boolean check, returns immediately
    if not self._enabled:
        return

    # Only execute expensive operations if enabled
    with self._handler_lock:
        handlers = list(self._handlers)

    for handler in handlers:
        try:
            handler(event)
        except Exception:
            pass  # Silently ignore to not break library
```

**Performance characteristics:**
- **Disabled (default)**: Single boolean check (~1-2 nanoseconds)
- **Enabled with 0 handlers**: Boolean check + lock acquisition (~50-100 nanoseconds)
- **Enabled with handlers**: Above + handler execution time (consumer responsibility)

### 4.2 Thread Safety

1. **Registry lock**: Protects handler list modifications
2. **Handler snapshot**: Copy list before calling to avoid deadlocks
3. **Context variables**: Thread-local storage via `contextvars`
4. **Event immutability**: Events are dataclasses (immutable after creation)

### 4.3 Memory Considerations

- Events are NOT stored by the library (consumers responsible for storage)
- No queues maintained in library (stateless emission)
- Garbage collection handles event cleanup after handlers return
- Consumers can implement buffering/queuing as needed

### 4.4 Async Considerations (Future)

While the current design is synchronous, it can be extended for async:

```python
# Future: Async event emitter
class AsyncEventEmitter:
    async def emit_async(self, event: Event):
        """Emit event asynchronously."""
        if not self._enabled:
            return

        with self._handler_lock:
            handlers = list(self._async_handlers)

        # Run handlers concurrently
        await asyncio.gather(
            *[handler(event) for handler in handlers],
            return_exceptions=True  # Don't let handler errors break emission
        )
```

---

## 5. Migration Path

### Phase 1: Core Infrastructure (Immediate)
1. Implement event schema, registry, context, emitter
2. Add to `src/qontinui/events/` package
3. Write comprehensive unit tests
4. Document API

### Phase 2: Key Integration Points (Week 1)
1. Integrate into `ActionExecution`
2. Integrate into `OpenCVMatcher` (HAL)
3. Integrate into `Find` action
4. Add to state transition logic

### Phase 3: Consumer Updates (Week 2)
1. Update runner bridge to register handler
2. Map events to existing Tauri events
3. Test end-to-end event flow
4. Update documentation

### Phase 4: Additional Events (Ongoing)
1. Add more event types as needed
2. Expand data schemas
3. Add performance metrics events
4. Create event filtering/routing utilities

---

## 6. Alternative Approaches Considered

### 6.1 Logger-Based
**Pros:**
- Familiar API (logging module)
- Built-in handlers, formatters, filters
- Multi-level severity

**Cons:**
- Designed for text logs, not structured events
- No type safety for event data
- No hierarchical relationships
- Harder to filter programmatically
- Global logger state can interfere with user's logging

**Verdict:** Not ideal for structured diagnostic events

### 6.2 Observer Pattern
**Pros:**
- Classic design pattern
- Direct subscription model

**Cons:**
- Tight coupling (observers know about observables)
- No global registry
- Difficult to manage across modules
- No built-in context management

**Verdict:** Too coupled for library use case

### 6.3 Queue-Based
**Pros:**
- Natural async boundary
- Buffering built-in
- Consumer controls pull rate

**Cons:**
- Requires background thread for processing
- More complex setup
- Overhead even when disabled
- Memory growth if consumer slow

**Verdict:** Overkill for synchronous emission

### 6.4 Callback-Based (Simple)
**Pros:**
- Very simple
- Minimal overhead

**Cons:**
- No hierarchy support
- Single global callback only
- No filtering
- No event metadata

**Verdict:** Too simplistic for complex diagnostics

---

## 7. Conclusion

The **Context-based with Global Registry** approach provides the best balance of:
- **Simplicity**: Easy to use, minimal API surface
- **Performance**: Near-zero overhead when disabled
- **Flexibility**: Supports multiple consumers, hierarchical events
- **Thread Safety**: Built-in thread safety with contextvars
- **Decoupling**: Library doesn't depend on consumers
- **Type Safety**: Strongly typed event schemas

This design enables rich diagnostic capabilities for the runner, future web mock execution, and testing, while maintaining the library's independence and performance characteristics.

---

## Appendix A: Complete File Structure

```
src/qontinui/events/
├── __init__.py          # Public API
├── schema.py            # Event types and data classes
├── registry.py          # Global handler registry
├── context.py           # Context management for hierarchy
├── emitter.py           # Event emission facade
└── README.md            # Usage documentation

tests/events/
├── test_schema.py       # Event schema tests
├── test_registry.py     # Registry tests
├── test_context.py      # Context tests
├── test_emitter.py      # Emitter tests
└── test_integration.py  # End-to-end tests
```

## Appendix B: Example Event Timeline

```
Timeline View (flat):
├─ [ACTION_STARTED] Click on login button (id=123, corr=123)
│  ├─ [IMAGE_SEARCH_STARTED] Searching for 'login_button.png' (id=124, parent=123, corr=123)
│  │  └─ [TEMPLATE_MATCH_RESULT] Match found at (350, 200) conf=0.95 (id=125, parent=124, corr=123)
│  └─ [ACTION_COMPLETED] Success in 45ms, 1 match (id=126, parent=123, corr=123)
├─ [ACTION_STARTED] Type text "username" (id=127, corr=127)
   └─ [ACTION_COMPLETED] Success in 120ms (id=128, parent=127, corr=127)

Tree View (hierarchical):
{
  "123": {
    "type": "action_started",
    "data": {"action_type": "Click", "description": "Click on login button"},
    "children": [
      {
        "id": "124",
        "type": "image_search_started",
        "data": {"pattern_name": "login_button.png"},
        "children": [
          {
            "id": "125",
            "type": "template_match_result",
            "data": {"found": true, "confidence": 0.95, "location": [350, 200]}
          }
        ]
      },
      {
        "id": "126",
        "type": "action_completed",
        "data": {"success": true, "duration_ms": 45, "match_count": 1}
      }
    ]
  }
}
```
