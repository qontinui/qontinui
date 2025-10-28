"""Execution Event Bus - Manages event streaming and subscriber notifications.

This module provides the ExecutionEventBus class that handles:
- Event subscriber management
- Event emission to subscribers
- Event storage and retrieval
- Thread-safe operations with asyncio
"""

import asyncio
import logging
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Event Bus Context
# ============================================================================


@dataclass
class EventBusContext:
    """Context for managing events for a single execution."""

    execution_id: str
    event_queue: deque = field(default_factory=deque)
    event_subscribers: set[Callable[[Any], None]] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ============================================================================
# Execution Event Bus
# ============================================================================


class ExecutionEventBus:
    """Manages event streaming and subscriber notifications.

    The ExecutionEventBus provides:
    - Thread-safe event emission
    - Subscriber management
    - Event storage and retrieval
    - Automatic cleanup of old events
    """

    def __init__(self, max_events_per_execution: int = 1000) -> None:
        """Initialize event bus.

        Args:
            max_events_per_execution: Maximum events to store per execution
        """
        self.contexts: dict[str, EventBusContext] = {}
        self.max_events = max_events_per_execution
        self._lock = asyncio.Lock()

        logger.info(f"ExecutionEventBus initialized (max_events={max_events_per_execution})")

    async def subscribe(
        self,
        execution_id: str,
        callback: Callable[[Any], None],
    ) -> None:
        """Subscribe to events for an execution.

        Args:
            execution_id: Execution ID to subscribe to
            callback: Callback function to receive events

        Raises:
            ValueError: If execution_id is not registered
        """
        async with self._lock:
            context = self.contexts.get(execution_id)
            if not context:
                raise ValueError(f"Execution {execution_id} not found in event bus")

            async with context.lock:
                context.event_subscribers.add(callback)

            logger.debug(f"Added subscriber to execution {execution_id}")

    async def unsubscribe(
        self,
        execution_id: str,
        callback: Callable[[Any], None],
    ) -> None:
        """Unsubscribe from events for an execution.

        Args:
            execution_id: Execution ID to unsubscribe from
            callback: Callback function to remove
        """
        async with self._lock:
            context = self.contexts.get(execution_id)
            if not context:
                return

            async with context.lock:
                context.event_subscribers.discard(callback)

            logger.debug(f"Removed subscriber from execution {execution_id}")

    async def emit(
        self,
        execution_id: str,
        event: Any,
    ) -> None:
        """Emit event to all subscribers for an execution.

        Args:
            execution_id: Execution ID to emit event for
            event: Event to emit

        Raises:
            ValueError: If execution_id is not registered
        """
        async with self._lock:
            context = self.contexts.get(execution_id)
            if not context:
                raise ValueError(f"Execution {execution_id} not found in event bus")

        async with context.lock:
            # Add to queue
            context.event_queue.append(event)

            # Limit queue size
            while len(context.event_queue) > self.max_events:
                context.event_queue.popleft()

            # Notify subscribers
            for subscriber in list(context.event_subscribers):
                try:
                    subscriber(event)
                except Exception as e:
                    logger.error(
                        f"Error in event subscriber for execution {execution_id}: {e}",
                        exc_info=True,
                    )

    def get_events(
        self,
        execution_id: str,
        limit: int | None = None,
    ) -> list[Any]:
        """Get recent events for an execution.

        Args:
            execution_id: Execution ID to get events for
            limit: Maximum number of events to return (None = all)

        Returns:
            List of events (most recent first)

        Raises:
            ValueError: If execution_id is not registered
        """
        context = self.contexts.get(execution_id)
        if not context:
            raise ValueError(f"Execution {execution_id} not found in event bus")

        # Convert deque to list (most recent last)
        events = list(context.event_queue)

        # Apply limit if specified
        if limit is not None:
            events = events[-limit:]

        return events

    def clear_events(self, execution_id: str) -> None:
        """Clear all events for an execution.

        Args:
            execution_id: Execution ID to clear events for

        Raises:
            ValueError: If execution_id is not registered
        """
        context = self.contexts.get(execution_id)
        if not context:
            raise ValueError(f"Execution {execution_id} not found in event bus")

        context.event_queue.clear()
        logger.debug(f"Cleared events for execution {execution_id}")

    async def register_execution(self, execution_id: str) -> None:
        """Register a new execution with the event bus.

        Args:
            execution_id: Execution ID to register
        """
        async with self._lock:
            if execution_id not in self.contexts:
                self.contexts[execution_id] = EventBusContext(execution_id=execution_id)
                logger.debug(f"Registered execution {execution_id} with event bus")

    async def unregister_execution(self, execution_id: str) -> None:
        """Unregister an execution from the event bus.

        Args:
            execution_id: Execution ID to unregister
        """
        async with self._lock:
            if execution_id in self.contexts:
                del self.contexts[execution_id]
                logger.debug(f"Unregistered execution {execution_id} from event bus")

    def get_subscriber_count(self, execution_id: str) -> int:
        """Get number of subscribers for an execution.

        Args:
            execution_id: Execution ID

        Returns:
            Number of active subscribers

        Raises:
            ValueError: If execution_id is not registered
        """
        context = self.contexts.get(execution_id)
        if not context:
            raise ValueError(f"Execution {execution_id} not found in event bus")

        return len(context.event_subscribers)
