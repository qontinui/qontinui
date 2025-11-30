"""Event emission for state execution operations.

This module handles all event emission for state transitions and navigation,
providing callbacks for integration with API/Runner components.

Architecture:
    - StateEventEmitter: Emits events for transitions and navigation
    - Supports optional callbacks for event-driven integration
    - Provides structured event data with timestamps

Key Features:
    1. Transition event emission (start, complete, failed)
    2. Navigation event emission (start, complete, partial, failed)
    3. Structured event data with context
    4. Optional callback integration
    5. Timestamp generation for events

Example:
    >>> emitter = StateEventEmitter()
    >>> emitter.emit_transition_start("login", callback)
    >>> emitter.emit_transition_complete(result, callback)
"""

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class StateEventEmitter:
    """Handles event emission for state execution operations.

    Emits structured events for transitions and navigation with optional
    callback integration for API/Runner components.

    Example:
        >>> emitter = StateEventEmitter()
        >>> def callback(event_type, data):
        ...     print(f"{event_type}: {data}")
        >>> emitter.emit_transition_start("login", callback)
    """

    @staticmethod
    def emit_transition_start(
        transition_id: str, emit_event_callback: Callable[[str, dict], None] | None
    ) -> None:
        """Emit transition start event.

        Args:
            transition_id: ID of transition starting
            emit_event_callback: Optional callback for emitting events
        """
        if emit_event_callback:
            emit_event_callback(
                "transition_start",
                {
                    "transition_id": transition_id,
                    "timestamp": StateEventEmitter._timestamp(),
                },
            )

    @staticmethod
    def emit_transition_complete(
        result: Any, emit_event_callback: Callable[[str, dict], None] | None
    ) -> None:
        """Emit transition completion event.

        Args:
            result: TransitionExecutionResult with execution details
            emit_event_callback: Optional callback for emitting events
        """
        if emit_event_callback:
            emit_event_callback(
                "transition_complete" if result.success else "transition_failed",
                {
                    "transition_id": result.transition_id,
                    "success": result.success,
                    "activated_states": list(result.activated_states),
                    "deactivated_states": list(result.deactivated_states),
                    "timestamp": StateEventEmitter._timestamp(),
                },
            )

    @staticmethod
    def emit_transition_failed(
        transition_id: str, error: str, emit_event_callback: Callable[[str, dict], None] | None
    ) -> None:
        """Emit transition failed event.

        Args:
            transition_id: ID of failed transition
            error: Error message
            emit_event_callback: Optional callback for emitting events
        """
        if emit_event_callback:
            emit_event_callback(
                "transition_failed",
                {
                    "transition_id": transition_id,
                    "error": error,
                    "timestamp": StateEventEmitter._timestamp(),
                },
            )

    @staticmethod
    def emit_navigation_start(
        target_state_ids: list[int],
        execute: bool,
        emit_event_callback: Callable[[str, dict], None] | None,
    ) -> None:
        """Emit navigation start event.

        Args:
            target_state_ids: IDs of target states
            execute: Whether executing or just computing path
            emit_event_callback: Optional callback for emitting events
        """
        if emit_event_callback:
            emit_event_callback(
                "navigation_start",
                {
                    "target_state_ids": target_state_ids,
                    "execute": execute,
                    "timestamp": StateEventEmitter._timestamp(),
                },
            )

    @staticmethod
    def emit_navigation_complete(
        result: Any,
        nav_context: Any,
        emit_event_callback: Callable[[str, dict], None] | None,
    ) -> None:
        """Emit navigation completion event.

        Args:
            result: NavigationResult with execution details
            nav_context: Navigation context from PathfindingNavigator
            emit_event_callback: Optional callback for emitting events
        """
        if emit_event_callback:
            emit_event_callback(
                "navigation_complete" if result.success else "navigation_partial",
                {
                    "target_state_ids": result.target_state_ids,
                    "success": result.success,
                    "targets_reached": list(nav_context.targets_reached),
                    "path_length": len(nav_context.path.transitions),
                    "timestamp": StateEventEmitter._timestamp(),
                },
            )

    @staticmethod
    def emit_navigation_failed(
        target_state_ids: list[int],
        error: str,
        emit_event_callback: Callable[[str, dict], None] | None,
    ) -> None:
        """Emit navigation failed event.

        Args:
            target_state_ids: IDs of target states
            error: Error message
            emit_event_callback: Optional callback for emitting events
        """
        if emit_event_callback:
            emit_event_callback(
                "navigation_failed",
                {
                    "target_state_ids": target_state_ids,
                    "error": error,
                    "timestamp": StateEventEmitter._timestamp(),
                },
            )

    @staticmethod
    def _timestamp() -> str:
        """Get current timestamp for events.

        Returns:
            ISO format timestamp string
        """
        return datetime.now().isoformat()
