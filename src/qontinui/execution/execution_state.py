"""
Execution state management for graph-based workflow execution.

This module provides a unified interface for execution state tracking and control,
using composition to separate concerns:
- ExecutionController: State modifications (commands)
- ExecutionTracker: State queries (queries)

The ExecutionState class provides a facade over both components.
"""

from datetime import datetime
from typing import Any

from .execution_controller import ExecutionController
from .execution_tracker import ExecutionTracker
from .execution_types import ActionExecutionRecord, ExecutionStatus, PendingAction


class ExecutionState:
    """
    Unified interface for workflow execution state management.

    This class uses composition to separate concerns:
    - Controller: Handles all state modifications (commands)
    - Tracker: Provides read-only access to state (queries)

    This design follows Command-Query Separation (CQS) pattern while
    maintaining a simple interface for callers.
    """

    def __init__(
        self,
        workflow_id: str,
        max_iterations: int = 10000,
        enable_history: bool = True,
    ) -> None:
        """
        Initialize execution state.

        Args:
            workflow_id: The workflow being executed
            max_iterations: Maximum iterations before stopping
            enable_history: Whether to record execution history
        """
        # Initialize controller
        self._controller = ExecutionController(
            workflow_id=workflow_id,
            max_iterations=max_iterations,
            enable_history=enable_history,
        )

        # Initialize tracker with references to controller's internal state
        state = self._controller._get_internal_state()
        self._tracker = ExecutionTracker(
            workflow_id=state["workflow_id"],
            status=state["status"],
            start_time=state["start_time"],
            end_time=state["end_time"],
            visited=state["visited"],
            pending=state["pending"],
            current_action=state["current_action"],
            iteration_count=state["iteration_count"],
            history=state["history"],
            action_records=state["action_records"],
            context=state["context"],
            errors=state["errors"],
            paused=state["paused"],
            max_iterations=state["max_iterations"],
        )

        # Store for direct access
        self.workflow_id = workflow_id
        self.max_iterations = max_iterations
        self.enable_history = enable_history

    # ============================================================================
    # Property Access
    # ============================================================================

    @property
    def status(self) -> ExecutionStatus:
        """Get current execution status."""
        return self._controller.status

    @property
    def start_time(self) -> datetime | None:
        """Get execution start time."""
        return self._controller.start_time

    @property
    def end_time(self) -> datetime | None:
        """Get execution end time."""
        return self._controller.end_time

    # ============================================================================
    # Controller Methods (State Modifications)
    # ============================================================================

    def start(self) -> None:
        """Mark execution as started."""
        self._controller.start()

    def complete(self) -> None:
        """Mark execution as completed."""
        self._controller.complete()

    def fail(self, error: str) -> None:
        """Mark execution as failed."""
        self._controller.fail(error)

    def cancel(self) -> None:
        """Mark execution as cancelled."""
        self._controller.cancel()

    def pause(self) -> None:
        """Pause execution."""
        self._controller.pause()

    def resume(self) -> None:
        """Resume execution."""
        self._controller.resume()

    def mark_visited(self, action_id: str) -> None:
        """Mark an action as visited."""
        self._controller.mark_visited(action_id)

    def set_current_action(self, action_id: str | None) -> None:
        """Set the current action being executed."""
        self._controller.set_current_action(action_id)

    def increment_iteration(self) -> bool:
        """Increment iteration counter and check limit."""
        return self._controller.increment_iteration()

    def add_pending(
        self,
        action_id: str,
        input_index: int = 0,
        context: dict[str, Any] | None = None,
        depth: int = 0,
    ) -> None:
        """Add an action to the pending queue."""
        self._controller.add_pending(action_id, input_index, context, depth)

    def get_next_pending(self) -> PendingAction | None:
        """Get the next pending action from the queue."""
        return self._controller.get_next_pending()

    def clear_pending(self) -> None:
        """Clear all pending actions."""
        self._controller.clear_pending()

    def start_action(self, action_id: str, action_type: str) -> ActionExecutionRecord:
        """Start recording an action execution."""
        return self._controller.start_action(action_id, action_type)

    def set_context(self, key: str, value: Any) -> None:
        """Set a context value."""
        self._controller.set_context(key, value)

    def update_context(self, updates: dict[str, Any]) -> None:
        """Update multiple context values."""
        self._controller.update_context(updates)

    def set_pause_at_action(self, action_id: str | None) -> None:
        """Set an action to pause at (breakpoint)."""
        self._controller.set_pause_at_action(action_id)

    def reset(self) -> None:
        """Reset execution state for re-run."""
        self._controller.reset()

    # ============================================================================
    # Tracker Methods (State Queries)
    # ============================================================================

    def is_paused(self) -> bool:
        """Check if execution is paused."""
        return self._tracker.is_paused()

    def is_visited(self, action_id: str) -> bool:
        """Check if an action has been visited."""
        return self._tracker.is_visited(action_id)

    def get_current_action(self) -> str | None:
        """Get the current action being executed."""
        return self._tracker.get_current_action()

    def get_iteration_count(self) -> int:
        """Get current iteration count."""
        return self._tracker.get_iteration_count()

    def has_pending(self) -> bool:
        """Check if there are pending actions."""
        return self._tracker.has_pending()

    def get_pending_count(self) -> int:
        """Get the number of pending actions."""
        return self._tracker.get_pending_count()

    def get_history(self) -> list[ActionExecutionRecord]:
        """Get the execution history."""
        return self._tracker.get_history()

    def get_action_record(self, action_id: str) -> ActionExecutionRecord | None:
        """Get the execution record for an action."""
        return self._tracker.get_action_record(action_id)

    def get_failed_actions(self) -> list[ActionExecutionRecord]:
        """Get all failed actions."""
        return self._tracker.get_failed_actions()

    def get_completed_actions(self) -> list[ActionExecutionRecord]:
        """Get all completed actions."""
        return self._tracker.get_completed_actions()

    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self._tracker.get_context(key, default)

    def get_all_context(self) -> dict[str, Any]:
        """Get all context data."""
        return self._tracker.get_all_context()

    def should_pause_at(self, action_id: str) -> bool:
        """Check if execution should pause at this action."""
        return self._controller.should_pause_at(action_id)

    def get_statistics(self) -> dict[str, Any]:
        """Get execution statistics."""
        return self._tracker.get_statistics()

    def get_errors(self) -> list[dict[str, Any]]:
        """Get all errors that occurred during execution."""
        return self._tracker.get_errors()
