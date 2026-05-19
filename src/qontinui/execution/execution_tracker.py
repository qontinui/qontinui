"""
Execution tracking for workflow execution state.

This module provides read-only access to execution state for monitoring,
statistics, and history queries. It implements the Query side of Command-Query
Separation (CQS) pattern.
"""

from datetime import datetime
from typing import TYPE_CHECKING, Any

from qontinui_schemas.common import utc_now

from .execution_types import ActionExecutionRecord, ActionStatus, ExecutionStatus

if TYPE_CHECKING:
    from .execution_controller import ExecutionController


class ExecutionTracker:
    """
    Provides read-only access to execution state for monitoring and statistics.

    This class implements the Query side of Command-Query Separation. It provides
    methods to query execution state, history, statistics, and progress without
    allowing any state modifications.

    The ExecutionTracker reads *live* state from the ExecutionController. It holds
    a reference to the controller and resolves every query against the controller's
    current internal state, so status/lifecycle transitions made via the controller
    are always reflected here. Mutating containers (visited set, pending queue,
    history) are shared by reference; scalars (status, timestamps, pause flag) are
    resolved live on each access rather than snapshotted at construction.
    """

    def __init__(self, controller: "ExecutionController") -> None:
        """
        Initialize execution tracker with a reference to the controller.

        Args:
            controller: The ExecutionController whose state this tracker queries
        """
        self._controller = controller

    # ------------------------------------------------------------------
    # Live state accessors (read-through to the controller)
    # ------------------------------------------------------------------

    @property
    def _workflow_id(self) -> str:
        return self._controller.workflow_id

    @property
    def _status(self) -> ExecutionStatus:
        return self._controller.status

    @property
    def _start_time(self) -> datetime | None:
        return self._controller.start_time

    @property
    def _end_time(self) -> datetime | None:
        return self._controller.end_time

    @property
    def _visited(self) -> set[str]:
        return self._controller._visited

    @property
    def _pending(self) -> list:
        return self._controller._pending

    @property
    def _current_action(self) -> str | None:
        return self._controller._current_action

    @property
    def _iteration_count(self) -> int:
        return self._controller._iteration_count

    @property
    def _history(self) -> list[ActionExecutionRecord]:
        return self._controller._history

    @property
    def _action_records(self) -> dict[str, ActionExecutionRecord]:
        return self._controller._action_records

    @property
    def _context(self) -> dict[str, Any]:
        return self._controller._context

    @property
    def _errors(self) -> list[dict[str, Any]]:
        return self._controller._errors

    @property
    def _paused(self) -> bool:
        return self._controller._paused

    @property
    def _max_iterations(self) -> int:
        return self._controller.max_iterations

    # ============================================================================
    # Status Queries
    # ============================================================================

    def get_status(self) -> ExecutionStatus:
        """Get current execution status."""
        return self._status

    def is_paused(self) -> bool:
        """Check if execution is paused."""
        return self._paused

    def is_running(self) -> bool:
        """Check if execution is running."""
        return self._status == ExecutionStatus.RUNNING

    def is_completed(self) -> bool:
        """Check if execution is completed."""
        return self._status == ExecutionStatus.COMPLETED

    def is_failed(self) -> bool:
        """Check if execution has failed."""
        return self._status == ExecutionStatus.FAILED

    # ============================================================================
    # Action Tracking Queries
    # ============================================================================

    def is_visited(self, action_id: str) -> bool:
        """
        Check if an action has been visited.

        Args:
            action_id: The action ID

        Returns:
            True if the action has been visited
        """
        return action_id in self._visited

    def get_current_action(self) -> str | None:
        """Get the current action being executed."""
        return self._current_action

    def get_iteration_count(self) -> int:
        """Get current iteration count."""
        return self._iteration_count

    def get_visited_actions(self) -> set[str]:
        """Get all visited action IDs."""
        return self._visited.copy()

    # ============================================================================
    # Queue Queries
    # ============================================================================

    def has_pending(self) -> bool:
        """Check if there are pending actions."""
        return len(self._pending) > 0

    def get_pending_count(self) -> int:
        """Get the number of pending actions."""
        return len(self._pending)

    # ============================================================================
    # History Queries
    # ============================================================================

    def get_history(self) -> list[ActionExecutionRecord]:
        """Get the execution history."""
        return self._history.copy()

    def get_action_record(self, action_id: str) -> ActionExecutionRecord | None:
        """
        Get the execution record for an action.

        Args:
            action_id: The action ID

        Returns:
            The execution record, or None if not found
        """
        return self._action_records.get(action_id)

    def get_failed_actions(self) -> list[ActionExecutionRecord]:
        """Get all failed actions."""
        return [r for r in self._history if r.status == ActionStatus.FAILED]

    def get_completed_actions(self) -> list[ActionExecutionRecord]:
        """Get all completed actions."""
        return [r for r in self._history if r.status == ActionStatus.COMPLETED]

    # ============================================================================
    # Context Queries
    # ============================================================================

    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Get a context value.

        Args:
            key: The context key
            default: Default value if key not found

        Returns:
            The context value
        """
        return self._context.get(key, default)

    def get_all_context(self) -> dict[str, Any]:
        """Get all context data."""
        return self._context.copy()

    # ============================================================================
    # Statistics and Reporting
    # ============================================================================

    def get_statistics(self) -> dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary of statistics
        """
        duration = None
        if self._start_time and self._end_time:
            duration = (self._end_time - self._start_time).total_seconds() * 1000

        return {
            "workflow_id": self._workflow_id,
            "status": self._status.value,
            "duration_ms": duration,
            "iterations": self._iteration_count,
            "visited_count": len(self._visited),
            "pending_count": len(self._pending),
            "completed_count": len(self.get_completed_actions()),
            "failed_count": len(self.get_failed_actions()),
            "error_count": len(self._errors),
        }

    def get_errors(self) -> list[dict[str, Any]]:
        """Get all errors that occurred during execution."""
        return self._errors.copy()

    def get_duration_ms(self) -> float | None:
        """
        Get execution duration in milliseconds.

        Returns:
            Duration in milliseconds, or None if not started/completed
        """
        if not self._start_time:
            return None
        end = self._end_time or utc_now()
        return (end - self._start_time).total_seconds() * 1000

    def get_progress_percentage(self, total_actions: int) -> float:
        """
        Calculate execution progress as a percentage.

        Args:
            total_actions: Total number of actions in workflow

        Returns:
            Progress percentage (0-100)
        """
        if total_actions == 0:
            return 100.0
        completed = len(self.get_completed_actions())
        return (completed / total_actions) * 100.0
