"""
Execution tracking for workflow execution state.

This module provides read-only access to execution state for monitoring,
statistics, and history queries. It implements the Query side of Command-Query
Separation (CQS) pattern.
"""

from datetime import datetime
from typing import Any

from .execution_types import ActionExecutionRecord, ActionStatus, ExecutionStatus


class ExecutionTracker:
    """
    Provides read-only access to execution state for monitoring and statistics.

    This class implements the Query side of Command-Query Separation. It provides
    methods to query execution state, history, statistics, and progress without
    allowing any state modifications.

    The ExecutionTracker operates on internal state managed by ExecutionController.
    """

    def __init__(
        self,
        workflow_id: str,
        status: ExecutionStatus,
        start_time: datetime | None,
        end_time: datetime | None,
        visited: set[str],
        pending: list,
        current_action: str | None,
        iteration_count: int,
        history: list[ActionExecutionRecord],
        action_records: dict[str, ActionExecutionRecord],
        context: dict[str, Any],
        errors: list[dict[str, Any]],
        paused: bool,
        max_iterations: int,
    ) -> None:
        """
        Initialize execution tracker with references to internal state.

        Args:
            workflow_id: The workflow being executed
            status: Current execution status
            start_time: Execution start time
            end_time: Execution end time
            visited: Set of visited action IDs
            pending: List of pending actions
            current_action: Current action ID
            iteration_count: Current iteration count
            history: Execution history
            action_records: Action execution records
            context: Execution context
            errors: List of errors
            paused: Whether execution is paused
            max_iterations: Maximum iterations allowed
        """
        self._workflow_id = workflow_id
        self._status = status
        self._start_time = start_time
        self._end_time = end_time
        self._visited = visited
        self._pending = pending
        self._current_action = current_action
        self._iteration_count = iteration_count
        self._history = history
        self._action_records = action_records
        self._context = context
        self._errors = errors
        self._paused = paused
        self._max_iterations = max_iterations

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
        end = self._end_time or datetime.now()
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
