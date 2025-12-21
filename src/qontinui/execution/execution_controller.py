"""
Execution control for workflow execution state.

This module provides state modification operations for execution control.
It implements the Command side of Command-Query Separation (CQS) pattern.
"""

from datetime import datetime
from typing import Any

from .execution_types import ActionExecutionRecord, ActionStatus, ExecutionStatus, PendingAction


class ExecutionController:
    """
    Manages execution state modifications and control operations.

    This class implements the Command side of Command-Query Separation. It provides
    methods to modify execution state, control lifecycle, manage queues, and record
    execution history.

    All state-modifying operations go through this controller to ensure consistency
    and proper state transitions.
    """

    def __init__(
        self,
        workflow_id: str,
        max_iterations: int = 10000,
        enable_history: bool = True,
    ) -> None:
        """
        Initialize execution controller.

        Args:
            workflow_id: The workflow being executed
            max_iterations: Maximum iterations before stopping
            enable_history: Whether to record execution history
        """
        self.workflow_id = workflow_id
        self.max_iterations = max_iterations
        self.enable_history = enable_history

        # Status tracking
        self.status = ExecutionStatus.PENDING
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None

        # Action tracking
        self._visited: set[str] = set()
        self._pending: list[PendingAction] = []
        self._current_action: str | None = None
        self._iteration_count: int = 0

        # Execution history
        self._history: list[ActionExecutionRecord] = []
        self._action_records: dict[str, ActionExecutionRecord] = {}

        # Context and variables
        self._context: dict[str, Any] = {}

        # Pause/resume support
        self._paused = False
        self._pause_at_action: str | None = None

        # Error tracking
        self._errors: list[dict[str, Any]] = []

    # ============================================================================
    # Internal State Access (for ExecutionTracker)
    # ============================================================================

    def _get_internal_state(self) -> dict[str, Any]:
        """
        Get internal state for ExecutionTracker.

        This method provides access to internal state for read-only operations.
        It should only be used by ExecutionTracker.

        Returns:
            Dictionary containing all internal state
        """
        return {
            "workflow_id": self.workflow_id,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "visited": self._visited,
            "pending": self._pending,
            "current_action": self._current_action,
            "iteration_count": self._iteration_count,
            "history": self._history,
            "action_records": self._action_records,
            "context": self._context,
            "errors": self._errors,
            "paused": self._paused,
            "max_iterations": self.max_iterations,
        }

    # ============================================================================
    # Lifecycle Control
    # ============================================================================

    def start(self) -> None:
        """Mark execution as started."""
        self.status = ExecutionStatus.RUNNING
        self.start_time = datetime.now()

    def complete(self) -> None:
        """Mark execution as completed."""
        self.status = ExecutionStatus.COMPLETED
        self.end_time = datetime.now()

    def fail(self, error: str) -> None:
        """
        Mark execution as failed.

        Args:
            error: Error message
        """
        self.status = ExecutionStatus.FAILED
        self.end_time = datetime.now()
        self._errors.append(
            {"time": datetime.now(), "error": error, "action": self._current_action}
        )

    def cancel(self) -> None:
        """Mark execution as cancelled."""
        self.status = ExecutionStatus.CANCELLED
        self.end_time = datetime.now()

    def pause(self) -> None:
        """Pause execution."""
        self.status = ExecutionStatus.PAUSED
        self._paused = True

    def resume(self) -> None:
        """Resume execution."""
        self.status = ExecutionStatus.RUNNING
        self._paused = False

    # ============================================================================
    # Action Tracking
    # ============================================================================

    def mark_visited(self, action_id: str) -> None:
        """
        Mark an action as visited.

        Args:
            action_id: The action ID
        """
        self._visited.add(action_id)

    def set_current_action(self, action_id: str | None) -> None:
        """
        Set the current action being executed.

        Args:
            action_id: The action ID, or None if no action is current
        """
        self._current_action = action_id

    def increment_iteration(self) -> bool:
        """
        Increment iteration counter and check limit.

        Returns:
            True if under limit, False if limit exceeded
        """
        self._iteration_count += 1
        return self._iteration_count < self.max_iterations

    # ============================================================================
    # Pending Actions Queue
    # ============================================================================

    def add_pending(
        self,
        action_id: str,
        input_index: int = 0,
        context: dict[str, Any] | None = None,
        depth: int = 0,
    ) -> None:
        """
        Add an action to the pending queue.

        Args:
            action_id: The action ID
            input_index: The input index
            context: Additional context for this action
            depth: Execution depth
        """
        self._pending.append(
            PendingAction(
                action_id=action_id,
                input_index=input_index,
                context=context or {},
                depth=depth,
            )
        )

    def get_next_pending(self) -> PendingAction | None:
        """
        Get the next pending action from the queue.

        Returns:
            The next pending action, or None if queue is empty
        """
        if not self._pending:
            return None
        return self._pending.pop(0)

    def clear_pending(self) -> None:
        """Clear all pending actions."""
        self._pending.clear()

    # ============================================================================
    # Execution History
    # ============================================================================

    def start_action(self, action_id: str, action_type: str) -> ActionExecutionRecord:
        """
        Start recording an action execution.

        Args:
            action_id: The action ID
            action_type: The action type

        Returns:
            The action execution record
        """
        record = ActionExecutionRecord(
            action_id=action_id,
            action_type=action_type,
            status=ActionStatus.RUNNING,
            start_time=datetime.now(),
        )

        if self.enable_history:
            self._history.append(record)
            self._action_records[action_id] = record

        return record

    # ============================================================================
    # Context Management
    # ============================================================================

    def set_context(self, key: str, value: Any) -> None:
        """
        Set a context value.

        Args:
            key: The context key
            value: The value
        """
        self._context[key] = value

    def update_context(self, updates: dict[str, Any]) -> None:
        """
        Update multiple context values.

        Args:
            updates: Dictionary of updates
        """
        self._context.update(updates)

    # ============================================================================
    # Pause/Resume Support
    # ============================================================================

    def set_pause_at_action(self, action_id: str | None) -> None:
        """
        Set an action to pause at (breakpoint).

        Args:
            action_id: The action ID to pause at, or None to clear
        """
        self._pause_at_action = action_id

    def should_pause_at(self, action_id: str) -> bool:
        """
        Check if execution should pause at this action.

        Args:
            action_id: The action ID

        Returns:
            True if execution should pause
        """
        return self._pause_at_action == action_id

    # ============================================================================
    # Reset
    # ============================================================================

    def reset(self) -> None:
        """Reset execution state for re-run."""
        self.status = ExecutionStatus.PENDING
        self.start_time = None
        self.end_time = None
        self._visited.clear()
        self._pending.clear()
        self._current_action = None
        self._iteration_count = 0
        self._history.clear()
        self._action_records.clear()
        self._errors.clear()
        self._paused = False
