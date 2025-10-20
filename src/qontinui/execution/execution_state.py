"""
Execution state tracking for graph-based workflow execution.

This module provides state management for workflow traversal, including:
- Tracking visited actions (cycle detection)
- Managing pending actions queue
- Recording execution history
- Supporting pause/resume functionality
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ExecutionStatus(str, Enum):
    """Status of workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ActionStatus(str, Enum):
    """Status of individual action execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ActionExecutionRecord:
    """Record of a single action execution."""

    action_id: str
    action_type: str
    status: ActionStatus
    start_time: datetime
    end_time: datetime | None = None
    duration_ms: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    output_type: str | None = None  # Which output was taken (for branching)
    output_index: int = 0

    def complete(
        self, result: dict[str, Any], output_type: str = "main", output_index: int = 0
    ) -> None:
        """
        Mark the action as completed successfully.

        Args:
            result: The execution result
            output_type: The output type taken
            output_index: The output index
        """
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = ActionStatus.COMPLETED
        self.result = result
        self.output_type = output_type
        self.output_index = output_index

    def fail(self, error: str) -> None:
        """
        Mark the action as failed.

        Args:
            error: Error message
        """
        self.end_time = datetime.now()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.status = ActionStatus.FAILED
        self.error = error

    def skip(self, reason: str) -> None:
        """
        Mark the action as skipped.

        Args:
            reason: Reason for skipping
        """
        self.end_time = datetime.now()
        self.duration_ms = 0
        self.status = ActionStatus.SKIPPED
        self.error = reason


@dataclass
class PendingAction:
    """
    Represents an action pending execution.

    Used for queue-based traversal.
    """

    action_id: str
    input_index: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    depth: int = 0  # Depth in execution tree (for debugging)


class ExecutionState:
    """
    Tracks the state of workflow execution during graph traversal.

    Provides:
    - Visited action tracking (cycle detection)
    - Pending action queue management
    - Execution history recording
    - Pause/resume support
    - Context management
    """

    def __init__(self, workflow_id: str, max_iterations: int = 10000, enable_history: bool = True):
        """
        Initialize execution state.

        Args:
            workflow_id: The workflow being executed
            max_iterations: Maximum iterations before stopping (infinite loop protection)
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
    # Status Management
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

    def is_paused(self) -> bool:
        """Check if execution is paused."""
        return self._paused

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

    def is_visited(self, action_id: str) -> bool:
        """
        Check if an action has been visited.

        Args:
            action_id: The action ID

        Returns:
            True if the action has been visited
        """
        return action_id in self._visited

    def set_current_action(self, action_id: str | None) -> None:
        """
        Set the current action being executed.

        Args:
            action_id: The action ID, or None if no action is current
        """
        self._current_action = action_id

    def get_current_action(self) -> str | None:
        """Get the current action being executed."""
        return self._current_action

    def increment_iteration(self) -> bool:
        """
        Increment iteration counter and check limit.

        Returns:
            True if under limit, False if limit exceeded
        """
        self._iteration_count += 1
        return self._iteration_count < self.max_iterations

    def get_iteration_count(self) -> int:
        """Get current iteration count."""
        return self._iteration_count

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
                action_id=action_id, input_index=input_index, context=context or {}, depth=depth
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

    def has_pending(self) -> bool:
        """Check if there are pending actions."""
        return len(self._pending) > 0

    def get_pending_count(self) -> int:
        """Get the number of pending actions."""
        return len(self._pending)

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
    # Statistics and Reporting
    # ============================================================================

    def get_statistics(self) -> dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary of statistics
        """
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds() * 1000

        return {
            "workflow_id": self.workflow_id,
            "status": self.status.value,
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
