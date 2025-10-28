"""
Type definitions for execution state management.

This module contains shared types used across execution state components.
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
