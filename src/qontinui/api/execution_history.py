"""Execution History - Manages execution history tracking and retrieval.

This module provides the ExecutionHistory class that handles:
- Adding execution records to history
- Retrieving history with optional filtering
- Calculating execution statistics
- Managing history size limits
"""

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .execution_manager import ExecutionContext

logger = logging.getLogger(__name__)


class ExecutionHistory:
    """Manages execution history tracking and retrieval.

    This class provides thread-safe history management for workflow executions,
    including adding records, retrieving filtered history, and calculating
    statistics.

    Attributes:
        history: List of execution records, newest first
        max_history: Maximum number of records to retain
    """

    def __init__(self, max_history: int = 100) -> None:
        """Initialize execution history manager.

        Args:
            max_history: Maximum number of history records to retain (default: 100)
        """
        self.history: list[dict] = []
        self.max_history = max_history
        self._lock = asyncio.Lock()

        logger.info(f"ExecutionHistory initialized with max_history={max_history}")

    async def add_record(self, context: "ExecutionContext") -> None:
        """Add execution to history.

        Creates a history record from the execution context and adds it to the
        beginning of the history list. Automatically trims history to max_history
        size.

        Args:
            context: Execution context to record
        """
        async with self._lock:
            # Calculate duration
            duration = 0
            if context.end_time:
                duration = int(
                    (context.end_time - context.start_time).total_seconds() * 1000
                )

            # Create record
            record = {
                "execution_id": context.execution_id,
                "workflow_id": context.workflow.id,
                "workflow_name": context.workflow.name,
                "start_time": context.start_time.isoformat(),
                "end_time": context.end_time.isoformat() if context.end_time else None,
                "status": context.status.value,
                "duration": duration,
                "total_actions": context.total_actions,
                "completed_actions": context.completed_actions,
                "failed_actions": context.failed_actions,
                "error": context.error.get("message") if context.error else None,
            }

            # Add to history (newest first)
            self.history.insert(0, record)

            # Trim history
            if len(self.history) > self.max_history:
                self.history = self.history[: self.max_history]

            logger.debug(
                f"Added execution {context.execution_id} to history "
                f"(status={context.status.value}, duration={duration}ms)"
            )

    async def get_history(
        self, workflow_id: str | None = None, limit: int | None = None
    ) -> list[dict]:
        """Get execution history, optionally filtered by workflow.

        Returns execution history records, ordered by most recent first.
        Can be filtered by workflow ID and limited to a specific number
        of records.

        Args:
            workflow_id: Filter by workflow ID (optional)
            limit: Maximum number of records to return (optional)

        Returns:
            List of execution history records
        """
        async with self._lock:
            result = self.history.copy()

            # Filter by workflow ID
            if workflow_id:
                result = [r for r in result if r["workflow_id"] == workflow_id]

            # Apply limit
            if limit:
                result = result[:limit]

            return result

    async def get_statistics(self, workflow_id: str | None = None) -> dict:
        """Get execution statistics.

        Calculates aggregate statistics across all executions or for a specific
        workflow. Includes counts by status, success rate, average duration,
        and action statistics.

        Args:
            workflow_id: Filter by workflow ID (optional)

        Returns:
            Dictionary containing execution statistics:
            - total_executions: Total number of executions
            - completed: Number of completed executions
            - failed: Number of failed executions
            - cancelled: Number of cancelled executions
            - success_rate: Percentage of successful executions
            - avg_duration_ms: Average execution duration in milliseconds
            - total_actions: Total number of actions executed
            - total_completed_actions: Total number of completed actions
            - total_failed_actions: Total number of failed actions
        """
        async with self._lock:
            records = self.history.copy()

            # Filter by workflow ID
            if workflow_id:
                records = [r for r in records if r["workflow_id"] == workflow_id]

            if not records:
                return {
                    "total_executions": 0,
                    "completed": 0,
                    "failed": 0,
                    "cancelled": 0,
                    "success_rate": 0.0,
                    "avg_duration_ms": 0,
                    "total_actions": 0,
                    "total_completed_actions": 0,
                    "total_failed_actions": 0,
                }

            # Count by status
            completed = sum(1 for r in records if r["status"] == "completed")
            failed = sum(1 for r in records if r["status"] == "failed")
            cancelled = sum(1 for r in records if r["status"] == "cancelled")

            # Calculate success rate
            total = len(records)
            success_rate = (completed / total * 100) if total > 0 else 0.0

            # Calculate average duration
            durations = [r["duration"] for r in records if r["duration"] > 0]
            avg_duration = sum(durations) / len(durations) if durations else 0

            # Calculate action statistics
            total_actions = sum(r["total_actions"] for r in records)
            total_completed_actions = sum(r["completed_actions"] for r in records)
            total_failed_actions = sum(r["failed_actions"] for r in records)

            return {
                "total_executions": total,
                "completed": completed,
                "failed": failed,
                "cancelled": cancelled,
                "success_rate": round(success_rate, 2),
                "avg_duration_ms": int(avg_duration),
                "total_actions": total_actions,
                "total_completed_actions": total_completed_actions,
                "total_failed_actions": total_failed_actions,
            }

    async def clear_history(self) -> None:
        """Clear all execution history.

        Removes all records from the history. This operation cannot be undone.
        """
        async with self._lock:
            count = len(self.history)
            self.history.clear()
            logger.info(f"Cleared {count} records from execution history")
