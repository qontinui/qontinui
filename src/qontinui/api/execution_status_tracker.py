"""Execution Status Tracker - Tracks and reports execution status.

This module provides the ExecutionStatusTracker class that handles:
- Getting execution status with progress calculation
- Listing all active executions
- Retrieving execution history
- Computing execution statistics
"""

import logging
from typing import TYPE_CHECKING, Any

from .execution_history import ExecutionHistory
from .execution_registry import ExecutionRegistry

if TYPE_CHECKING:
    from .execution_manager import ExecutionContext

logger = logging.getLogger(__name__)


class ExecutionStatusTracker:
    """Tracks and reports execution status.

    The ExecutionStatusTracker handles:
    - Status retrieval with progress calculation
    - Active execution listing
    - History retrieval with filtering
    - Execution statistics computation

    This class focuses solely on status tracking and reporting.
    """

    def __init__(
        self,
        registry: ExecutionRegistry,
        history: ExecutionHistory,
    ) -> None:
        """Initialize execution status tracker.

        Args:
            registry: Execution registry for active executions
            history: History tracker for execution records
        """
        self.registry = registry
        self.history = history

        logger.info("ExecutionStatusTracker initialized")

    def get_status(self, execution_id: str) -> dict[str, Any]:
        """Get execution status.

        Args:
            execution_id: Execution ID

        Returns:
            Status dictionary with:
            - execution_id: Execution identifier
            - workflow_id: Workflow identifier
            - status: Current execution status
            - start_time: Execution start timestamp
            - end_time: Execution end timestamp (if completed)
            - current_action: Currently executing action ID
            - progress: Execution progress percentage
            - total_actions: Total number of actions
            - completed_actions: Number of completed actions
            - failed_actions: Number of failed actions
            - skipped_actions: Number of skipped actions
            - action_states: Dictionary mapping action IDs to states
            - error: Error information (if failed)
            - variables: Current execution variables

        Raises:
            ValueError: If execution not found
        """
        context = self.registry.get(execution_id)
        if not context:
            raise ValueError(f"Execution {execution_id} not found")

        # Calculate progress
        progress = self._calculate_progress(context)

        status = {
            "execution_id": execution_id,
            "workflow_id": context.workflow.id,
            "status": context.status.value,
            "start_time": context.start_time.isoformat(),
            "end_time": context.end_time.isoformat() if context.end_time else None,
            "current_action": context.current_action,
            "progress": progress,
            "total_actions": context.total_actions,
            "completed_actions": context.completed_actions,
            "failed_actions": context.failed_actions,
            "skipped_actions": context.skipped_actions,
            "action_states": context.action_states,
            "error": context.error,
            "variables": context.variables,
        }

        return status

    def get_all_executions(self) -> list[dict[str, Any]]:
        """Get all active executions.

        Returns:
            List of execution status dictionaries
        """
        return [self.get_status(exec_id) for exec_id in self.registry.get_all_ids()]

    async def get_execution_history(
        self, workflow_id: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Get execution history.

        Args:
            workflow_id: Filter by workflow ID (optional)
            limit: Maximum number of records to return

        Returns:
            List of execution records
        """
        return await self.history.get_history(workflow_id=workflow_id, limit=limit)

    async def get_execution_statistics(self, workflow_id: str | None = None) -> dict[str, Any]:
        """Get execution statistics.

        Args:
            workflow_id: Filter by workflow ID (optional)

        Returns:
            Dictionary with execution statistics
        """
        return await self.history.get_statistics(workflow_id=workflow_id)

    def _calculate_progress(self, context: "ExecutionContext") -> float:
        """Calculate execution progress percentage.

        Args:
            context: Execution context

        Returns:
            Progress as percentage (0.0 to 100.0)
        """
        if context.total_actions == 0:
            return 0.0

        completed = context.completed_actions + context.failed_actions + context.skipped_actions
        return (completed / context.total_actions) * 100.0
