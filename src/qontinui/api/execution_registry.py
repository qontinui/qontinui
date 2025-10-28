"""Execution Registry - Manages active workflow executions storage.

This module provides the ExecutionRegistry class that handles:
- Storage of active workflow executions
- Thread-safe execution lookup by ID
- Listing all active executions
- Adding and removing executions from the registry
"""

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .execution_manager import ExecutionContext

logger = logging.getLogger(__name__)


class ExecutionRegistry:
    """Registry for active workflow executions.

    This class provides thread-safe storage and lookup for active
    workflow executions. All operations are protected by a threading
    lock to ensure thread safety in concurrent environments.
    """

    def __init__(self) -> None:
        """Initialize execution registry."""
        self._executions: dict[str, "ExecutionContext"] = {}
        self._lock = threading.RLock()

        logger.debug("ExecutionRegistry initialized")

    def add(self, context: "ExecutionContext") -> None:
        """Add execution to registry.

        Args:
            context: Execution context to add
        """
        with self._lock:
            self._executions[context.execution_id] = context
            logger.debug(f"Added execution {context.execution_id} to registry")

    def get(self, execution_id: str) -> "ExecutionContext | None":
        """Get execution by ID.

        Args:
            execution_id: Execution ID to look up

        Returns:
            Execution context if found, None otherwise
        """
        with self._lock:
            return self._executions.get(execution_id)

    def remove(self, execution_id: str) -> "ExecutionContext | None":
        """Remove and return execution.

        Args:
            execution_id: Execution ID to remove

        Returns:
            Removed execution context if found, None otherwise
        """
        with self._lock:
            context = self._executions.pop(execution_id, None)
            if context:
                logger.debug(f"Removed execution {execution_id} from registry")
            return context

    def get_all(self) -> list["ExecutionContext"]:
        """Get all active executions.

        Returns:
            List of all execution contexts in the registry
        """
        with self._lock:
            return list(self._executions.values())

    def has(self, execution_id: str) -> bool:
        """Check if execution exists.

        Args:
            execution_id: Execution ID to check

        Returns:
            True if execution exists in registry, False otherwise
        """
        with self._lock:
            return execution_id in self._executions

    def clear(self) -> None:
        """Clear all executions.

        This removes all executions from the registry. Use with caution
        as it does not cancel or clean up any running tasks.
        """
        with self._lock:
            count = len(self._executions)
            self._executions.clear()
            logger.debug(f"Cleared {count} executions from registry")

    def get_all_ids(self) -> list[str]:
        """Get all execution IDs.

        Returns:
            List of all execution IDs in the registry
        """
        with self._lock:
            return list(self._executions.keys())
