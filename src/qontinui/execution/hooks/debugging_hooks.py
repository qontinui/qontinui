"""Debugging and tracking hooks for execution analysis.

This module provides hooks for debugging and variable tracking.
"""

import logging
import time
from typing import Any

from ...config import Action
from .base import ExecutionHook

logger = logging.getLogger(__name__)


class DebugHook(ExecutionHook):
    """Integrates with existing debug system for breakpoints and stepping.

    This hook connects to the qontinui debugging system to enable:
    - Breakpoints on action IDs or types
    - Step-through execution
    - Variable inspection
    - Execution recording

    Attributes:
        debug_manager: Reference to DebugManager instance
        paused: Whether execution is currently paused
    """

    def __init__(self, debug_manager: Any | None = None) -> None:
        """Initialize debug hook.

        Args:
            debug_manager: DebugManager instance (imports lazily if None)
        """
        self.debug_manager = debug_manager
        self.paused = False

        if debug_manager is None:
            try:
                from ...debugging.debug_manager import DebugManager

                self.debug_manager = DebugManager.get_instance()
            except ImportError:
                logger.warning("Debug system not available - DebugHook disabled")
                self.debug_manager = None

    def before_action(self, action: Action, context: dict[str, Any]):
        """Check for breakpoints before action execution."""
        if not self.debug_manager:
            return

        if self.debug_manager.should_break(action, context):
            logger.info(f"Breakpoint hit at action '{action.id}'")
            self.paused = True
            self._pause_and_wait_for_continue(action, context)

    def after_action(self, action: Action, context: dict[str, Any], result: dict[str, Any]):
        """Record action execution in debug system."""
        if not self.debug_manager:
            return

        try:
            self.debug_manager.record_action_execution(action, result, context)
        except Exception as e:
            logger.warning(f"Failed to record action execution: {e}")

    def on_error(self, action: Action, context: dict[str, Any], error: Exception):
        """Handle error breakpoints."""
        if not self.debug_manager:
            return

        if self.debug_manager.has_error_breakpoint():
            logger.info(f"Error breakpoint hit at action '{action.id}': {error}")
            self.paused = True
            self._pause_and_wait_for_continue(action, context, error=error)

    def _pause_and_wait_for_continue(
        self, action: Action, context: dict[str, Any], error: Exception | None = None
    ):
        """Pause execution and wait for continue signal.

        Args:
            action: Current action
            context: Current context
            error: Optional error if paused due to error
        """
        if not self.debug_manager:
            return

        while self.paused:
            time.sleep(0.1)

            if not self.debug_manager.is_paused():
                self.paused = False
                break


class VariableTrackingHook(ExecutionHook):
    """Tracks variable changes throughout execution.

    Attributes:
        variable_history: List of variable state snapshots
        tracked_variables: Set of variable names to track (None = track all)
    """

    def __init__(self, tracked_variables: list[str] | None = None) -> None:
        """Initialize variable tracking hook.

        Args:
            tracked_variables: List of variable names to track (None for all)
        """
        self.variable_history: list[dict[str, Any]] = []
        self.tracked_variables = set(tracked_variables) if tracked_variables else None

    def before_action(self, action: Action, context: dict[str, Any]):
        """Capture variable state before action."""
        self._capture_variables(action.id, "before", context)

    def after_action(self, action: Action, context: dict[str, Any], result: dict[str, Any]):
        """Capture variable state after action."""
        self._capture_variables(action.id, "after", context)

    def on_error(self, action: Action, context: dict[str, Any], error: Exception):
        """Capture variable state on error."""
        self._capture_variables(action.id, "error", context)

    def _capture_variables(self, action_id: str, stage: str, context: dict[str, Any]):
        """Capture current variable state.

        Args:
            action_id: ID of current action
            stage: Stage of execution (before/after/error)
            context: Current context
        """
        if self.tracked_variables:
            snapshot = {var: context.get(var) for var in self.tracked_variables if var in context}
        else:
            snapshot = context.copy()

        self.variable_history.append(
            {
                "action_id": action_id,
                "stage": stage,
                "timestamp": time.time(),
                "variables": snapshot,
            }
        )

    def get_variable_history(self, variable_name: str | None = None) -> list[dict[str, Any]]:
        """Get variable change history.

        Args:
            variable_name: Optional variable name to filter by

        Returns:
            List of variable history entries
        """
        if variable_name:
            return [entry for entry in self.variable_history if variable_name in entry["variables"]]
        return self.variable_history

    def get_variable_changes(self, variable_name: str) -> list[dict[str, Any]]:
        """Get all changes to a specific variable.

        Args:
            variable_name: Variable to track changes for

        Returns:
            List of changes with action_id and values
        """
        changes = []
        prev_value = None

        for entry in self.variable_history:
            if variable_name in entry["variables"]:
                current_value = entry["variables"][variable_name]

                if current_value != prev_value:
                    changes.append(
                        {
                            "action_id": entry["action_id"],
                            "stage": entry["stage"],
                            "prev_value": prev_value,
                            "new_value": current_value,
                        }
                    )
                    prev_value = current_value

        return changes
