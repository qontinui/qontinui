"""Execution hooks for monitoring and debugging workflow execution.

This module provides a hook system for intercepting and monitoring
workflow execution at key points.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from ..config import Action

logger = logging.getLogger(__name__)


class ExecutionHook(ABC):
    """Abstract base class for execution hooks.

    Hooks can intercept execution at three points:
    - before_action: Called before action execution
    - after_action: Called after successful action execution
    - on_error: Called when action execution fails
    """

    @abstractmethod
    def before_action(self, action: Action, context: dict[str, Any]):
        """Called before action execution.

        Args:
            action: The action about to execute
            context: Current execution context
        """
        pass

    @abstractmethod
    def after_action(self, action: Action, context: dict[str, Any], result: dict[str, Any]):
        """Called after successful action execution.

        Args:
            action: The action that executed
            context: Current execution context
            result: Execution result
        """
        pass

    @abstractmethod
    def on_error(self, action: Action, context: dict[str, Any], error: Exception):
        """Called when action execution fails.

        Args:
            action: The action that failed
            context: Current execution context
            error: The exception that occurred
        """
        pass


class LoggingHook(ExecutionHook):
    """Logs execution progress at various log levels.

    Attributes:
        logger_name: Name of logger to use
        log_level: Default log level
        log_context: Whether to log full context
    """

    def __init__(
        self,
        logger_name: str = "qontinui.execution",
        log_level: int = logging.INFO,
        log_context: bool = False,
    ):
        """Initialize logging hook.

        Args:
            logger_name: Name of logger to use
            log_level: Default log level (DEBUG, INFO, WARNING, ERROR)
            log_context: Whether to log full context (can be verbose)
        """
        self.logger = logging.getLogger(logger_name)
        self.log_level = log_level
        self.log_context = log_context

    def before_action(self, action: Action, context: dict[str, Any]):
        """Log before action execution."""
        msg = f"Executing action '{action.id}' (type={action.type})"
        if action.name:
            msg += f" - {action.name}"

        self.logger.log(self.log_level, msg)

        if self.log_context:
            self.logger.debug(f"Context: {context}")

    def after_action(self, action: Action, context: dict[str, Any], result: dict[str, Any]):
        """Log after successful execution."""
        success = result.get("success", False)
        msg = f"Action '{action.id}' completed: {'SUCCESS' if success else 'FAILED'}"

        self.logger.log(self.log_level, msg)

        if not success and "error" in result:
            self.logger.warning(f"Action '{action.id}' error: {result['error']}")

    def on_error(self, action: Action, context: dict[str, Any], error: Exception):
        """Log execution error."""
        self.logger.error(f"Action '{action.id}' raised exception: {type(error).__name__}: {error}")


class ProgressHook(ExecutionHook):
    """Reports execution progress as percentage.

    Attributes:
        total_actions: Total number of actions in workflow
        completed_actions: Number of actions completed
        progress_callback: Optional callback function for progress updates
    """

    def __init__(self, total_actions: int, progress_callback: callable | None = None):
        """Initialize progress hook.

        Args:
            total_actions: Total number of actions in workflow
            progress_callback: Optional callback(action_id, progress_percent)
        """
        self.total_actions = total_actions
        self.completed_actions = 0
        self.progress_callback = progress_callback

    def before_action(self, action: Action, context: dict[str, Any]):
        """Track action start."""
        pass  # Progress tracked on completion

    def after_action(self, action: Action, context: dict[str, Any], result: dict[str, Any]):
        """Update progress after action completion."""
        self.completed_actions += 1
        progress_percent = (self.completed_actions / self.total_actions) * 100

        logger.info(
            f"Progress: {progress_percent:.1f}% ({self.completed_actions}/{self.total_actions})"
        )

        if self.progress_callback:
            try:
                self.progress_callback(action.id, progress_percent)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def on_error(self, action: Action, context: dict[str, Any], error: Exception):
        """Count failed actions as completed for progress."""
        self.completed_actions += 1

    def get_progress(self) -> float:
        """Get current progress percentage.

        Returns:
            Progress as percentage (0-100)
        """
        if self.total_actions == 0:
            return 100.0
        return (self.completed_actions / self.total_actions) * 100


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

    def __init__(self, debug_manager: Any | None = None):
        """Initialize debug hook.

        Args:
            debug_manager: DebugManager instance (imports lazily if None)
        """
        self.debug_manager = debug_manager
        self.paused = False

        if debug_manager is None:
            try:
                from ..debugging.debug_manager import DebugManager

                self.debug_manager = DebugManager.get_instance()
            except ImportError:
                logger.warning("Debug system not available - DebugHook disabled")
                self.debug_manager = None

    def before_action(self, action: Action, context: dict[str, Any]):
        """Check for breakpoints before action execution."""
        if not self.debug_manager:
            return

        # Check if we should break
        if self.debug_manager.should_break(action, context):
            logger.info(f"Breakpoint hit at action '{action.id}'")
            self.paused = True

            # Pause execution (implementation depends on debug system)
            # This is a simplified version - actual implementation would
            # integrate with the full debug session system
            self._pause_and_wait_for_continue(action, context)

    def after_action(self, action: Action, context: dict[str, Any], result: dict[str, Any]):
        """Record action execution in debug system."""
        if not self.debug_manager:
            return

        # Record execution (if recording is enabled)
        try:
            self.debug_manager.record_action_execution(action, result, context)
        except Exception as e:
            logger.warning(f"Failed to record action execution: {e}")

    def on_error(self, action: Action, context: dict[str, Any], error: Exception):
        """Handle error breakpoints."""
        if not self.debug_manager:
            return

        # Check for error breakpoints
        if self.debug_manager.has_error_breakpoint():
            logger.info(f"Error breakpoint hit at action '{action.id}': {error}")
            self.paused = True
            self._pause_and_wait_for_continue(action, context, error=error)

    def _pause_and_wait_for_continue(
        self, action: Action, context: dict[str, Any], error: Exception | None = None
    ):
        """Pause execution and wait for continue signal.

        This is a simplified implementation. The actual implementation
        would integrate with the full DebugSession system.

        Args:
            action: Current action
            context: Current context
            error: Optional error if paused due to error
        """
        if not self.debug_manager:
            return

        # Wait for continue signal from debug system
        # This would typically involve checking a session state
        while self.paused:
            time.sleep(0.1)

            # Check if user requested continue
            if not self.debug_manager.is_paused():
                self.paused = False
                break


class TimingHook(ExecutionHook):
    """Tracks execution timing for performance analysis.

    Attributes:
        action_timings: Dictionary mapping action_id to execution time
        start_times: Dictionary tracking current action start times
    """

    def __init__(self):
        """Initialize timing hook."""
        self.action_timings: dict[str, float] = {}
        self.start_times: dict[str, float] = {}

    def before_action(self, action: Action, context: dict[str, Any]):
        """Record action start time."""
        self.start_times[action.id] = time.time()

    def after_action(self, action: Action, context: dict[str, Any], result: dict[str, Any]):
        """Record action completion time."""
        if action.id in self.start_times:
            elapsed = time.time() - self.start_times[action.id]
            self.action_timings[action.id] = elapsed
            logger.debug(f"Action '{action.id}' executed in {elapsed:.3f}s")
            del self.start_times[action.id]

    def on_error(self, action: Action, context: dict[str, Any], error: Exception):
        """Record timing even on error."""
        if action.id in self.start_times:
            elapsed = time.time() - self.start_times[action.id]
            self.action_timings[action.id] = elapsed
            del self.start_times[action.id]

    def get_timing_report(self) -> dict[str, Any]:
        """Get timing analysis report.

        Returns:
            Dictionary with timing statistics
        """
        if not self.action_timings:
            return {"error": "No timing data available"}

        total_time = sum(self.action_timings.values())
        avg_time = total_time / len(self.action_timings)

        sorted_timings = sorted(self.action_timings.items(), key=lambda x: x[1], reverse=True)

        return {
            "total_actions": len(self.action_timings),
            "total_time": total_time,
            "average_time": avg_time,
            "slowest_actions": sorted_timings[:10],
            "fastest_actions": sorted_timings[-10:],
            "all_timings": self.action_timings,
        }


class VariableTrackingHook(ExecutionHook):
    """Tracks variable changes throughout execution.

    Attributes:
        variable_history: List of variable state snapshots
        tracked_variables: Set of variable names to track (None = track all)
    """

    def __init__(self, tracked_variables: list[str] | None = None):
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
            # Track only specified variables
            snapshot = {var: context.get(var) for var in self.tracked_variables if var in context}
        else:
            # Track all variables
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


class CompositeHook(ExecutionHook):
    """Composite hook that delegates to multiple hooks.

    Allows combining multiple hooks into a single hook instance.

    Attributes:
        hooks: List of child hooks
    """

    def __init__(self, hooks: list[ExecutionHook] | None = None):
        """Initialize composite hook.

        Args:
            hooks: List of hooks to compose (optional)
        """
        self.hooks = hooks or []

    def add_hook(self, hook: ExecutionHook):
        """Add a hook to the composite.

        Args:
            hook: Hook to add
        """
        self.hooks.append(hook)

    def remove_hook(self, hook: ExecutionHook):
        """Remove a hook from the composite.

        Args:
            hook: Hook to remove
        """
        if hook in self.hooks:
            self.hooks.remove(hook)

    def before_action(self, action: Action, context: dict[str, Any]):
        """Call before_action on all child hooks."""
        for hook in self.hooks:
            try:
                hook.before_action(action, context)
            except Exception as e:
                logger.warning(f"Hook {type(hook).__name__}.before_action failed: {e}")

    def after_action(self, action: Action, context: dict[str, Any], result: dict[str, Any]):
        """Call after_action on all child hooks."""
        for hook in self.hooks:
            try:
                hook.after_action(action, context, result)
            except Exception as e:
                logger.warning(f"Hook {type(hook).__name__}.after_action failed: {e}")

    def on_error(self, action: Action, context: dict[str, Any], error: Exception):
        """Call on_error on all child hooks."""
        for hook in self.hooks:
            try:
                hook.on_error(action, context, error)
            except Exception as e:
                logger.warning(f"Hook {type(hook).__name__}.on_error failed: {e}")
