"""Execution context for workflow orchestration.

Manages variable storage, state tracking, and execution statistics during workflow execution.

Thread Safety:
    This module is thread-safe. All mutable shared state is protected by RLock.
    Multiple threads can safely access and modify the execution context concurrently.
"""

import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class ExecutionStatistics:
    """Statistics collected during workflow execution."""

    total_actions: int = 0
    """Total number of actions executed."""

    successful_actions: int = 0
    """Number of successful actions."""

    failed_actions: int = 0
    """Number of failed actions."""

    retried_actions: int = 0
    """Number of actions that required retries."""

    total_retries: int = 0
    """Total number of retry attempts."""

    start_time: datetime | None = None
    """When workflow execution started."""

    end_time: datetime | None = None
    """When workflow execution ended."""

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage.

        Returns:
            Success rate (0-100)
        """
        if self.total_actions == 0:
            return 0.0
        return (self.successful_actions / self.total_actions) * 100

    @property
    def duration_seconds(self) -> float:
        """Calculate execution duration in seconds.

        Returns:
            Duration in seconds, or 0 if not completed
        """
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def __str__(self) -> str:
        """String representation of statistics."""
        return (
            f"ExecutionStatistics("
            f"total={self.total_actions}, "
            f"successful={self.successful_actions}, "
            f"failed={self.failed_actions}, "
            f"retried={self.retried_actions}, "
            f"success_rate={self.success_rate:.1f}%, "
            f"duration={self.duration_seconds:.2f}s)"
        )


@dataclass
class ActionState:
    """State information for a single action execution."""

    action_index: int
    """Index of the action in the workflow."""

    action_name: str
    """Name or description of the action."""

    attempt_count: int = 0
    """Number of execution attempts for this action."""

    success: bool = False
    """Whether the action succeeded."""

    error: Exception | None = None
    """Error that occurred, if any."""

    start_time: float | None = None
    """When this action started executing."""

    end_time: float | None = None
    """When this action completed."""

    @property
    def duration(self) -> float:
        """Calculate action execution duration.

        Returns:
            Duration in seconds
        """
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


class ExecutionContext:
    """Context for workflow execution.

    Manages variables, state tracking, and execution statistics throughout
    the lifecycle of a workflow execution.

    Thread Safety:
        All methods are thread-safe. Concurrent access is protected by an RLock.
        This allows safe usage in multi-threaded workflow execution scenarios.
    """

    def __init__(self, initial_variables: dict[str, Any] | None = None) -> None:
        """Initialize execution context.

        Args:
            initial_variables: Optional initial variable values
        """
        self._lock = threading.RLock()
        self._variables: dict[str, Any] = initial_variables or {}
        self._action_states: list[ActionState] = []
        self._statistics = ExecutionStatistics()
        self._metadata: dict[str, Any] = {}

    @property
    def variables(self) -> dict[str, Any]:
        """Get current variable values.

        Returns:
            Dictionary of variable names to values

        Thread Safety:
            Returns a copy to prevent external modifications.
        """
        with self._lock:
            return self._variables.copy()

    @property
    def statistics(self) -> ExecutionStatistics:
        """Get execution statistics.

        Returns:
            ExecutionStatistics instance

        Thread Safety:
            Returns the statistics object. Note that ExecutionStatistics itself
            is a dataclass and modifications should be done through ExecutionContext methods.
        """
        with self._lock:
            return self._statistics

    @property
    def action_states(self) -> list[ActionState]:
        """Get all action states.

        Returns:
            List of ActionState instances

        Thread Safety:
            Returns a copy to prevent external modifications.
        """
        with self._lock:
            return self._action_states.copy()

    @property
    def metadata(self) -> dict[str, Any]:
        """Get execution metadata.

        Returns:
            Dictionary of metadata

        Thread Safety:
            Returns a copy to prevent external modifications.
        """
        with self._lock:
            return self._metadata.copy()

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable value.

        Args:
            name: Variable name
            value: Variable value

        Thread Safety:
            Protected by lock for concurrent access.
        """
        with self._lock:
            self._variables[name] = value

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a variable value.

        Args:
            name: Variable name
            default: Default value if variable not found

        Returns:
            Variable value or default

        Thread Safety:
            Protected by lock for concurrent access.
        """
        with self._lock:
            return self._variables.get(name, default)

    def has_variable(self, name: str) -> bool:
        """Check if a variable exists.

        Args:
            name: Variable name

        Returns:
            True if variable exists

        Thread Safety:
            Protected by lock for concurrent access.
        """
        with self._lock:
            return name in self._variables

    def delete_variable(self, name: str) -> None:
        """Delete a variable.

        Args:
            name: Variable name

        Thread Safety:
            Protected by lock for concurrent access.
        """
        with self._lock:
            self._variables.pop(name, None)

    def clear_variables(self) -> None:
        """Clear all variables.

        Thread Safety:
            Protected by lock for concurrent access.
        """
        with self._lock:
            self._variables.clear()

    def substitute_variables(self, text: str) -> str:
        """Substitute variable placeholders in text.

        Replaces ${variable_name} with variable values.

        Args:
            text: Text containing variable placeholders

        Returns:
            Text with variables substituted

        Thread Safety:
            Protected by lock to ensure consistent variable reads.
        """
        if not text:
            return text

        with self._lock:

            def replace_var(match: re.Match[str]) -> str:
                var_name = match.group(1)
                value = self._variables.get(var_name)
                return str(value) if value is not None else match.group(0)

            return re.sub(r"\$\{([^}]+)\}", replace_var, text)

    def start_action(self, index: int, name: str) -> ActionState:
        """Record the start of an action execution.

        Args:
            index: Action index in workflow
            name: Action name

        Returns:
            ActionState instance

        Thread Safety:
            Protected by lock for concurrent action tracking.
        """
        with self._lock:
            state = ActionState(action_index=index, action_name=name, start_time=time.time())
            self._action_states.append(state)
            self._statistics.total_actions += 1
            return state

    def complete_action(
        self, state: ActionState, success: bool, error: Exception | None = None
    ) -> None:
        """Record the completion of an action.

        Args:
            state: ActionState instance
            success: Whether action succeeded
            error: Error that occurred, if any

        Thread Safety:
            Protected by lock for concurrent statistics updates.
        """
        with self._lock:
            state.end_time = time.time()
            state.success = success
            state.error = error

            if success:
                self._statistics.successful_actions += 1
            else:
                self._statistics.failed_actions += 1

    def record_retry(self, state: ActionState) -> None:
        """Record a retry attempt for an action.

        Args:
            state: ActionState instance

        Thread Safety:
            Protected by lock for concurrent retry tracking.
        """
        with self._lock:
            state.attempt_count += 1
            self._statistics.total_retries += 1

            if state.attempt_count == 1:
                # First retry - increment retried actions count
                self._statistics.retried_actions += 1

    def start_workflow(self) -> None:
        """Mark the start of workflow execution.

        Thread Safety:
            Protected by lock for concurrent access.
        """
        with self._lock:
            self._statistics.start_time = datetime.now()

    def complete_workflow(self) -> None:
        """Mark the completion of workflow execution.

        Thread Safety:
            Protected by lock for concurrent access.
        """
        with self._lock:
            self._statistics.end_time = datetime.now()

    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value.

        Args:
            key: Metadata key
            value: Metadata value

        Thread Safety:
            Protected by lock for concurrent access.
        """
        with self._lock:
            self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value.

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value or default

        Thread Safety:
            Protected by lock for concurrent access.
        """
        with self._lock:
            return self._metadata.get(key, default)

    def get_last_action_state(self) -> ActionState | None:
        """Get the state of the last executed action.

        Returns:
            ActionState or None if no actions executed

        Thread Safety:
            Protected by lock for concurrent access.
        """
        with self._lock:
            if self._action_states:
                return self._action_states[-1]
            return None

    def get_failed_actions(self) -> list[ActionState]:
        """Get all failed action states.

        Returns:
            List of failed ActionState instances

        Thread Safety:
            Protected by lock for concurrent access.
        """
        with self._lock:
            return [state for state in self._action_states if not state.success]

    def __str__(self) -> str:
        """String representation of context.

        Thread Safety:
            Protected by lock for concurrent access.
        """
        with self._lock:
            return f"ExecutionContext(variables={len(self._variables)}, {self._statistics})"
