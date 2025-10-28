"""
Merge context management for tracking merge node state.

This module provides the MergeContext class which tracks:
- Which input paths have arrived at a merge node
- Execution contexts from each incoming path
- Readiness state for execution
- Variable conflict resolution
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class InputRecord:
    """Record of a single input arriving at a merge node."""

    from_action_id: str
    context: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    execution_path: list[str] = field(default_factory=list)


class VariableConflictResolution:
    """
    Strategies for resolving variable conflicts when merging contexts.

    When multiple paths define the same variable with different values,
    we need a strategy to resolve the conflict.
    """

    FIRST_WINS = "first_wins"  # First path to arrive wins
    LAST_WINS = "last_wins"  # Last path to arrive wins
    MERGE_LISTS = "merge_lists"  # Combine values into a list
    ERROR_ON_CONFLICT = "error_on_conflict"  # Raise error if conflict
    PRIORITY_ORDER = "priority_order"  # Use explicit priority order

    @staticmethod
    def resolve(
        variable_name: str,
        values: list[tuple[str, Any]],  # List of (action_id, value) tuples
        strategy: str = LAST_WINS,
        priority_order: list[str] | None = None,
    ) -> Any:
        """
        Resolve a variable conflict using the specified strategy.

        Args:
            variable_name: Name of the conflicting variable
            values: List of (action_id, value) tuples
            strategy: Resolution strategy to use
            priority_order: Priority order for PRIORITY_ORDER strategy

        Returns:
            Resolved value

        Raises:
            ValueError: If ERROR_ON_CONFLICT strategy and conflict exists
        """
        if len(values) == 1:
            return values[0][1]

        if strategy == VariableConflictResolution.FIRST_WINS:
            return values[0][1]

        elif strategy == VariableConflictResolution.LAST_WINS:
            return values[-1][1]

        elif strategy == VariableConflictResolution.MERGE_LISTS:
            # Combine all values into a list
            return [v[1] for v in values]

        elif strategy == VariableConflictResolution.ERROR_ON_CONFLICT:
            action_ids = [v[0] for v in values]
            raise ValueError(f"Variable conflict for '{variable_name}' from actions: {action_ids}")

        elif strategy == VariableConflictResolution.PRIORITY_ORDER:
            if not priority_order:
                raise ValueError("priority_order must be provided for PRIORITY_ORDER strategy")
            # Find the first action in priority order that has this variable
            for action_id in priority_order:
                for vid, value in values:
                    if vid == action_id:
                        return value
            # If none match priority order, use last wins
            return values[-1][1]

        else:
            raise ValueError(f"Unknown conflict resolution strategy: {strategy}")


class MergeContext:
    """
    Tracks the state of a merge node in a workflow.

    A merge node is an action with multiple incoming connections. This class
    manages the state of which paths have arrived, their contexts, and when
    the merge is ready to execute.

    Thread-safe for parallel execution scenarios.
    """

    def __init__(
        self,
        action_id: str,
        expected_inputs: set[str],
        conflict_resolution: str = VariableConflictResolution.LAST_WINS,
        priority_order: list[str] | None = None,
    ) -> None:
        """
        Initialize a merge context.

        Args:
            action_id: ID of the merge action
            expected_inputs: Set of action IDs that should provide input
            conflict_resolution: Strategy for resolving variable conflicts
            priority_order: Priority order for PRIORITY_ORDER resolution
        """
        self.action_id = action_id
        self.expected_inputs = expected_inputs.copy()
        self.conflict_resolution = conflict_resolution
        self.priority_order = priority_order or []

        # Track received inputs
        self._inputs: dict[str, InputRecord] = {}
        self._lock = threading.RLock()
        self._ready_event = threading.Event()

        # Execution state
        self._executed = False
        self._result: dict[str, Any] | None = None
        self._error: Exception | None = None

    def register_input(
        self,
        from_action_id: str,
        context: dict[str, Any],
        execution_path: list[str] | None = None,
    ) -> None:
        """
        Register that an input path has arrived.

        Args:
            from_action_id: ID of the action providing input
            context: Execution context from that action
            execution_path: Optional list of action IDs in execution path

        Raises:
            ValueError: If from_action_id is not an expected input
        """
        with self._lock:
            if from_action_id not in self.expected_inputs:
                raise ValueError(
                    f"Unexpected input from '{from_action_id}'. "
                    f"Expected inputs: {self.expected_inputs}"
                )

            if from_action_id in self._inputs:
                # Input already received - this could be a retry or duplicate
                # Update with new context
                pass

            self._inputs[from_action_id] = InputRecord(
                from_action_id=from_action_id,
                context=context.copy(),
                execution_path=execution_path or [],
            )

    def has_input_from(self, action_id: str) -> bool:
        """Check if input from a specific action has been received."""
        with self._lock:
            return action_id in self._inputs

    def get_received_inputs(self) -> set[str]:
        """Get set of action IDs that have provided input."""
        with self._lock:
            return set(self._inputs.keys())

    def get_pending_inputs(self) -> set[str]:
        """Get set of action IDs still pending."""
        with self._lock:
            return self.expected_inputs - set(self._inputs.keys())

    def is_complete(self) -> bool:
        """Check if all expected inputs have been received."""
        with self._lock:
            return len(self._inputs) == len(self.expected_inputs)

    def get_input_count(self) -> int:
        """Get count of received inputs."""
        with self._lock:
            return len(self._inputs)

    def get_expected_count(self) -> int:
        """Get count of expected inputs."""
        return len(self.expected_inputs)

    def is_ready(self, from_strategy=None) -> bool:
        """
        Check if the merge is ready to execute.

        This can be called by a MergeStrategy to determine readiness.
        Default behavior is to check if all inputs are complete.

        Args:
            from_strategy: Optional strategy object (for custom checks)

        Returns:
            True if ready to execute
        """
        if from_strategy is not None:
            # Let the strategy decide
            return from_strategy.should_execute(len(self._inputs), len(self.expected_inputs))

        # Default: all inputs must be received
        return self.is_complete()

    def get_merged_context(self) -> dict[str, Any]:
        """
        Merge contexts from all received inputs.

        Returns:
            Merged execution context with conflicts resolved

        Raises:
            ValueError: If conflict resolution fails
        """
        with self._lock:
            if not self._inputs:
                return {}

            # Start with empty merged context
            merged: dict[str, Any] = {}

            # Track which actions contribute to each variable
            variable_sources: dict[str, list[tuple[str, Any]]] = {}

            # Collect all variables from all contexts
            for action_id, record in self._inputs.items():
                context = record.context

                for key, value in context.items():
                    if key not in variable_sources:
                        variable_sources[key] = []
                    variable_sources[key].append((action_id, value))

            # Resolve conflicts for each variable
            for var_name, sources in variable_sources.items():
                merged[var_name] = VariableConflictResolution.resolve(
                    var_name, sources, self.conflict_resolution, self.priority_order
                )

            return merged

    def get_input_contexts(self) -> dict[str, dict[str, Any]]:
        """
        Get all input contexts without merging.

        Returns:
            Dict mapping action_id to its context
        """
        with self._lock:
            return {action_id: record.context.copy() for action_id, record in self._inputs.items()}

    def get_input_record(self, action_id: str) -> InputRecord | None:
        """Get the input record for a specific action."""
        with self._lock:
            return self._inputs.get(action_id)

    def get_all_input_records(self) -> list[InputRecord]:
        """Get all input records."""
        with self._lock:
            return list(self._inputs.values())

    def mark_executed(self, result: dict[str, Any] | None = None) -> None:
        """
        Mark the merge node as executed.

        Args:
            result: Optional execution result
        """
        with self._lock:
            self._executed = True
            self._result = result
            self._ready_event.set()

    def mark_error(self, error: Exception) -> None:
        """
        Mark the merge node as failed.

        Args:
            error: The exception that occurred
        """
        with self._lock:
            self._executed = True
            self._error = error
            self._ready_event.set()

    def has_executed(self) -> bool:
        """Check if the merge node has executed."""
        with self._lock:
            return self._executed

    def get_result(self) -> dict[str, Any] | None:
        """Get execution result (if executed successfully)."""
        with self._lock:
            return self._result

    def get_error(self) -> Exception | None:
        """Get execution error (if execution failed)."""
        with self._lock:
            return self._error

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """
        Wait until the merge is ready or executed.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            True if ready/executed, False if timeout
        """
        if self.is_complete() or self._executed:
            return True
        return self._ready_event.wait(timeout)

    def reset(self) -> None:
        """
        Reset the merge context to initial state.

        This allows the merge node to be re-executed with new inputs.
        """
        with self._lock:
            self._inputs.clear()
            self._executed = False
            self._result = None
            self._error = None
            self._ready_event.clear()

    def get_statistics(self) -> dict[str, Any]:
        """
        Get statistics about the merge state.

        Returns:
            Dict with merge statistics
        """
        with self._lock:
            received = set(self._inputs.keys())
            pending = self.expected_inputs - received

            return {
                "action_id": self.action_id,
                "expected_inputs": list(self.expected_inputs),
                "received_inputs": list(received),
                "pending_inputs": list(pending),
                "input_count": len(self._inputs),
                "expected_count": len(self.expected_inputs),
                "is_complete": self.is_complete(),
                "has_executed": self._executed,
                "has_error": self._error is not None,
                "conflict_resolution": self.conflict_resolution,
            }

    def __repr__(self) -> str:
        """String representation."""
        with self._lock:
            received = len(self._inputs)
            expected = len(self.expected_inputs)
            status = (
                "executed" if self._executed else ("complete" if self.is_complete() else "waiting")
            )
            return (
                f"MergeContext(action='{self.action_id}', "
                f"inputs={received}/{expected}, status={status})"
            )
