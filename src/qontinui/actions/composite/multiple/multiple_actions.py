"""MultipleActions - ported from Qontinui framework.

Execute multiple actions in parallel or with specific ordering.
"""

import threading
import time
from enum import Enum, auto
from typing import Any

from ....model.action.action_record import ActionRecord
from ...action_config import ActionConfig, ActionConfigBuilder
from ...action_interface import ActionInterface
from ...action_result import ActionResult
from ...action_type import ActionType
from ...object_collection import ObjectCollection
from .executor import TaskExecutor
from .strategies import (
    BaseExecutionStrategy,
    GroupedStrategy,
    ParallelStrategy,
    PriorityStrategy,
    RoundRobinStrategy,
)
from .task import ActionTask


class ExecutionStrategy(Enum):
    """Strategy for executing multiple actions."""

    PARALLEL = auto()  # Execute all in parallel
    ROUND_ROBIN = auto()  # Alternate between actions
    PRIORITY = auto()  # Execute by priority
    GROUPED = auto()  # Execute in groups


class MultipleActionsOptions(ActionConfig):
    """Options for multiple actions execution.

    Port of MultipleActionsOptions from Qontinui framework.

    This class encapsulates all parameters for executing multiple actions with
    various strategies including parallel execution, priority-based, round-robin,
    and grouped execution. It is an immutable object and must be constructed using
    its inner Builder.

    Example usage:
        options = MultipleActionsOptionsBuilder()
            .set_strategy(ExecutionStrategy.PARALLEL)
            .set_max_parallel(10)
            .set_timeout(30.0)
            .set_fail_fast(True)
            .build()
    """

    def __init__(self, builder: "MultipleActionsOptionsBuilder") -> None:
        """Initialize MultipleActionsOptions from builder.

        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self.strategy: ExecutionStrategy = builder.strategy
        self.max_parallel: int = builder.max_parallel
        self.timeout: float = builder.timeout
        self.wait_between_groups: float = builder.wait_between_groups
        self.fail_fast: bool = builder.fail_fast
        self.record_actions: bool = builder.record_actions
        self._pause_before_begin: float = builder.pause_before_begin
        self._pause_after_end: float = builder.pause_after_end

    def get_strategy(self) -> ExecutionStrategy:
        """Get the execution strategy."""
        return self.strategy

    def get_max_parallel(self) -> int:
        """Get the maximum number of parallel executions."""
        return self.max_parallel

    def get_timeout(self) -> float:
        """Get the timeout for all actions."""
        return self.timeout

    def get_wait_between_groups(self) -> float:
        """Get the wait time between groups."""
        return self.wait_between_groups

    def get_fail_fast(self) -> bool:
        """Get whether fail-fast mode is enabled."""
        return self.fail_fast

    def get_record_actions(self) -> bool:
        """Get whether action recording is enabled."""
        return self.record_actions

    # Override parent methods to return the private attributes
    def get_pause_before_begin(self) -> float:
        """Get pause duration before action begins."""
        return self._pause_before_begin

    def get_pause_after_end(self) -> float:
        """Get pause duration after action ends."""
        return self._pause_after_end


class MultipleActionsOptionsBuilder(ActionConfigBuilder):
    """Builder for constructing MultipleActionsOptions with a fluent API.

    Port of MultipleActionsOptions.Builder from Qontinui framework.
    """

    def __init__(self, original: MultipleActionsOptions | None = None) -> None:
        """Initialize builder.

        Args:
            original: Optional MultipleActionsOptions instance to copy values from
        """
        super().__init__(original)

        if original:
            self.strategy = original.strategy
            self.max_parallel = original.max_parallel
            self.timeout = original.timeout
            self.wait_between_groups = original.wait_between_groups
            self.fail_fast = original.fail_fast
            self.record_actions = original.record_actions
        else:
            self.strategy = ExecutionStrategy.PARALLEL
            self.max_parallel = 10
            self.timeout = 30.0
            self.wait_between_groups = 0.5
            self.fail_fast = False
            self.record_actions = True

    def set_strategy(
        self, strategy: ExecutionStrategy
    ) -> "MultipleActionsOptionsBuilder":
        """Set the execution strategy.

        Args:
            strategy: The execution strategy to use

        Returns:
            This builder instance for chaining
        """
        self.strategy = strategy
        return self

    def set_max_parallel(self, max_parallel: int) -> "MultipleActionsOptionsBuilder":
        """Set the maximum number of parallel executions.

        Args:
            max_parallel: Maximum parallel workers

        Returns:
            This builder instance for chaining
        """
        self.max_parallel = max_parallel
        return self

    def set_timeout(self, timeout: float) -> "MultipleActionsOptionsBuilder":
        """Set the overall timeout for all actions.

        Args:
            timeout: Timeout in seconds

        Returns:
            This builder instance for chaining
        """
        self.timeout = timeout
        return self

    def set_wait_between_groups(
        self, wait_between_groups: float
    ) -> "MultipleActionsOptionsBuilder":
        """Set the wait time between groups.

        Args:
            wait_between_groups: Wait time in seconds

        Returns:
            This builder instance for chaining
        """
        self.wait_between_groups = wait_between_groups
        return self

    def set_fail_fast(self, fail_fast: bool) -> "MultipleActionsOptionsBuilder":
        """Set whether to stop on first failure.

        Args:
            fail_fast: True to stop on first failure, False otherwise

        Returns:
            This builder instance for chaining
        """
        self.fail_fast = fail_fast
        return self

    def set_record_actions(
        self, record_actions: bool
    ) -> "MultipleActionsOptionsBuilder":
        """Set whether to record action execution.

        Args:
            record_actions: True to record actions, False otherwise

        Returns:
            This builder instance for chaining
        """
        self.record_actions = record_actions
        return self

    def parallel(self, max_workers: int = 10) -> "MultipleActionsOptionsBuilder":
        """Configure for parallel execution.

        Args:
            max_workers: Maximum parallel workers

        Returns:
            This builder instance for chaining
        """
        self.strategy = ExecutionStrategy.PARALLEL
        self.max_parallel = max_workers
        return self

    def round_robin(self) -> "MultipleActionsOptionsBuilder":
        """Configure for round-robin execution.

        Returns:
            This builder instance for chaining
        """
        self.strategy = ExecutionStrategy.ROUND_ROBIN
        return self

    def by_priority(self) -> "MultipleActionsOptionsBuilder":
        """Configure for priority-based execution.

        Returns:
            This builder instance for chaining
        """
        self.strategy = ExecutionStrategy.PRIORITY
        return self

    def grouped(self, wait_between: float = 0.5) -> "MultipleActionsOptionsBuilder":
        """Configure for grouped execution.

        Args:
            wait_between: Wait time between groups in seconds

        Returns:
            This builder instance for chaining
        """
        self.strategy = ExecutionStrategy.GROUPED
        self.wait_between_groups = wait_between
        return self

    def with_timeout(self, seconds: float) -> "MultipleActionsOptionsBuilder":
        """Set overall timeout (convenience method).

        Args:
            seconds: Timeout in seconds

        Returns:
            This builder instance for chaining
        """
        self.timeout = seconds
        return self

    def fail_fast_enabled(self) -> "MultipleActionsOptionsBuilder":
        """Enable fail-fast mode (convenience method).

        Returns:
            This builder instance for chaining
        """
        self.fail_fast = True
        return self

    def build(self) -> MultipleActionsOptions:
        """Build the immutable MultipleActionsOptions object.

        Returns:
            A new instance of MultipleActionsOptions
        """
        return MultipleActionsOptions(self)

    def _self(self) -> "MultipleActionsOptionsBuilder":
        """Return self for fluent interface.

        Returns:
            This builder instance
        """
        return self


class MultipleActions(ActionInterface):
    """Multiple actions execution implementation.

    Port of MultipleActions from Qontinui framework class.

    Executes multiple actions with various strategies including
    parallel execution, priority-based, round-robin, and grouped.
    """

    def __init__(self, options: MultipleActionsOptions | None = None) -> None:
        """Initialize MultipleActions.

        Args:
            options: Execution options
        """
        self.options = options or MultipleActionsOptionsBuilder().build()
        self._tasks: list[ActionTask] = []
        self._executor = TaskExecutor(
            max_workers=self.options.max_parallel,
            timeout=self.options.timeout,
            record_actions=self.options.record_actions,
        )
        self._strategy: BaseExecutionStrategy = self._create_strategy()
        self._stop_flag = threading.Event()

    def _create_strategy(self) -> BaseExecutionStrategy:
        """Create execution strategy based on options.

        Returns:
            Execution strategy instance
        """
        if self.options.strategy == ExecutionStrategy.PARALLEL:
            return ParallelStrategy(
                fail_fast=self.options.fail_fast,
                record_actions=self.options.record_actions,
            )
        elif self.options.strategy == ExecutionStrategy.ROUND_ROBIN:
            return RoundRobinStrategy(
                fail_fast=self.options.fail_fast,
                record_actions=self.options.record_actions,
            )
        elif self.options.strategy == ExecutionStrategy.PRIORITY:
            return PriorityStrategy(
                fail_fast=self.options.fail_fast,
                record_actions=self.options.record_actions,
            )
        elif self.options.strategy == ExecutionStrategy.GROUPED:
            return GroupedStrategy(
                fail_fast=self.options.fail_fast,
                record_actions=self.options.record_actions,
                wait_between_groups=self.options.wait_between_groups,
            )
        else:
            return ParallelStrategy(
                fail_fast=self.options.fail_fast,
                record_actions=self.options.record_actions,
            )

    def get_action_type(self) -> ActionType:
        """Return the action type.

        Returns:
            ActionType for the first action in tasks, or FIND if no tasks
        """
        if self._tasks:
            first_action = self._tasks[0].action
            if hasattr(first_action, "get_action_type"):
                return first_action.get_action_type()
        # Default to FIND as a generic action type
        return ActionType.FIND

    def perform(
        self, matches: ActionResult, *object_collections: ObjectCollection
    ) -> None:
        """Execute multiple actions using the Qontinui framework pattern.

        Args:
            matches: Contains ActionOptions and accumulates execution results
            object_collections: Collections containing targets for the actions
        """
        # Execute all actions
        success = self.execute()

        # Update matches with results
        object.__setattr__(matches, "success", success)

        # Add execution history to matches
        for record in self._executor.get_execution_history():
            matches.add_execution_record(record)  # type: ignore[arg-type, attr-defined]

    def add(
        self,
        action: ActionInterface,
        target: Any | None = None,
        priority: int = 0,
        group: int = 0,
        name: str | None = None,
    ) -> "MultipleActions":
        """Add action to execute.

        Args:
            action: Action to add
            target: Target for action
            priority: Execution priority
            group: Group number
            name: Optional name

        Returns:
            Self for fluent interface
        """
        task = ActionTask(action, target, priority, group, name)
        self._tasks.append(task)
        return self

    def add_all(self, actions: list[ActionInterface]) -> "MultipleActions":
        """Add multiple actions.

        Args:
            actions: List of actions

        Returns:
            Self for fluent interface
        """
        for action in actions:
            self.add(action)
        return self

    def execute(self) -> bool:
        """Execute all actions based on strategy.

        Returns:
            True if execution successful
        """
        if not self._tasks:
            return True

        self._stop_flag.clear()
        self._executor.clear_history()

        # Apply pre-action pause
        self._pause_before()

        # Execute using strategy
        result = self._strategy.execute(self._tasks, self._executor)

        # Apply post-action pause
        self._pause_after()

        return result

    def _pause_before(self):
        """Apply pre-action pause from options."""
        if self.options.pause_before_begin > 0:
            time.sleep(self.options.pause_before_begin)

    def _pause_after(self):
        """Apply post-action pause from options."""
        if self.options.pause_after_end > 0:
            time.sleep(self.options.pause_after_end)

    def get_results(self) -> dict[str, bool]:
        """Get results of all tasks.

        Returns:
            Dictionary of task name to result
        """
        return {
            task.name: (task.result if task.result is not None else False)
            for task in self._tasks
        }

    def get_successful_tasks(self) -> list[ActionTask]:
        """Get list of successful tasks.

        Returns:
            List of successful tasks
        """
        return [task for task in self._tasks if task.result]

    def get_failed_tasks(self) -> list[ActionTask]:
        """Get list of failed tasks.

        Returns:
            List of failed tasks
        """
        return [task for task in self._tasks if not task.result]

    def get_execution_history(self) -> list[ActionRecord]:
        """Get execution history.

        Returns:
            List of action records
        """
        return self._executor.get_execution_history()

    def clear(self) -> "MultipleActions":
        """Clear all tasks.

        Returns:
            Self for fluent interface
        """
        self._tasks.clear()
        self._executor.clear_history()
        return self

    def size(self) -> int:
        """Get number of tasks.

        Returns:
            Number of tasks
        """
        return len(self._tasks)

    @staticmethod
    def parallel(*actions: ActionInterface) -> bool:
        """Execute actions in parallel.

        Args:
            *actions: Actions to execute

        Returns:
            True if all successful
        """
        ma = MultipleActions(MultipleActionsOptionsBuilder().parallel().build())
        ma.add_all(list(actions))
        return ma.execute()
