"""MultipleActions - ported from Qontinui framework.

Execute multiple actions in parallel or with specific ordering.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from ....model.action.action_record import ActionRecord
from ...action_config import ActionConfig
from ...action_interface import ActionInterface


class ExecutionStrategy(Enum):
    """Strategy for executing multiple actions."""

    PARALLEL = auto()  # Execute all in parallel
    ROUND_ROBIN = auto()  # Alternate between actions
    PRIORITY = auto()  # Execute by priority
    GROUPED = auto()  # Execute in groups


class ActionTask:
    """Individual action task for multiple execution."""

    def __init__(
        self,
        action: ActionInterface,
        target: Any | None = None,
        priority: int = 0,
        group: int = 0,
        name: str | None = None,
    ):
        """Initialize action task.

        Args:
            action: Action to execute
            target: Target for action
            priority: Priority (higher = earlier)
            group: Group number for grouped execution
            name: Optional name for identification
        """
        self.action = action
        self.target = target
        self.priority = priority
        self.group = group
        self.name = name or f"Task_{id(self)}"
        self.result: bool | None = None
        self.error: Exception | None = None
        self.start_time: float | None = None
        self.end_time: float | None = None

    def execute(self) -> bool:
        """Execute the action task.

        Returns:
            True if successful
        """
        self.start_time = time.time()
        try:
            if self.target is not None:
                if hasattr(self.action, "execute"):
                    self.result = self.action.execute(self.target)
                else:
                    self.result = self.action(self.target)
            else:
                if hasattr(self.action, "execute"):
                    self.result = self.action.execute()
                else:
                    self.result = self.action()

            self.end_time = time.time()
            return self.result

        except Exception as e:
            self.error = e
            self.result = False
            self.end_time = time.time()
            return False

    @property
    def duration(self) -> float:
        """Get execution duration.

        Returns:
            Duration in seconds
        """
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0


@dataclass
class MultipleActionsOptions(ActionConfig):
    """Options for multiple actions execution.

    Port of MultipleActionsOptions from Qontinui framework.
    """

    strategy: ExecutionStrategy = ExecutionStrategy.PARALLEL
    max_parallel: int = 10  # Maximum parallel executions
    timeout: float = 30.0  # Timeout for all actions
    wait_between_groups: float = 0.5  # Wait between groups
    fail_fast: bool = False  # Stop on first failure
    record_actions: bool = True

    def parallel(self, max_workers: int = 10) -> "MultipleActionsOptions":
        """Use parallel execution.

        Args:
            max_workers: Maximum parallel workers

        Returns:
            Self for fluent interface
        """
        self.strategy = ExecutionStrategy.PARALLEL
        self.max_parallel = max_workers
        return self

    def round_robin(self) -> "MultipleActionsOptions":
        """Use round-robin execution.

        Returns:
            Self for fluent interface
        """
        self.strategy = ExecutionStrategy.ROUND_ROBIN
        return self

    def by_priority(self) -> "MultipleActionsOptions":
        """Execute by priority.

        Returns:
            Self for fluent interface
        """
        self.strategy = ExecutionStrategy.PRIORITY
        return self

    def grouped(self, wait_between: float = 0.5) -> "MultipleActionsOptions":
        """Execute in groups.

        Args:
            wait_between: Wait time between groups

        Returns:
            Self for fluent interface
        """
        self.strategy = ExecutionStrategy.GROUPED
        self.wait_between_groups = wait_between
        return self

    def with_timeout(self, seconds: float) -> "MultipleActionsOptions":
        """Set overall timeout.

        Args:
            seconds: Timeout in seconds

        Returns:
            Self for fluent interface
        """
        self.timeout = seconds
        return self

    def fail_fast_enabled(self) -> "MultipleActionsOptions":
        """Enable fail-fast mode.

        Returns:
            Self for fluent interface
        """
        self.fail_fast = True
        return self


class MultipleActions(ActionInterface):
    """Multiple actions execution implementation.

    Port of MultipleActions from Qontinui framework class.

    Executes multiple actions with various strategies including
    parallel execution, priority-based, round-robin, and grouped.
    """

    def __init__(self, options: MultipleActionsOptions | None = None):
        """Initialize MultipleActions.

        Args:
            options: Execution options
        """
        self.options = options or MultipleActionsOptions()
        self._tasks: list[ActionTask] = []
        self._execution_history: list[ActionRecord] = []
        self._stop_flag = threading.Event()

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
        self._execution_history.clear()

        # Apply pre-action pause
        self._pause_before()

        # Execute based on strategy
        result = False
        if self.options.strategy == ExecutionStrategy.PARALLEL:
            result = self._execute_parallel()
        elif self.options.strategy == ExecutionStrategy.ROUND_ROBIN:
            result = self._execute_round_robin()
        elif self.options.strategy == ExecutionStrategy.PRIORITY:
            result = self._execute_by_priority()
        elif self.options.strategy == ExecutionStrategy.GROUPED:
            result = self._execute_grouped()

        # Apply post-action pause
        self._pause_after()

        return result

    def _execute_parallel(self) -> bool:
        """Execute actions in parallel.

        Returns:
            True if all successful or not fail-fast
        """
        with ThreadPoolExecutor(max_workers=self.options.max_parallel) as executor:
            # Submit all tasks
            futures = {executor.submit(task.execute): task for task in self._tasks}

            all_success = True
            completed = 0

            # Process completed tasks
            for future in as_completed(futures, timeout=self.options.timeout):
                task = futures[future]
                completed += 1

                try:
                    result = future.result()
                    if not result:
                        all_success = False
                        if self.options.fail_fast:
                            # Cancel remaining tasks
                            for f in futures:
                                f.cancel()
                            break

                    # Record action
                    if self.options.record_actions:
                        self._record_task(task)

                except Exception as e:
                    print(f"Task {task.name} failed: {e}")
                    all_success = False
                    if self.options.fail_fast:
                        break

            return all_success if self.options.fail_fast else completed > 0

    def _execute_round_robin(self) -> bool:
        """Execute actions in round-robin fashion.

        Returns:
            True if all successful
        """
        # Group tasks by target for round-robin
        task_queues: dict[int, list[ActionTask]] = {}
        for task in self._tasks:
            key = hash(str(task.target))
            if key not in task_queues:
                task_queues[key] = []
            task_queues[key].append(task)

        all_success = True
        max_rounds = max(len(queue) for queue in task_queues.values())

        for round_num in range(max_rounds):
            for queue in task_queues.values():
                if round_num < len(queue):
                    task = queue[round_num]

                    if not task.execute():
                        all_success = False
                        if self.options.fail_fast:
                            return False

                    if self.options.record_actions:
                        self._record_task(task)

        return all_success

    def _execute_by_priority(self) -> bool:
        """Execute actions by priority.

        Returns:
            True if all successful
        """
        # Sort by priority (higher first)
        sorted_tasks = sorted(self._tasks, key=lambda t: t.priority, reverse=True)

        all_success = True
        for task in sorted_tasks:
            if not task.execute():
                all_success = False
                if self.options.fail_fast:
                    return False

            if self.options.record_actions:
                self._record_task(task)

        return all_success

    def _execute_grouped(self) -> bool:
        """Execute actions in groups.

        Returns:
            True if all successful
        """
        # Group tasks
        groups: dict[int, list[ActionTask]] = {}
        for task in self._tasks:
            if task.group not in groups:
                groups[task.group] = []
            groups[task.group].append(task)

        all_success = True

        # Execute each group
        for group_num in sorted(groups.keys()):
            group_tasks = groups[group_num]

            # Execute group in parallel
            with ThreadPoolExecutor(max_workers=self.options.max_parallel) as executor:
                futures = {executor.submit(task.execute): task for task in group_tasks}

                for future in as_completed(futures, timeout=self.options.timeout):
                    task = futures[future]

                    try:
                        result = future.result()
                        if not result:
                            all_success = False
                            if self.options.fail_fast:
                                return False

                        if self.options.record_actions:
                            self._record_task(task)

                    except Exception as e:
                        print(f"Task {task.name} failed: {e}")
                        all_success = False
                        if self.options.fail_fast:
                            return False

            # Wait between groups
            if group_num < max(groups.keys()):
                time.sleep(self.options.wait_between_groups)

        return all_success

    def _record_task(self, task: ActionTask):
        """Record task execution.

        Args:
            task: Executed task
        """
        record = ActionRecord(
            action_config=getattr(task.action, "options", None),
            text=task.name,
            duration=task.duration,
            success=task.result or False,
        )
        self._execution_history.append(record)

    def _pause_before(self):
        """Apply pre-action pause from options."""
        if self.options.pause_before > 0:
            time.sleep(self.options.pause_before)

    def _pause_after(self):
        """Apply post-action pause from options."""
        if self.options.pause_after > 0:
            time.sleep(self.options.pause_after)

    def get_results(self) -> dict[str, bool]:
        """Get results of all tasks.

        Returns:
            Dictionary of task name to result
        """
        return {task.name: task.result for task in self._tasks}

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
        return self._execution_history.copy()

    def clear(self) -> "MultipleActions":
        """Clear all tasks.

        Returns:
            Self for fluent interface
        """
        self._tasks.clear()
        self._execution_history.clear()
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
        ma = MultipleActions(MultipleActionsOptions().parallel())
        ma.add_all(list(actions))
        return ma.execute()
