"""Task executor for managing ThreadPool execution."""

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Callable

from ....model.action.action_record import ActionRecord
from .task import ActionTask


class TaskExecutor:
    """Manages ThreadPool execution of tasks."""

    def __init__(
        self,
        max_workers: int = 10,
        timeout: float = 30.0,
        record_actions: bool = True,
    ) -> None:
        """Initialize task executor.

        Args:
            max_workers: Maximum parallel workers
            timeout: Timeout for task execution
            record_actions: Whether to record actions
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.record_actions = record_actions
        self._execution_history: list[ActionRecord] = []

    def execute_parallel(
        self,
        tasks: list[ActionTask],
        on_complete: Callable[[ActionTask, bool], None] | None = None,
    ) -> bool:
        """Execute tasks in parallel.

        Args:
            tasks: Tasks to execute
            on_complete: Callback for each completed task

        Returns:
            True if at least one task completed
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(task.execute): task for task in tasks}
            completed = 0

            for future in as_completed(futures, timeout=self.timeout):
                task = futures[future]
                completed += 1

                try:
                    result = future.result()
                    if self.record_actions:
                        self._record_task(task)
                    if on_complete:
                        on_complete(task, result)
                except Exception as e:
                    print(f"Task {task.name} failed: {e}")
                    if on_complete:
                        on_complete(task, False)

            return completed > 0

    def execute_sequential(
        self,
        tasks: list[ActionTask],
        on_complete: Callable[[ActionTask, bool], None] | None = None,
    ) -> bool:
        """Execute tasks sequentially.

        Args:
            tasks: Tasks to execute
            on_complete: Callback for each completed task

        Returns:
            True if at least one task completed
        """
        completed = 0
        for task in tasks:
            result = task.execute()
            completed += 1

            if self.record_actions:
                self._record_task(task)
            if on_complete:
                on_complete(task, result)

        return completed > 0

    def _record_task(self, task: ActionTask):
        """Record task execution.

        Args:
            task: Executed task
        """
        record = ActionRecord(
            action_config=getattr(task.action, "options", None),
            text=task.name,
            duration=task.duration,
            action_success=task.result or False,
        )
        self._execution_history.append(record)

    def get_execution_history(self) -> list[ActionRecord]:
        """Get execution history.

        Returns:
            List of action records
        """
        return self._execution_history.copy()

    def clear_history(self):
        """Clear execution history."""
        self._execution_history.clear()
