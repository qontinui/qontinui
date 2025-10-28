"""Priority-based execution strategy."""

from .base_strategy import BaseExecutionStrategy


class PriorityStrategy(BaseExecutionStrategy):
    """Execute actions by priority order."""

    def execute(self, tasks, executor) -> bool:
        """Execute actions by priority (higher first).

        Args:
            tasks: List of tasks to execute
            executor: Task executor

        Returns:
            True if all successful
        """
        # Sort by priority (higher first)
        sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)

        all_success = True
        for task in sorted_tasks:
            if not task.execute():
                all_success = False
                if self.fail_fast:
                    return False

            if self.record_actions:
                executor._record_task(task)

        return all_success
