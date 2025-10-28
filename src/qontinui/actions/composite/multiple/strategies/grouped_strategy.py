"""Grouped execution strategy."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_strategy import BaseExecutionStrategy


class GroupedStrategy(BaseExecutionStrategy):
    """Execute actions in groups with wait time between groups."""

    def __init__(
        self,
        fail_fast: bool = False,
        record_actions: bool = True,
        wait_between_groups: float = 0.5,
    ) -> None:
        """Initialize grouped strategy.

        Args:
            fail_fast: Stop execution on first failure
            record_actions: Whether to record action execution
            wait_between_groups: Wait time between groups in seconds
        """
        super().__init__(fail_fast, record_actions)
        self.wait_between_groups = wait_between_groups

    def execute(self, tasks, executor) -> bool:
        """Execute actions in groups.

        Each group is executed in parallel, with a wait between groups.

        Args:
            tasks: List of tasks to execute
            executor: Task executor

        Returns:
            True if all successful
        """
        # Group tasks by group number
        groups: dict[int, list] = {}
        for task in tasks:
            if task.group not in groups:
                groups[task.group] = []
            groups[task.group].append(task)

        all_success = True

        # Execute each group
        for group_num in sorted(groups.keys()):
            group_tasks = groups[group_num]

            # Execute group in parallel
            with ThreadPoolExecutor(max_workers=executor.max_workers) as pool:
                futures = {pool.submit(task.execute): task for task in group_tasks}

                for future in as_completed(futures, timeout=executor.timeout):
                    task = futures[future]

                    try:
                        result = future.result()
                        if not result:
                            all_success = False
                            if self.fail_fast:
                                return False

                        if self.record_actions:
                            executor._record_task(task)

                    except Exception as e:
                        print(f"Task {task.name} failed: {e}")
                        all_success = False
                        if self.fail_fast:
                            return False

            # Wait between groups
            if group_num < max(groups.keys()):
                time.sleep(self.wait_between_groups)

        return all_success
