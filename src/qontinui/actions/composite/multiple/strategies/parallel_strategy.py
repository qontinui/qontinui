"""Parallel execution strategy."""

from concurrent.futures import as_completed

from .base_strategy import BaseExecutionStrategy


class ParallelStrategy(BaseExecutionStrategy):
    """Execute all tasks in parallel."""

    def execute(self, tasks, executor) -> bool:
        """Execute all tasks in parallel.

        Args:
            tasks: List of tasks to execute
            executor: Task executor with ThreadPool

        Returns:
            True if all successful (or not fail-fast)
        """
        from concurrent.futures import ThreadPoolExecutor

        all_success = True
        completed = 0

        with ThreadPoolExecutor(max_workers=executor.max_workers) as pool:
            futures = {pool.submit(task.execute): task for task in tasks}

            for future in as_completed(futures, timeout=executor.timeout):
                task = futures[future]
                completed += 1

                try:
                    result = future.result()
                    if not result:
                        all_success = False
                        if self.fail_fast:
                            # Cancel remaining tasks
                            for f in futures:
                                f.cancel()
                            break

                    # Record action
                    if self.record_actions:
                        executor._record_task(task)

                except Exception as e:
                    print(f"Task {task.name} failed: {e}")
                    all_success = False
                    if self.fail_fast:
                        break

        return all_success if self.fail_fast else completed > 0
