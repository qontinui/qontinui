"""Round-robin execution strategy."""

from .base_strategy import BaseExecutionStrategy


class RoundRobinStrategy(BaseExecutionStrategy):
    """Execute actions in round-robin fashion."""

    def execute(self, tasks, executor) -> bool:
        """Execute actions in round-robin fashion.

        Groups tasks by target hash and alternates between groups.

        Args:
            tasks: List of tasks to execute
            executor: Task executor

        Returns:
            True if all successful
        """
        # Group tasks by target for round-robin
        task_queues: dict[int, list] = {}
        for task in tasks:
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
                        if self.fail_fast:
                            return False

                    if self.record_actions:
                        executor._record_task(task)

        return all_success
