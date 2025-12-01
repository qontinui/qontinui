"""Task executor for the scheduling system.

Executes scheduled tasks with proper error handling and monitoring.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime
from typing import Any

from .scheduled_task import ScheduledTask, TaskStatus

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Executes scheduled tasks with monitoring and timeout.

    Following Brobot principles:
    - Robust error handling
    - Task timeout support
    - Performance monitoring
    - Thread-safe execution
    """

    def __init__(self, max_workers: int = 4, default_timeout: float = 60.0) -> None:
        """Initialize the task executor.

        Args:
            max_workers: Maximum number of concurrent task executions
            default_timeout: Default timeout for tasks in seconds
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.default_timeout = default_timeout

        # Statistics
        self.total_executed = 0
        self.total_succeeded = 0
        self.total_failed = 0
        self.total_timeout = 0

        logger.info(f"TaskExecutor initialized with {max_workers} workers")

    def execute(self, task: ScheduledTask, timeout: float | None = None) -> bool:
        """Execute a task with timeout and monitoring.

        Args:
            task: Task to execute
            timeout: Timeout in seconds (uses default if None)

        Returns:
            True if task succeeded
        """
        if not task.task_function:
            logger.error(f"Task '{task.name}' has no function to execute")
            task.status = TaskStatus.FAILED
            task.last_error = "No task function defined"
            return False

        timeout = timeout or self.default_timeout

        logger.info(f"Executing task '{task.name}' with timeout {timeout}s")

        # Update task status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()

        # Submit task to executor
        future = self.executor.submit(self._execute_task_function, task)

        try:
            # Wait for completion with timeout
            success = future.result(timeout=timeout)

            # Update statistics
            self.total_executed += 1
            if success:
                self.total_succeeded += 1
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                logger.info(f"Task '{task.name}' completed successfully")
            else:
                self.total_failed += 1
                task.status = TaskStatus.FAILED
                logger.error(f"Task '{task.name}' failed")

            return success

        except TimeoutError:
            # Task timed out
            self.total_timeout += 1
            self.total_failed += 1

            task.status = TaskStatus.FAILED
            task.last_error = f"Task timed out after {timeout}s"

            # Cancel the future
            future.cancel()

            logger.error(f"Task '{task.name}' timed out after {timeout}s")
            return False

        except Exception as e:
            # Unexpected error
            self.total_failed += 1

            task.status = TaskStatus.FAILED
            task.last_error = str(e)

            logger.error(f"Task '{task.name}' raised exception: {e}")
            return False

    def _execute_task_function(self, task: ScheduledTask) -> bool:
        """Execute the actual task function.

        Args:
            task: Task to execute

        Returns:
            True if successful
        """
        try:
            # Measure execution time
            start_time = time.time()

            # Execute the task
            if task.task_function is None:
                raise RuntimeError("Task function is None")
            result = task.task_function()

            # Record execution time
            execution_time = time.time() - start_time
            task.metadata["execution_time"] = execution_time

            # Ensure boolean result
            success = bool(result)

            if not success:
                task.last_error = "Task function returned False"

            return success

        except Exception as e:
            task.last_error = str(e)
            logger.error(f"Exception in task '{task.name}': {e}")
            return False

    def execute_with_retry(
        self,
        task: ScheduledTask,
        max_retries: int | None = None,
        retry_delay: float = 1.0,
    ) -> bool:
        """Execute a task with automatic retry on failure.

        Args:
            task: Task to execute
            max_retries: Maximum retry attempts (uses task setting if None)
            retry_delay: Delay between retries in seconds

        Returns:
            True if task eventually succeeded
        """
        max_retries = max_retries or task.max_retries

        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.info(
                    f"Retrying task '{task.name}' (attempt {attempt + 1}/{max_retries + 1})"
                )
                time.sleep(retry_delay)

            if self.execute(task):
                task.retry_count = 0  # Reset on success
                return True

            task.retry_count = attempt + 1

        logger.error(f"Task '{task.name}' failed after {max_retries + 1} attempts")
        return False

    def shutdown(self, wait: bool = True):
        """Shutdown the executor.

        Args:
            wait: Whether to wait for pending tasks to complete
        """
        logger.info("Shutting down TaskExecutor")
        self.executor.shutdown(wait=wait)

    def get_statistics(self) -> dict[str, Any]:
        """Get executor statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "total_executed": self.total_executed,
            "total_succeeded": self.total_succeeded,
            "total_failed": self.total_failed,
            "total_timeout": self.total_timeout,
            "success_rate": (
                self.total_succeeded / self.total_executed if self.total_executed > 0 else 0.0
            ),
        }
