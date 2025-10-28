"""State-aware scheduler for Qontinui.

Schedules and executes tasks based on application state.
"""

import heapq
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from ..actions import Actions
from ..model.state.path_finder import PathFinder
from .scheduled_task import ScheduledTask, TaskPriority, TaskStatus
from .state_tracker import StateTracker

logger = logging.getLogger(__name__)


class StateScheduler:
    """State-aware task scheduler.

    Following Brobot principles:
    - Tasks execute based on current state
    - Automatic path finding to reach required states
    - Priority-based task queue
    - Handles state transitions for tasks
    """

    def __init__(self, actions: Actions | None = None, path_finder: PathFinder | None = None) -> None:
        """Initialize the scheduler.

        Args:
            actions: Actions instance for executing transitions
            path_finder: PathFinder for navigating between states
        """
        self.actions = actions or Actions()
        self.path_finder = path_finder or PathFinder()
        self.state_tracker = StateTracker()

        # Task management
        self._task_queue: list[ScheduledTask] = []  # Priority queue
        self._blocked_tasks: list[ScheduledTask] = []  # Tasks blocked by state
        self._completed_tasks: list[ScheduledTask] = []
        self._running_task: ScheduledTask | None = None

        # Execution control
        self._running = False
        self._executor_thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Configuration
        self.max_path_length = 5  # Maximum transitions to reach required state
        self.state_check_interval = 0.5  # Seconds between state checks
        self.task_timeout = 60.0  # Maximum task execution time

        # Register state change listener
        self.state_tracker.add_listener(self._on_state_change)

        logger.info("StateScheduler initialized")

    def add_task(self, task: ScheduledTask) -> bool:
        """Add a task to the scheduler.

        Args:
            task: Task to schedule

        Returns:
            True if task was added
        """
        with self._lock:
            # Check if task can run in current states
            current_states = self.state_tracker.get_active_states()

            if task.can_run_in_states(current_states):
                task.status = TaskStatus.READY
                heapq.heappush(self._task_queue, task)
                logger.info(f"Task '{task.name}' added to ready queue")
            else:
                task.status = TaskStatus.BLOCKED
                self._blocked_tasks.append(task)
                logger.info(f"Task '{task.name}' blocked by state requirements")

            return True

    def schedule_task(
        self,
        name: str,
        task_function: Callable[[], bool],
        required_states: set[str] | None = None,
        forbidden_states: set[str] | None = None,
        target_state: str | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        recurring: bool = False,
    ) -> ScheduledTask:
        """Create and schedule a task.

        Args:
            name: Task name
            task_function: Function to execute
            required_states: States that must be active
            forbidden_states: States that must not be active
            target_state: State to navigate to before execution
            priority: Task priority
            recurring: Whether task should repeat

        Returns:
            The scheduled task
        """
        task = ScheduledTask(
            name=name,
            task_function=task_function,
            required_states=required_states or set(),
            forbidden_states=forbidden_states or set(),
            target_state=target_state,
            priority=priority,
            recurring=recurring,
        )

        self.add_task(task)
        return task

    def start(self):
        """Start the scheduler execution thread."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._executor_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self._executor_thread.start()
        logger.info("Scheduler started")

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._executor_thread:
            self._executor_thread.join(timeout=5.0)
            self._executor_thread = None
        logger.info("Scheduler stopped")

    def execute_next(self) -> bool:
        """Execute the next ready task.

        Returns:
            True if a task was executed
        """
        task = self._get_next_task()
        if not task:
            return False

        return self._execute_task(task)

    def _execution_loop(self):
        """Main execution loop for the scheduler."""
        logger.info("Scheduler execution loop started")

        while self._running:
            try:
                # Check for ready tasks
                if self.execute_next():
                    # Small delay after task execution
                    time.sleep(0.1)
                else:
                    # No tasks ready, wait a bit
                    time.sleep(self.state_check_interval)

            except Exception as e:
                logger.error(f"Error in scheduler execution loop: {e}")
                time.sleep(1.0)

        logger.info("Scheduler execution loop stopped")

    def _get_next_task(self) -> ScheduledTask | None:
        """Get the next task to execute.

        Returns:
            Next task or None
        """
        with self._lock:
            current_states = self.state_tracker.get_active_states()

            # Check if any blocked tasks are now ready
            self._check_blocked_tasks(current_states)

            # Find highest priority ready task
            while self._task_queue:
                task = heapq.heappop(self._task_queue)

                if task.is_ready(current_states):
                    self._running_task = task
                    return task
                elif task.status == TaskStatus.PENDING:
                    # Task is blocked, move to blocked queue
                    task.status = TaskStatus.BLOCKED
                    self._blocked_tasks.append(task)

            return None

    def _execute_task(self, task: ScheduledTask) -> bool:
        """Execute a scheduled task.

        Args:
            task: Task to execute

        Returns:
            True if task executed successfully
        """
        logger.info(f"Executing task: {task.name}")

        try:
            # Navigate to target state if needed
            if task.target_state:
                if not self._navigate_to_state(task.target_state):
                    logger.error(f"Failed to navigate to state: {task.target_state}")
                    task.status = TaskStatus.FAILED
                    task.last_error = "Failed to reach target state"
                    return False

            # Execute the task
            success = task.execute()

            if success:
                logger.info(f"Task '{task.name}' completed successfully")

                # Handle recurring tasks
                if task.recurring:
                    task.reset()
                    self.add_task(task)
                else:
                    self._completed_tasks.append(task)
            else:
                logger.error(f"Task '{task.name}' failed: {task.last_error}")

                # Retry if needed
                if task.should_retry():
                    task.status = TaskStatus.PENDING
                    self.add_task(task)
                else:
                    self._completed_tasks.append(task)

            return success

        finally:
            with self._lock:
                self._running_task = None

    def _navigate_to_state(self, target_state: str) -> bool:
        """Navigate to a target state.

        Args:
            target_state: State to navigate to

        Returns:
            True if navigation successful
        """
        current_states = self.state_tracker.get_active_states()

        # Already in target state
        if target_state in current_states:
            return True

        # Find path to target state
        # This is simplified - real implementation would use PathFinder
        logger.info(f"Navigating to state: {target_state}")

        # For now, just update the tracker
        # In real implementation, would execute transitions
        self.state_tracker.activate_state(target_state)
        return True

    def _check_blocked_tasks(self, current_states: set[str]):
        """Check if blocked tasks can now run.

        Args:
            current_states: Current active states
        """
        unblocked = []

        for i, task in enumerate(self._blocked_tasks):
            if task.can_run_in_states(current_states):
                task.status = TaskStatus.READY
                heapq.heappush(self._task_queue, task)
                unblocked.append(i)
                logger.info(f"Task '{task.name}' unblocked")

        # Remove unblocked tasks from blocked list
        for i in reversed(unblocked):
            del self._blocked_tasks[i]

    def _on_state_change(self, active_states: set[str]):
        """Handle state change notification.

        Args:
            active_states: New active states
        """
        logger.debug(f"State change detected: {active_states}")

        # Check blocked tasks on state change
        with self._lock:
            self._check_blocked_tasks(active_states)

    def get_task_status(self, task_name: str) -> TaskStatus | None:
        """Get status of a task by name.

        Args:
            task_name: Name of task

        Returns:
            Task status or None
        """
        # Check all task lists
        all_tasks = list(self._task_queue) + self._blocked_tasks + self._completed_tasks

        if self._running_task and self._running_task.name == task_name:
            return self._running_task.status

        for task in all_tasks:
            if task.name == task_name:
                return task.status

        return None

    def get_statistics(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            return {
                "ready_tasks": len(self._task_queue),
                "blocked_tasks": len(self._blocked_tasks),
                "completed_tasks": len(self._completed_tasks),
                "running_task": self._running_task.name if self._running_task else None,
                "active_states": list(self.state_tracker.get_active_states()),
                "state_statistics": self.state_tracker.get_state_statistics(),
            }
