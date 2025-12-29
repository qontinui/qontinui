"""Scheduler executor for managing multiple scheduled processes.

Orchestrates schedule execution with the JSONRunner and StateAwareScheduler.
"""

import logging
import threading
from datetime import datetime
from typing import Any, cast

from croniter import croniter  # type: ignore[import-untyped]
from qontinui_schemas.common import utc_now

from .schedule_config import ExecutionRecord, ScheduleConfig, TriggerType
from .state_aware_scheduler import StateAwareScheduler

logger = logging.getLogger(__name__)


class SchedulerExecutor:
    """Manages multiple scheduled processes.

    Integrates StateAwareScheduler with the JSONRunner to execute
    scheduled automation workflows.
    """

    def __init__(
        self,
        runner: Any,
        state_executor: Any | None = None,
        schedules: list[ScheduleConfig] | None = None,
    ) -> None:
        """Initialize the scheduler executor.

        Args:
            runner: JSONRunner instance
            state_executor: StateExecutor instance (for state detection)
            schedules: Initial schedules to register
        """
        self.runner = runner
        self.state_executor = state_executor

        # Initialize StateAwareScheduler with state detection
        state_detector = getattr(state_executor, "state_detector", None) if state_executor else None
        state_memory = getattr(state_executor, "state_memory", None) if state_executor else None

        self.scheduler = StateAwareScheduler(
            state_detector=state_detector,
            state_memory=state_memory,
        )

        # Register initial schedules
        if schedules:
            for schedule in schedules:
                if schedule.enabled:
                    self.register_schedule(schedule)

        # Control
        self._running = False
        self._scheduler_thread: threading.Thread | None = None
        self._lock = threading.Lock()

        logger.info("SchedulerExecutor initialized")

    def register_schedule(self, schedule: ScheduleConfig):
        """Register a new schedule and start execution if appropriate.

        Args:
            schedule: Schedule configuration
        """
        self.scheduler.register_schedule(schedule)

        # Start execution based on trigger type
        if schedule.trigger_type == TriggerType.INTERVAL:
            self._start_interval_schedule(schedule)
        elif schedule.trigger_type == TriggerType.TIME_BASED:
            self._start_time_based_schedule(schedule)
        # MANUAL and STATE_BASED schedules are started explicitly

        logger.info(f"Registered and started schedule: {schedule.name}")

    def unregister_schedule(self, schedule_id: str):
        """Remove a schedule.

        Args:
            schedule_id: ID of schedule to remove
        """
        self.scheduler.unregister_schedule(schedule_id)
        logger.info(f"Unregistered schedule: {schedule_id}")

    def execute_schedule(self, schedule_id: str) -> bool:
        """Manually execute a schedule.

        Args:
            schedule_id: ID of schedule to execute

        Returns:
            True if execution started successfully
        """
        schedule = self.scheduler.get_schedule(schedule_id)
        if not schedule:
            logger.error(f"Schedule not found: {schedule_id}")
            return False

        if not schedule.enabled:
            logger.warning(f"Schedule '{schedule.name}' is disabled")
            return False

        # Create task function that runs the process
        def task() -> bool:
            try:
                logger.info(
                    f"Executing process '{schedule.process_id}' for schedule '{schedule.name}'"
                )
                result = self.runner.run(schedule.process_id)

                # Update last executed time
                schedule.last_executed = utc_now()

                return cast(bool, result)
            except Exception as e:
                logger.error(f"Error executing schedule '{schedule.name}': {e}")
                return False

        # Execute with state checking
        self.scheduler.schedule_with_state_check(schedule, task)
        logger.info(f"Started execution for schedule: {schedule.name}")

        return True

    def _start_interval_schedule(self, schedule: ScheduleConfig):
        """Start an interval-based schedule.

        Args:
            schedule: Schedule configuration
        """
        if not schedule.interval_seconds:
            logger.error(f"Interval schedule '{schedule.name}' has no interval configured")
            return

        def task() -> bool:
            try:
                logger.info(
                    f"Executing process '{schedule.process_id}' for interval schedule '{schedule.name}'"
                )
                result = self.runner.run(schedule.process_id)
                schedule.last_executed = utc_now()
                return cast(bool, result)
            except Exception as e:
                logger.error(f"Error in interval schedule '{schedule.name}': {e}")
                return False

        # Schedule with initial delay
        def delayed_start():
            if schedule.initial_delay_seconds > 0:
                import time

                time.sleep(schedule.initial_delay_seconds)

            # Execute using state-aware scheduler
            self.scheduler.schedule_with_state_check(schedule, task)

        # Start in background thread
        thread = threading.Thread(target=delayed_start, daemon=True)
        thread.start()

    def _start_time_based_schedule(self, schedule: ScheduleConfig):
        """Start a time-based (cron) schedule.

        Args:
            schedule: Schedule configuration
        """
        if not schedule.cron_expression:
            logger.error(f"Time-based schedule '{schedule.name}' has no cron expression")
            return

        try:
            cron = croniter(schedule.cron_expression, utc_now())
        except Exception as e:
            logger.error(f"Invalid cron expression for schedule '{schedule.name}': {e}")
            return

        def task() -> bool:
            try:
                logger.info(
                    f"Executing process '{schedule.process_id}' for time-based schedule '{schedule.name}'"
                )
                result = self.runner.run(schedule.process_id)
                schedule.last_executed = utc_now()
                return cast(bool, result)
            except Exception as e:
                logger.error(f"Error in time-based schedule '{schedule.name}': {e}")
                return False

        def cron_loop():
            """Run cron loop in background."""
            import time

            while schedule.enabled:
                # Calculate next execution time
                next_run = cron.get_next(datetime)
                schedule.next_execution = next_run

                # Wait until next execution
                wait_seconds = (next_run - utc_now()).total_seconds()
                if wait_seconds > 0:
                    time.sleep(wait_seconds)

                # Check if still enabled
                if not schedule.enabled:
                    break

                # Check end time
                if schedule.end_time and utc_now() > schedule.end_time:
                    logger.info(f"Schedule '{schedule.name}' reached end time, stopping")
                    break

                # Execute with state checking
                self.scheduler.schedule_with_state_check(schedule, task)

        # Start cron loop in background
        thread = threading.Thread(target=cron_loop, daemon=True)
        thread.start()

    def start(self):
        """Start the scheduler executor.

        This enables all registered schedules.
        """
        with self._lock:
            if self._running:
                logger.warning("SchedulerExecutor already running")
                return

            self._running = True
            logger.info("SchedulerExecutor started")

    def stop(self):
        """Stop the scheduler executor.

        This stops all running schedules.
        """
        with self._lock:
            if not self._running:
                logger.warning("SchedulerExecutor not running")
                return

            self._running = False

            # Stop all schedules
            for schedule in self.scheduler.get_all_schedules():
                self.scheduler.stop_schedule(schedule.id)

            logger.info("SchedulerExecutor stopped")

    def shutdown(self):
        """Shutdown the scheduler executor.

        Stops all schedules and shuts down the underlying scheduler.
        """
        self.stop()
        self.scheduler.shutdown()
        logger.info("SchedulerExecutor shutdown complete")

    def get_schedule(self, schedule_id: str) -> ScheduleConfig | None:
        """Get a schedule by ID.

        Args:
            schedule_id: Schedule ID

        Returns:
            Schedule configuration or None
        """
        return self.scheduler.get_schedule(schedule_id)

    def get_all_schedules(self) -> list[ScheduleConfig]:
        """Get all registered schedules.

        Returns:
            List of all schedules
        """
        return self.scheduler.get_all_schedules()

    def get_execution_history(self, schedule_id: str | None = None) -> list[ExecutionRecord]:
        """Get execution history.

        Args:
            schedule_id: Optional schedule ID to filter by

        Returns:
            List of execution records
        """
        return self.scheduler.get_execution_records(schedule_id)

    def enable_schedule(self, schedule_id: str):
        """Enable a schedule.

        Args:
            schedule_id: Schedule ID
        """
        schedule = self.scheduler.get_schedule(schedule_id)
        if schedule:
            schedule.enabled = True
            logger.info(f"Enabled schedule: {schedule.name}")

            # Re-register to start execution
            self.register_schedule(schedule)

    def disable_schedule(self, schedule_id: str):
        """Disable a schedule.

        Args:
            schedule_id: Schedule ID
        """
        schedule = self.scheduler.get_schedule(schedule_id)
        if schedule:
            schedule.enabled = False
            self.scheduler.stop_schedule(schedule_id)
            logger.info(f"Disabled schedule: {schedule.name}")

    def get_statistics(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dictionary of statistics
        """
        base_stats = self.scheduler.get_statistics()

        # Add SchedulerExecutor-specific stats
        with self._lock:
            base_stats.update(
                {
                    "executor_running": self._running,
                    "enabled_schedules": sum(
                        1 for s in self.scheduler.get_all_schedules() if s.enabled
                    ),
                    "disabled_schedules": sum(
                        1 for s in self.scheduler.get_all_schedules() if not s.enabled
                    ),
                }
            )

        return base_stats
