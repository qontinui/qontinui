"""State-aware scheduler adapted from Brobot's StateAwareScheduler.

Provides state-aware scheduling capabilities for automation tasks.
Ensures tasks run with proper state context and validation.
"""

import asyncio
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import Any

from .schedule_config import CheckMode, ExecutionRecord, ScheduleConfig, StateCheckResult

logger = logging.getLogger(__name__)


class StateAwareScheduler:
    """State-aware task scheduler.

    Adapted from Brobot's StateAwareScheduler (Java) to Python.
    Executes scheduled tasks with state validation and recovery.

    Key features:
    - State validation before execution
    - Flexible checking modes (CHECK_ALL, CHECK_INACTIVE_ONLY)
    - Auto-recovery via state rebuilding
    - Iteration limits
    - Configurable behavior

    This scheduler focuses on the intersection of scheduling and state awareness,
    delegating actual state operations to StateDetector and StateMemory.
    """

    def __init__(
        self,
        state_detector: Any | None = None,
        state_memory: Any | None = None,
        max_workers: int = 4,
    ) -> None:
        """Initialize the state-aware scheduler.

        Args:
            state_detector: StateDetector instance for checking states
            state_memory: StateMemory instance for tracking active states
            max_workers: Maximum concurrent executions
        """
        self.state_detector = state_detector
        self.state_memory = state_memory
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="StateAwareScheduler"
        )

        # Active schedules
        self._schedules: dict[str, ScheduleConfig] = {}
        self._schedule_futures: dict[str, Future] = {}
        self._iteration_counts: dict[str, int] = {}

        # Execution records
        self._execution_records: list[ExecutionRecord] = []

        # Control
        self._running = False
        self._lock = threading.Lock()

        logger.info("StateAwareScheduler initialized")

    def register_schedule(self, schedule: ScheduleConfig):
        """Register a new schedule.

        Args:
            schedule: Schedule configuration
        """
        with self._lock:
            self._schedules[schedule.id] = schedule
            self._iteration_counts[schedule.id] = 0
            logger.info(f"Registered schedule: {schedule.name} (ID: {schedule.id})")

    def unregister_schedule(self, schedule_id: str):
        """Remove a schedule.

        Args:
            schedule_id: ID of schedule to remove
        """
        with self._lock:
            if schedule_id in self._schedules:
                # Cancel any running execution
                if schedule_id in self._schedule_futures:
                    future = self._schedule_futures[schedule_id]
                    future.cancel()
                    del self._schedule_futures[schedule_id]

                del self._schedules[schedule_id]
                if schedule_id in self._iteration_counts:
                    del self._iteration_counts[schedule_id]

                logger.info(f"Unregistered schedule: {schedule_id}")

    def schedule_with_state_check(
        self,
        schedule: ScheduleConfig,
        task: Callable[[], bool],
    ) -> Future:
        """Schedule a task with state checking.

        Adapted from Brobot's scheduleWithStateCheck method.

        Args:
            schedule: Schedule configuration
            task: Task to execute after state validation

        Returns:
            Future that can be used to cancel the task
        """
        iteration_count = 0
        max_iterations = schedule.max_iterations

        def wrapped_task():
            nonlocal iteration_count

            # Create execution record
            record = ExecutionRecord(
                id=f"{schedule.id}_{datetime.now().timestamp()}",
                schedule_id=schedule.id,
                process_id=schedule.process_id,
                start_time=datetime.now(),
            )

            try:
                while True:
                    # Check iteration limit
                    if max_iterations > 0:
                        iteration_count += 1
                        if iteration_count > max_iterations:
                            logger.info(
                                f"Reached maximum iterations ({max_iterations}) for schedule '{schedule.name}', stopping"
                            )
                            record.status = "completed"
                            record.end_time = datetime.now()
                            record.iteration_count = iteration_count - 1
                            break

                    # Perform state check (async)
                    state_check_result = asyncio.run(self._perform_state_check_async(schedule))
                    record.state_checks.append(state_check_result)

                    if not state_check_result.check_passed:
                        if schedule.skip_if_states_missing:
                            logger.warning(
                                f"Skipping execution for '{schedule.name}': {state_check_result.error_message}"
                            )
                            record.status = "skipped"
                            record.error_message = state_check_result.error_message
                            record.end_time = datetime.now()
                            break
                        else:
                            logger.error(f"State check failed for '{schedule.name}'")
                            record.status = "failed"
                            record.error_message = "State check failed"
                            record.end_time = datetime.now()
                            break

                    # Execute task
                    try:
                        success = task()
                        if not success:
                            logger.warning(f"Task for schedule '{schedule.name}' returned False")
                            record.status = "failed"
                            record.error_message = "Task returned False"
                            record.end_time = datetime.now()
                            break
                    except Exception as e:
                        logger.error(f"Error executing task for '{schedule.name}': {e}")
                        record.status = "failed"
                        record.error_message = str(e)
                        record.end_time = datetime.now()
                        break

                    # For interval-based schedules, sleep
                    if schedule.interval_seconds:
                        time.sleep(schedule.interval_seconds)
                    else:
                        # Single execution
                        record.status = "success"
                        record.end_time = datetime.now()
                        record.iteration_count = iteration_count
                        break

            except Exception as e:
                logger.error(f"Unexpected error in scheduled task '{schedule.name}': {e}")
                record.status = "failed"
                record.error_message = f"Unexpected error: {e}"
                record.end_time = datetime.now()

            finally:
                # Store execution record
                with self._lock:
                    self._execution_records.append(record)
                    if schedule.id in self._iteration_counts:
                        self._iteration_counts[schedule.id] = iteration_count

        # Submit to executor
        future = self.executor.submit(wrapped_task)
        with self._lock:
            self._schedule_futures[schedule.id] = future

        return future

    async def _perform_state_check_async(self, schedule: ScheduleConfig) -> StateCheckResult:
        """Async version of state checking with parallel state detection.

        Uses parallel state detection for significantly faster checks when
        multiple states need to be verified. This is the new default approach.

        Performance improvement:
        - Sequential: 5 states × 500ms = 2.5 seconds
        - Async parallel: ~300-500ms regardless of state count

        Args:
            schedule: Schedule configuration

        Returns:
            StateCheckResult with check details
        """
        timestamp = datetime.now()
        active_states = self._get_active_states()
        active_state_set = set(active_states)

        # Determine which states need to be checked
        if schedule.check_mode == CheckMode.CHECK_ALL:
            states_to_check = schedule.required_states
            logger.debug(f"Checking all required states: {states_to_check}")
        else:
            # Only check states that are currently inactive (CHECK_INACTIVE_ONLY)
            states_to_check = [
                state for state in schedule.required_states if state not in active_state_set
            ]
            logger.debug(
                f"Checking only inactive states: {states_to_check} (active states: {active_states})"
            )

        # If all required states are already active and we're in CHECK_INACTIVE_ONLY mode
        if not states_to_check and schedule.check_mode == CheckMode.CHECK_INACTIVE_ONLY:
            logger.debug("All required states are already active, skipping state detection")
            return StateCheckResult(
                timestamp=timestamp,
                required_states=schedule.required_states,
                forbidden_states=schedule.forbidden_states,
                active_states=active_states,
                check_passed=True,
                check_mode=schedule.check_mode,
            )

        # Perform parallel state checking (KEY OPTIMIZATION)
        all_required_states_active = True
        if states_to_check and self.state_detector:
            # Use parallel async state detection
            found_states = await self.state_detector.find_states_parallel_async(states_to_check)

            # Check if all required states were found
            for state_name in states_to_check:
                if state_name not in found_states:
                    all_required_states_active = False
                    logger.info(f"Required state '{state_name}' is not active")

        # Re-check active states after detection
        active_states = self._get_active_states()
        active_state_set = set(active_states)
        all_required_states_active = all(
            state in active_state_set for state in schedule.required_states
        )

        # Check forbidden states
        forbidden_states_active = any(
            state in active_state_set for state in schedule.forbidden_states
        )
        if forbidden_states_active:
            forbidden_found = [
                state for state in schedule.forbidden_states if state in active_state_set
            ]
            logger.warning(f"Forbidden states are active: {forbidden_found}")
            return StateCheckResult(
                timestamp=timestamp,
                required_states=schedule.required_states,
                forbidden_states=schedule.forbidden_states,
                active_states=active_states,
                check_passed=False,
                check_mode=schedule.check_mode,
                error_message=f"Forbidden states active: {forbidden_found}",
            )

        states_rebuilt = False
        rebuild_success = None

        if not all_required_states_active:
            logger.info(
                f"Not all required states are active. Current: {active_states}, Required: {schedule.required_states}"
            )

            if schedule.rebuild_on_mismatch and self.state_detector:
                logger.info("Rebuilding active states (async)")
                states_rebuilt = True
                try:
                    # Use async rebuild
                    await self.state_detector.rebuild_active_states_async()
                    rebuild_success = True

                    # Re-check after rebuild
                    active_states = self._get_active_states()
                    active_state_set = set(active_states)
                    all_required_states_active = all(
                        state in active_state_set for state in schedule.required_states
                    )
                except Exception as e:
                    logger.error(f"Error rebuilding states: {e}")
                    rebuild_success = False

        check_passed = all_required_states_active and not forbidden_states_active

        result = StateCheckResult(
            timestamp=timestamp,
            required_states=schedule.required_states,
            forbidden_states=schedule.forbidden_states,
            active_states=active_states,
            check_passed=check_passed,
            check_mode=schedule.check_mode,
            states_rebuilt=states_rebuilt,
            rebuild_success=rebuild_success,
        )

        if not check_passed:
            result.error_message = f"Required states not active. Current: {active_states}, Required: {schedule.required_states}"

        logger.debug(f"Async state check completed. Check passed: {check_passed}")
        return result

    def _get_active_states(self) -> list[str]:
        """Get currently active state names.

        Returns:
            List of active state names
        """
        if self.state_memory:
            return self.state_memory.get_active_state_names()
        return []

    def get_execution_records(self, schedule_id: str | None = None) -> list[ExecutionRecord]:
        """Get execution records.

        Args:
            schedule_id: Optional schedule ID to filter by

        Returns:
            List of execution records
        """
        with self._lock:
            if schedule_id:
                return [
                    record
                    for record in self._execution_records
                    if record.schedule_id == schedule_id
                ]
            return list(self._execution_records)

    def get_schedule(self, schedule_id: str) -> ScheduleConfig | None:
        """Get a schedule by ID.

        Args:
            schedule_id: Schedule ID

        Returns:
            Schedule configuration or None
        """
        with self._lock:
            return self._schedules.get(schedule_id)

    def get_all_schedules(self) -> list[ScheduleConfig]:
        """Get all registered schedules.

        Returns:
            List of all schedules
        """
        with self._lock:
            return list(self._schedules.values())

    def get_iteration_count(self, schedule_id: str) -> int:
        """Get current iteration count for a schedule.

        Args:
            schedule_id: Schedule ID

        Returns:
            Iteration count
        """
        with self._lock:
            return self._iteration_counts.get(schedule_id, 0)

    def stop_schedule(self, schedule_id: str) -> bool:
        """Stop a running schedule.

        Args:
            schedule_id: Schedule ID

        Returns:
            True if schedule was stopped
        """
        with self._lock:
            if schedule_id in self._schedule_futures:
                future = self._schedule_futures[schedule_id]
                cancelled = future.cancel()
                if cancelled:
                    del self._schedule_futures[schedule_id]
                    logger.info(f"Stopped schedule: {schedule_id}")
                return cancelled
            return False

    def shutdown(self, wait: bool = True):
        """Shutdown the scheduler.

        Args:
            wait: Whether to wait for pending tasks
        """
        logger.info("Shutting down StateAwareScheduler")
        self.executor.shutdown(wait=wait)
        logger.info("StateAwareScheduler shutdown complete")

    def get_statistics(self) -> dict[str, Any]:
        """Get scheduler statistics.

        Returns:
            Dictionary of statistics
        """
        with self._lock:
            return {
                "total_schedules": len(self._schedules),
                "running_schedules": len(self._schedule_futures),
                "total_executions": len(self._execution_records),
                "successful_executions": sum(
                    1 for r in self._execution_records if r.status == "success"
                ),
                "failed_executions": sum(
                    1 for r in self._execution_records if r.status == "failed"
                ),
                "active_states": self._get_active_states(),
            }
