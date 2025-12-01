"""Tests for StateAwareScheduler."""

import time
from unittest.mock import Mock

from qontinui.scheduling import CheckMode, ScheduleConfig, StateAwareScheduler


class MockStateMemory:
    """Mock StateMemory for testing."""

    def __init__(self, active_states=None):
        self.active_states = active_states or []

    def get_active_state_names(self):
        return self.active_states


class MockStateDetector:
    """Mock StateDetector for testing."""

    def __init__(self, find_results=None, rebuild_results=None):
        self.find_results = find_results or {}
        self.rebuild_results = rebuild_results or []
        self.find_calls = []
        self.rebuild_calls = 0

    def find_state(self, state_name):
        self.find_calls.append(state_name)
        return self.find_results.get(state_name, False)

    def rebuild_active_states(self):
        self.rebuild_calls += 1
        # Update state memory with rebuilt states
        if hasattr(self, "state_memory") and self.rebuild_results:
            self.state_memory.active_states = self.rebuild_results


class TestStateAwareScheduler:
    """Tests for StateAwareScheduler class."""

    def test_scheduler_initialization(self):
        """Test initializing the scheduler."""
        state_memory = MockStateMemory()
        state_detector = MockStateDetector()

        scheduler = StateAwareScheduler(
            state_detector=state_detector, state_memory=state_memory, max_workers=2
        )

        assert scheduler.state_detector == state_detector
        assert scheduler.state_memory == state_memory
        assert len(scheduler._schedules) == 0

    def test_register_schedule(self):
        """Test registering a schedule."""
        scheduler = StateAwareScheduler()
        schedule = ScheduleConfig(
            id="test-1", name="Test Schedule", process_id="process-1"
        )

        scheduler.register_schedule(schedule)

        assert "test-1" in scheduler._schedules
        assert scheduler._schedules["test-1"] == schedule
        assert "test-1" in scheduler._iteration_counts
        assert scheduler._iteration_counts["test-1"] == 0

    def test_unregister_schedule(self):
        """Test unregistering a schedule."""
        scheduler = StateAwareScheduler()
        schedule = ScheduleConfig(
            id="test-1", name="Test Schedule", process_id="process-1"
        )

        scheduler.register_schedule(schedule)
        assert "test-1" in scheduler._schedules

        scheduler.unregister_schedule("test-1")
        assert "test-1" not in scheduler._schedules
        assert "test-1" not in scheduler._iteration_counts

    def test_state_check_all_states_active(self):
        """Test state check when all required states are active."""
        state_memory = MockStateMemory(active_states=["state1", "state2"])
        state_detector = MockStateDetector()
        scheduler = StateAwareScheduler(
            state_detector=state_detector, state_memory=state_memory
        )

        schedule = ScheduleConfig(
            id="test-1",
            name="Test",
            process_id="process-1",
            required_states=["state1", "state2"],
            check_mode=CheckMode.CHECK_INACTIVE_ONLY,
        )

        result = scheduler._perform_state_check(schedule)

        assert result.check_passed is True
        assert result.active_states == ["state1", "state2"]
        assert result.check_mode == CheckMode.CHECK_INACTIVE_ONLY
        assert len(state_detector.find_calls) == 0  # No detection needed

    def test_state_check_inactive_states(self):
        """Test state check with inactive states."""
        state_memory = MockStateMemory(active_states=["state1"])
        state_detector = MockStateDetector(find_results={"state2": True})
        scheduler = StateAwareScheduler(
            state_detector=state_detector, state_memory=state_memory
        )

        schedule = ScheduleConfig(
            id="test-1",
            name="Test",
            process_id="process-1",
            required_states=["state1", "state2"],
            check_mode=CheckMode.CHECK_INACTIVE_ONLY,
        )

        # Update state memory after detection
        state_memory.active_states = ["state1", "state2"]

        result = scheduler._perform_state_check(schedule)

        assert "state2" in state_detector.find_calls
        assert result.check_passed is True

    def test_state_check_with_forbidden_states(self):
        """Test state check with forbidden states present."""
        state_memory = MockStateMemory(active_states=["state1", "error"])
        state_detector = MockStateDetector()
        scheduler = StateAwareScheduler(
            state_detector=state_detector, state_memory=state_memory
        )

        schedule = ScheduleConfig(
            id="test-1",
            name="Test",
            process_id="process-1",
            required_states=["state1"],
            forbidden_states=["error"],
        )

        result = scheduler._perform_state_check(schedule)

        assert result.check_passed is False
        assert result.error_message is not None
        assert "error" in result.error_message

    def test_state_check_with_rebuild(self):
        """Test state check with state rebuilding."""
        state_memory = MockStateMemory(active_states=[])
        state_detector = MockStateDetector(
            find_results={"state1": False}, rebuild_results=["state1"]
        )
        state_detector.state_memory = state_memory
        scheduler = StateAwareScheduler(
            state_detector=state_detector, state_memory=state_memory
        )

        schedule = ScheduleConfig(
            id="test-1",
            name="Test",
            process_id="process-1",
            required_states=["state1"],
            rebuild_on_mismatch=True,
        )

        result = scheduler._perform_state_check(schedule)

        assert state_detector.rebuild_calls == 1
        assert result.states_rebuilt is True

    def test_state_check_check_all_mode(self):
        """Test CHECK_ALL mode checks all states."""
        state_memory = MockStateMemory(active_states=["state1", "state2"])
        state_detector = MockStateDetector(
            find_results={"state1": True, "state2": True}
        )
        scheduler = StateAwareScheduler(
            state_detector=state_detector, state_memory=state_memory
        )

        schedule = ScheduleConfig(
            id="test-1",
            name="Test",
            process_id="process-1",
            required_states=["state1", "state2"],
            check_mode=CheckMode.CHECK_ALL,
        )

        _result = scheduler._perform_state_check(schedule)

        # CHECK_ALL should check all required states
        assert "state1" in state_detector.find_calls
        assert "state2" in state_detector.find_calls

    def test_schedule_execution_with_iteration_limit(self):
        """Test schedule execution respects iteration limits."""
        state_memory = MockStateMemory(active_states=["state1"])
        state_detector = MockStateDetector()
        scheduler = StateAwareScheduler(
            state_detector=state_detector, state_memory=state_memory
        )

        execution_count = 0

        def task():
            nonlocal execution_count
            execution_count += 1
            time.sleep(0.01)  # Small delay
            return True

        schedule = ScheduleConfig(
            id="test-1",
            name="Test",
            process_id="process-1",
            required_states=["state1"],
            max_iterations=3,
            interval_seconds=0.05,  # Very short interval for testing
        )

        scheduler.register_schedule(schedule)
        _future = scheduler.schedule_with_state_check(schedule, task)

        # Wait for execution to complete
        time.sleep(0.5)

        # Should execute exactly 3 times
        assert execution_count == 3

        # Check execution record
        records = scheduler.get_execution_records(schedule.id)
        assert len(records) > 0
        assert records[0].status in ["completed"]

    def test_schedule_execution_with_state_failure(self):
        """Test schedule handles state check failures."""
        state_memory = MockStateMemory(active_states=[])
        state_detector = MockStateDetector()
        scheduler = StateAwareScheduler(
            state_detector=state_detector, state_memory=state_memory
        )

        task_executed = False

        def task():
            nonlocal task_executed
            task_executed = True
            return True

        schedule = ScheduleConfig(
            id="test-1",
            name="Test",
            process_id="process-1",
            required_states=["missing-state"],
            skip_if_states_missing=True,
            max_iterations=1,
        )

        scheduler.register_schedule(schedule)
        _future = scheduler.schedule_with_state_check(schedule, task)

        # Wait for execution
        time.sleep(0.2)

        # Task should not execute due to state failure
        assert task_executed is False

        # Check execution record shows skipped
        records = scheduler.get_execution_records(schedule.id)
        assert len(records) > 0
        assert records[0].status == "skipped"

    def test_get_execution_records(self):
        """Test retrieving execution records."""
        scheduler = StateAwareScheduler()

        # Add some mock execution records
        record1 = Mock()
        record1.schedule_id = "schedule-1"
        record2 = Mock()
        record2.schedule_id = "schedule-2"

        scheduler._execution_records = [record1, record2]

        # Get all records
        all_records = scheduler.get_execution_records()
        assert len(all_records) == 2

        # Get filtered records
        schedule1_records = scheduler.get_execution_records("schedule-1")
        assert len(schedule1_records) == 1
        assert schedule1_records[0].schedule_id == "schedule-1"

    def test_get_schedule(self):
        """Test retrieving a schedule by ID."""
        scheduler = StateAwareScheduler()
        schedule = ScheduleConfig(id="test-1", name="Test", process_id="process-1")

        scheduler.register_schedule(schedule)

        retrieved = scheduler.get_schedule("test-1")
        assert retrieved == schedule

        # Non-existent schedule
        assert scheduler.get_schedule("non-existent") is None

    def test_get_all_schedules(self):
        """Test retrieving all schedules."""
        scheduler = StateAwareScheduler()

        schedule1 = ScheduleConfig(id="test-1", name="Test 1", process_id="process-1")
        schedule2 = ScheduleConfig(id="test-2", name="Test 2", process_id="process-2")

        scheduler.register_schedule(schedule1)
        scheduler.register_schedule(schedule2)

        all_schedules = scheduler.get_all_schedules()
        assert len(all_schedules) == 2
        assert schedule1 in all_schedules
        assert schedule2 in all_schedules

    def test_get_iteration_count(self):
        """Test getting iteration count for a schedule."""
        scheduler = StateAwareScheduler()
        schedule = ScheduleConfig(id="test-1", name="Test", process_id="process-1")

        scheduler.register_schedule(schedule)
        assert scheduler.get_iteration_count("test-1") == 0

        # Manually increment for testing
        scheduler._iteration_counts["test-1"] = 5
        assert scheduler.get_iteration_count("test-1") == 5

    def test_stop_schedule(self):
        """Test stopping a running schedule."""
        scheduler = StateAwareScheduler()
        schedule = ScheduleConfig(
            id="test-1",
            name="Test",
            process_id="process-1",
            interval_seconds=1,
            max_iterations=-1,
        )

        def task():
            time.sleep(0.1)
            return True

        scheduler.register_schedule(schedule)
        _future = scheduler.schedule_with_state_check(schedule, task)

        # Let it run a bit
        time.sleep(0.05)

        # Stop it
        stopped = scheduler.stop_schedule("test-1")
        assert stopped is True

    def test_statistics(self):
        """Test getting scheduler statistics."""
        state_memory = MockStateMemory(active_states=["state1", "state2"])
        state_detector = MockStateDetector()
        scheduler = StateAwareScheduler(
            state_detector=state_detector, state_memory=state_memory
        )

        schedule = ScheduleConfig(id="test-1", name="Test", process_id="process-1")
        scheduler.register_schedule(schedule)

        stats = scheduler.get_statistics()

        assert stats["total_schedules"] == 1
        assert stats["active_states"] == ["state1", "state2"]
        assert "total_executions" in stats
        assert "successful_executions" in stats
        assert "failed_executions" in stats

    def test_shutdown(self):
        """Test shutting down the scheduler."""
        scheduler = StateAwareScheduler(max_workers=2)

        schedule = ScheduleConfig(
            id="test-1",
            name="Test",
            process_id="process-1",
            interval_seconds=1,
        )

        def task():
            time.sleep(0.1)
            return True

        scheduler.register_schedule(schedule)
        _future = scheduler.schedule_with_state_check(schedule, task)

        # Shutdown
        scheduler.shutdown(wait=False)

        # Executor should be shut down
        assert scheduler.executor._shutdown is True
