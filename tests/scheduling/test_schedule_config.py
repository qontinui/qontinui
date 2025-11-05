"""Tests for ScheduleConfig and related dataclasses."""

from datetime import datetime

from qontinui.scheduling import (
    CheckMode,
    ExecutionRecord,
    ScheduleConfig,
    ScheduleType,
    StateCheckResult,
    TriggerType,
)


class TestScheduleConfig:
    """Tests for ScheduleConfig dataclass."""

    def test_schedule_config_creation(self):
        """Test creating a basic ScheduleConfig."""
        config = ScheduleConfig(
            id="test-schedule",
            name="Test Schedule",
            description="Test description",
            process_id="test-process",
        )

        assert config.id == "test-schedule"
        assert config.name == "Test Schedule"
        assert config.description == "Test description"
        assert config.process_id == "test-process"
        assert config.trigger_type == TriggerType.MANUAL
        assert config.enabled is True
        assert config.priority == 10

    def test_schedule_config_with_time_based_trigger(self):
        """Test creating a time-based schedule."""
        config = ScheduleConfig(
            id="daily-schedule",
            name="Daily Schedule",
            process_id="daily-process",
            trigger_type=TriggerType.TIME_BASED,
            cron_expression="0 9 * * *",
            start_time=datetime(2025, 1, 13, 9, 0, 0),
            end_time=datetime(2025, 12, 31, 23, 59, 59),
        )

        assert config.trigger_type == TriggerType.TIME_BASED
        assert config.cron_expression == "0 9 * * *"
        assert config.start_time is not None
        assert config.end_time is not None

    def test_schedule_config_with_interval_trigger(self):
        """Test creating an interval-based schedule."""
        config = ScheduleConfig(
            id="interval-schedule",
            name="Interval Schedule",
            process_id="interval-process",
            trigger_type=TriggerType.INTERVAL,
            interval_seconds=300,
            initial_delay_seconds=60,
        )

        assert config.trigger_type == TriggerType.INTERVAL
        assert config.interval_seconds == 300
        assert config.initial_delay_seconds == 60

    def test_schedule_config_with_state_requirements(self):
        """Test schedule with state requirements."""
        config = ScheduleConfig(
            id="state-schedule",
            name="State Schedule",
            process_id="state-process",
            required_states=["main-menu", "logged-in"],
            forbidden_states=["error-dialog"],
            check_mode=CheckMode.CHECK_INACTIVE_ONLY,
            rebuild_on_mismatch=True,
            skip_if_states_missing=False,
        )

        assert config.required_states == ["main-menu", "logged-in"]
        assert config.forbidden_states == ["error-dialog"]
        assert config.check_mode == CheckMode.CHECK_INACTIVE_ONLY
        assert config.rebuild_on_mismatch is True
        assert config.skip_if_states_missing is False

    def test_schedule_config_with_iteration_limit(self):
        """Test schedule with iteration limit."""
        config = ScheduleConfig(
            id="limited-schedule",
            name="Limited Schedule",
            process_id="limited-process",
            max_iterations=10,
        )

        assert config.max_iterations == 10

    def test_schedule_config_to_dict(self):
        """Test converting ScheduleConfig to dictionary."""
        config = ScheduleConfig(
            id="test-schedule",
            name="Test Schedule",
            description="Test description",
            process_id="test-process",
            trigger_type=TriggerType.INTERVAL,
            interval_seconds=300,
            required_states=["main-menu"],
            forbidden_states=["error"],
            max_iterations=5,
            enabled=True,
            priority=10,
        )

        data = config.to_dict()

        assert data["id"] == "test-schedule"
        assert data["name"] == "Test Schedule"
        assert data["processId"] == "test-process"
        assert data["triggerType"] == "interval"
        assert data["intervalSeconds"] == 300
        assert data["requiredStates"] == ["main-menu"]
        assert data["forbiddenStates"] == ["error"]
        assert data["maxIterations"] == 5
        assert data["enabled"] is True
        assert data["priority"] == 10

    def test_schedule_config_from_dict(self):
        """Test creating ScheduleConfig from dictionary."""
        data = {
            "id": "test-schedule",
            "name": "Test Schedule",
            "description": "Test description",
            "processId": "test-process",
            "triggerType": "interval",
            "scheduleType": "fixed_rate",
            "intervalSeconds": 300,
            "initialDelaySeconds": 60,
            "requiredStates": ["main-menu"],
            "forbiddenStates": ["error"],
            "checkMode": "CHECK_INACTIVE_ONLY",
            "rebuildOnMismatch": True,
            "skipIfStatesMissing": False,
            "maxIterations": 5,
            "enabled": True,
            "priority": 10,
        }

        config = ScheduleConfig.from_dict(data)

        assert config.id == "test-schedule"
        assert config.name == "Test Schedule"
        assert config.process_id == "test-process"
        assert config.trigger_type == TriggerType.INTERVAL
        assert config.schedule_type == ScheduleType.FIXED_RATE
        assert config.interval_seconds == 300
        assert config.initial_delay_seconds == 60
        assert config.required_states == ["main-menu"]
        assert config.forbidden_states == ["error"]
        assert config.check_mode == CheckMode.CHECK_INACTIVE_ONLY
        assert config.rebuild_on_mismatch is True
        assert config.skip_if_states_missing is False
        assert config.max_iterations == 5
        assert config.enabled is True
        assert config.priority == 10

    def test_schedule_config_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        original = ScheduleConfig(
            id="roundtrip-test",
            name="Roundtrip Test",
            process_id="test-process",
            trigger_type=TriggerType.TIME_BASED,
            cron_expression="0 9 * * *",
            required_states=["state1", "state2"],
            forbidden_states=["state3"],
            check_mode=CheckMode.CHECK_ALL,
            max_iterations=10,
        )

        data = original.to_dict()
        restored = ScheduleConfig.from_dict(data)

        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.process_id == original.process_id
        assert restored.trigger_type == original.trigger_type
        assert restored.cron_expression == original.cron_expression
        assert restored.required_states == original.required_states
        assert restored.forbidden_states == original.forbidden_states
        assert restored.check_mode == original.check_mode
        assert restored.max_iterations == original.max_iterations


class TestExecutionRecord:
    """Tests for ExecutionRecord dataclass."""

    def test_execution_record_creation(self):
        """Test creating an ExecutionRecord."""
        start = datetime.now()
        record = ExecutionRecord(
            id="exec-1",
            schedule_id="schedule-1",
            process_id="process-1",
            start_time=start,
        )

        assert record.id == "exec-1"
        assert record.schedule_id == "schedule-1"
        assert record.process_id == "process-1"
        assert record.start_time == start
        assert record.status == "running"
        assert record.iteration_count == 0

    def test_execution_record_duration(self):
        """Test calculating execution duration."""
        start = datetime(2025, 1, 13, 10, 0, 0)
        end = datetime(2025, 1, 13, 10, 5, 30)

        record = ExecutionRecord(
            id="exec-1",
            schedule_id="schedule-1",
            process_id="process-1",
            start_time=start,
            end_time=end,
        )

        duration = record.duration_seconds()
        assert duration == 330.0  # 5 minutes 30 seconds

    def test_execution_record_duration_no_end(self):
        """Test duration when execution hasn't ended."""
        record = ExecutionRecord(
            id="exec-1",
            schedule_id="schedule-1",
            process_id="process-1",
            start_time=datetime.now(),
        )

        assert record.duration_seconds() is None

    def test_execution_record_to_dict(self):
        """Test converting ExecutionRecord to dictionary."""
        start = datetime(2025, 1, 13, 10, 0, 0)
        end = datetime(2025, 1, 13, 10, 5, 0)

        record = ExecutionRecord(
            id="exec-1",
            schedule_id="schedule-1",
            process_id="process-1",
            start_time=start,
            end_time=end,
            status="success",
            iteration_count=5,
        )

        data = record.to_dict()

        assert data["id"] == "exec-1"
        assert data["scheduleId"] == "schedule-1"
        assert data["processId"] == "process-1"
        assert data["status"] == "success"
        assert data["iterationCount"] == 5
        assert data["durationSeconds"] == 300.0


class TestStateCheckResult:
    """Tests for StateCheckResult dataclass."""

    def test_state_check_result_creation(self):
        """Test creating a StateCheckResult."""
        timestamp = datetime.now()
        result = StateCheckResult(
            timestamp=timestamp,
            required_states=["state1", "state2"],
            forbidden_states=["state3"],
            active_states=["state1", "state2"],
            check_passed=True,
            check_mode=CheckMode.CHECK_ALL,
        )

        assert result.timestamp == timestamp
        assert result.required_states == ["state1", "state2"]
        assert result.forbidden_states == ["state3"]
        assert result.active_states == ["state1", "state2"]
        assert result.check_passed is True
        assert result.check_mode == CheckMode.CHECK_ALL
        assert result.states_rebuilt is False

    def test_state_check_result_with_rebuild(self):
        """Test StateCheckResult with state rebuilding."""
        result = StateCheckResult(
            timestamp=datetime.now(),
            required_states=["state1"],
            forbidden_states=[],
            active_states=["state1"],
            check_passed=True,
            check_mode=CheckMode.CHECK_INACTIVE_ONLY,
            states_rebuilt=True,
            rebuild_success=True,
        )

        assert result.states_rebuilt is True
        assert result.rebuild_success is True

    def test_state_check_result_failed(self):
        """Test failed state check."""
        result = StateCheckResult(
            timestamp=datetime.now(),
            required_states=["state1", "state2"],
            forbidden_states=[],
            active_states=["state1"],
            check_passed=False,
            check_mode=CheckMode.CHECK_ALL,
            error_message="State 'state2' is not active",
        )

        assert result.check_passed is False
        assert result.error_message == "State 'state2' is not active"

    def test_state_check_result_to_dict(self):
        """Test converting StateCheckResult to dictionary."""
        timestamp = datetime(2025, 1, 13, 10, 0, 0)
        result = StateCheckResult(
            timestamp=timestamp,
            required_states=["state1"],
            forbidden_states=["state2"],
            active_states=["state1"],
            check_passed=True,
            check_mode=CheckMode.CHECK_INACTIVE_ONLY,
        )

        data = result.to_dict()

        assert data["requiredStates"] == ["state1"]
        assert data["forbiddenStates"] == ["state2"]
        assert data["activeStates"] == ["state1"]
        assert data["checkPassed"] is True
        assert data["checkMode"] == "CHECK_INACTIVE_ONLY"


class TestTriggerType:
    """Tests for TriggerType enum."""

    def test_trigger_type_values(self):
        """Test TriggerType enum values."""
        assert TriggerType.TIME_BASED.value == "time_based"
        assert TriggerType.INTERVAL.value == "interval"
        assert TriggerType.STATE_BASED.value == "state_based"
        assert TriggerType.MANUAL.value == "manual"

    def test_trigger_type_from_string(self):
        """Test creating TriggerType from string."""
        assert TriggerType("time_based") == TriggerType.TIME_BASED
        assert TriggerType("interval") == TriggerType.INTERVAL
        assert TriggerType("state_based") == TriggerType.STATE_BASED
        assert TriggerType("manual") == TriggerType.MANUAL


class TestCheckMode:
    """Tests for CheckMode enum."""

    def test_check_mode_values(self):
        """Test CheckMode enum values."""
        assert CheckMode.CHECK_ALL.value == "CHECK_ALL"
        assert CheckMode.CHECK_INACTIVE_ONLY.value == "CHECK_INACTIVE_ONLY"

    def test_check_mode_from_string(self):
        """Test creating CheckMode from string."""
        assert CheckMode("CHECK_ALL") == CheckMode.CHECK_ALL
        assert CheckMode("CHECK_INACTIVE_ONLY") == CheckMode.CHECK_INACTIVE_ONLY


class TestScheduleType:
    """Tests for ScheduleType enum."""

    def test_schedule_type_values(self):
        """Test ScheduleType enum values."""
        assert ScheduleType.FIXED_RATE.value == "fixed_rate"
        assert ScheduleType.FIXED_DELAY.value == "fixed_delay"

    def test_schedule_type_from_string(self):
        """Test creating ScheduleType from string."""
        assert ScheduleType("fixed_rate") == ScheduleType.FIXED_RATE
        assert ScheduleType("fixed_delay") == ScheduleType.FIXED_DELAY
