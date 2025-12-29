"""Schedule configuration for state-aware scheduling.

Adapted from Brobot's StateAwareScheduler configuration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from qontinui_schemas.common import utc_now


class TriggerType(Enum):
    """Types of schedule triggers."""

    TIME_BASED = "time_based"  # Scheduled at specific times (cron)
    INTERVAL = "interval"  # Repeating at fixed intervals
    STATE_BASED = "state_based"  # Triggered by state changes
    MANUAL = "manual"  # Manual execution only


class CheckMode(Enum):
    """How states should be checked before execution.

    Adapted from Brobot's StateCheckConfiguration.CheckMode.
    """

    CHECK_ALL = "CHECK_ALL"
    """Check all required states, regardless of current active status.
    More thorough but potentially less efficient."""

    CHECK_INACTIVE_ONLY = "CHECK_INACTIVE_ONLY"
    """Only check states that are currently inactive.
    More efficient when some states are known to be active."""


class ScheduleType(Enum):
    """Type of schedule execution."""

    FIXED_RATE = "fixed_rate"  # Execute at fixed rate (scheduleAtFixedRate)
    FIXED_DELAY = "fixed_delay"  # Execute with fixed delay between runs


@dataclass
class ScheduleConfig:
    """Configuration for a scheduled process execution.

    Combines scheduling parameters with state-aware execution settings
    from Brobot's StateAwareScheduler.
    """

    # Identification (required fields first)
    id: str
    name: str
    process_id: str
    """ID of the process to execute on this schedule."""

    # Optional identification
    description: str = ""

    # Trigger configuration
    trigger_type: TriggerType = TriggerType.MANUAL
    schedule_type: ScheduleType = ScheduleType.FIXED_RATE

    # Time-based scheduling
    cron_expression: str | None = None
    """Cron expression for time-based scheduling (e.g., '0 9 * * *')."""

    start_time: datetime | None = None
    """When to start the schedule."""

    end_time: datetime | None = None
    """When to stop the schedule (optional)."""

    # Interval-based scheduling
    interval_seconds: int | None = None
    """Interval between executions in seconds."""

    initial_delay_seconds: int = 0
    """Initial delay before first execution."""

    # State requirements (from Brobot)
    required_states: list[str] = field(default_factory=list)
    """States that must be active before execution."""

    forbidden_states: list[str] = field(default_factory=list)
    """States that must NOT be active before execution."""

    # State checking behavior (from Brobot)
    check_mode: CheckMode = CheckMode.CHECK_INACTIVE_ONLY
    """How to check state requirements."""

    rebuild_on_mismatch: bool = True
    """Whether to rebuild states if requirements aren't met."""

    skip_if_states_missing: bool = False
    """Whether to skip execution if states can't be satisfied."""

    # Execution limits (from Brobot)
    max_iterations: int = -1
    """Maximum number of executions. -1 = unlimited."""

    timeout_seconds: int | None = None
    """Timeout for each execution."""

    # Execution control
    enabled: bool = True
    """Whether the schedule is active."""

    priority: int = 10
    """Execution priority (lower = higher priority)."""

    # Retry configuration
    max_retries: int = 3
    """Maximum retry attempts on failure."""

    retry_delay_seconds: int = 5
    """Delay between retry attempts."""

    # Metadata
    project_name: str = ""
    """Associated project name."""

    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    last_executed: datetime | None = None
    next_execution: datetime | None = None

    tags: list[str] = field(default_factory=list)
    """Custom tags for organization."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "processId": self.process_id,
            "triggerType": self.trigger_type.value,
            "scheduleType": self.schedule_type.value,
            "cronExpression": self.cron_expression,
            "startTime": self.start_time.isoformat() if self.start_time else None,
            "endTime": self.end_time.isoformat() if self.end_time else None,
            "intervalSeconds": self.interval_seconds,
            "initialDelaySeconds": self.initial_delay_seconds,
            "requiredStates": self.required_states,
            "forbiddenStates": self.forbidden_states,
            "checkMode": self.check_mode.value,
            "rebuildOnMismatch": self.rebuild_on_mismatch,
            "skipIfStatesMissing": self.skip_if_states_missing,
            "maxIterations": self.max_iterations,
            "timeoutSeconds": self.timeout_seconds,
            "enabled": self.enabled,
            "priority": self.priority,
            "maxRetries": self.max_retries,
            "retryDelaySeconds": self.retry_delay_seconds,
            "projectName": self.project_name,
            "createdAt": self.created_at.isoformat(),
            "lastModified": self.last_modified.isoformat(),
            "lastExecuted": (self.last_executed.isoformat() if self.last_executed else None),
            "nextExecution": (self.next_execution.isoformat() if self.next_execution else None),
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduleConfig":
        """Create from dictionary (JSON deserialization)."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            process_id=data["processId"],
            trigger_type=TriggerType(data.get("triggerType", "manual")),
            schedule_type=ScheduleType(data.get("scheduleType", "fixed_rate")),
            cron_expression=data.get("cronExpression"),
            start_time=(
                datetime.fromisoformat(data["startTime"]) if data.get("startTime") else None
            ),
            end_time=(datetime.fromisoformat(data["endTime"]) if data.get("endTime") else None),
            interval_seconds=data.get("intervalSeconds"),
            initial_delay_seconds=data.get("initialDelaySeconds", 0),
            required_states=data.get("requiredStates", []),
            forbidden_states=data.get("forbiddenStates", []),
            check_mode=CheckMode(data.get("checkMode", "CHECK_INACTIVE_ONLY")),
            rebuild_on_mismatch=data.get("rebuildOnMismatch", True),
            skip_if_states_missing=data.get("skipIfStatesMissing", False),
            max_iterations=data.get("maxIterations", -1),
            timeout_seconds=data.get("timeoutSeconds"),
            enabled=data.get("enabled", True),
            priority=data.get("priority", 10),
            max_retries=data.get("maxRetries", 3),
            retry_delay_seconds=data.get("retryDelaySeconds", 5),
            project_name=data.get("projectName", ""),
            created_at=(
                datetime.fromisoformat(data["createdAt"]) if data.get("createdAt") else utc_now()
            ),
            last_modified=(
                datetime.fromisoformat(data["lastModified"])
                if data.get("lastModified")
                else utc_now()
            ),
            last_executed=(
                datetime.fromisoformat(data["lastExecuted"]) if data.get("lastExecuted") else None
            ),
            next_execution=(
                datetime.fromisoformat(data["nextExecution"]) if data.get("nextExecution") else None
            ),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExecutionRecord:
    """Record of a schedule execution.

    Tracks execution history for monitoring and debugging.
    """

    id: str
    schedule_id: str
    process_id: str
    start_time: datetime
    end_time: datetime | None = None
    status: str = "running"  # running, success, failed, cancelled, timeout
    iteration_count: int = 0
    state_checks: list["StateCheckResult"] = field(default_factory=list)
    error_message: str | None = None
    error_details: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)

    def duration_seconds(self) -> float | None:
        """Calculate execution duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "scheduleId": self.schedule_id,
            "processId": self.process_id,
            "startTime": self.start_time.isoformat(),
            "endTime": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "iterationCount": self.iteration_count,
            "stateChecks": [check.to_dict() for check in self.state_checks],
            "errorMessage": self.error_message,
            "errorDetails": self.error_details,
            "results": self.results,
            "durationSeconds": self.duration_seconds(),
        }


@dataclass
class StateCheckResult:
    """Result of a state check before execution.

    Records whether state requirements were met.
    """

    timestamp: datetime
    required_states: list[str]
    forbidden_states: list[str]
    active_states: list[str]
    check_passed: bool
    check_mode: CheckMode
    states_rebuilt: bool = False
    rebuild_success: bool | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "requiredStates": self.required_states,
            "forbiddenStates": self.forbidden_states,
            "activeStates": self.active_states,
            "checkPassed": self.check_passed,
            "checkMode": self.check_mode.value,
            "statesRebuilt": self.states_rebuilt,
            "rebuildSuccess": self.rebuild_success,
            "errorMessage": self.error_message,
        }
