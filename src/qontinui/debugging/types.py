"""Type definitions and enums for the debugging system.

This module defines the core types and enumerations used throughout
the debugging subsystem.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, cast


class ExecutionState(Enum):
    """Execution state of a debug session."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STEPPING = "stepping"
    COMPLETED = "completed"
    ERROR = "error"


class BreakpointType(Enum):
    """Types of breakpoints supported by the debugging system."""

    ACTION_ID = "action_id"  # Break on specific action ID
    ACTION_TYPE = "action_type"  # Break on action type (e.g., Click, Find)
    CONDITIONAL = "conditional"  # Break when condition evaluates to True
    ERROR = "error"  # Break on any error
    MATCH_COUNT = "match_count"  # Break when match count meets condition
    STATE_CHANGE = "state_change"  # Break when state changes


class StepMode(Enum):
    """Step execution modes."""

    INTO = "into"  # Step into nested actions
    OVER = "over"  # Step over nested actions
    OUT = "out"  # Step out of current action


@dataclass
class Breakpoint:
    """Represents a breakpoint in the debugging system."""

    id: str
    type: BreakpointType
    enabled: bool = True
    hit_count: int = 0

    # Type-specific fields
    action_id: str | None = None
    action_type: str | None = None
    condition: Callable[[Any], bool] | None = None
    condition_str: str | None = None  # String representation for display

    created_at: datetime = field(default_factory=datetime.now)

    def should_break(self, context: dict[str, Any]) -> bool:
        """Determine if this breakpoint should trigger.

        Args:
            context: Current execution context containing action info

        Returns:
            True if breakpoint conditions are met
        """
        if not self.enabled:
            return False

        if self.type == BreakpointType.ACTION_ID:
            return context.get("action_id") == self.action_id

        elif self.type == BreakpointType.ACTION_TYPE:
            return context.get("action_type") == self.action_type

        elif self.type == BreakpointType.CONDITIONAL:
            if self.condition:
                try:
                    return self.condition(context)
                except (ValueError, TypeError, AttributeError, KeyError, RuntimeError):
                    # Condition evaluation failed, treat as not breaking
                    return False
            return False

        elif self.type == BreakpointType.ERROR:
            return cast(bool, context.get("has_error", False))

        elif self.type == BreakpointType.MATCH_COUNT:
            if self.condition:
                try:
                    return self.condition(context)
                except (ValueError, TypeError, AttributeError, KeyError, RuntimeError):
                    # Condition evaluation failed, treat as not breaking
                    return False
            return False

        elif self.type == BreakpointType.STATE_CHANGE:
            return cast(bool, context.get("state_changed", False))

        return False


@dataclass
class ExecutionRecord:
    """Record of a single action execution."""

    timestamp: datetime
    action_id: str
    action_type: str
    action_description: str
    success: bool
    duration_ms: float

    # Snapshot data
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    # Match information
    match_count: int = 0
    matches: list[dict[str, Any]] = field(default_factory=list)

    # Context
    session_id: str = ""
    parent_action_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "action_id": self.action_id,
            "action_type": self.action_type,
            "action_description": self.action_description,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error_message": self.error_message,
            "match_count": self.match_count,
            "matches": self.matches,
            "session_id": self.session_id,
            "parent_action_id": self.parent_action_id,
        }


@dataclass
class VariableSnapshot:
    """Snapshot of variables at a specific point in execution."""

    timestamp: datetime
    action_id: str
    variables: dict[str, Any] = field(default_factory=dict)

    def get(self, name: str, default: Any = None) -> Any:
        """Get variable value from snapshot.

        Args:
            name: Variable name
            default: Default value if not found

        Returns:
            Variable value or default
        """
        return self.variables.get(name, default)


@dataclass
class DebugHookContext:
    """Context passed to debug hooks."""

    session_id: str
    action_id: str
    action_type: str
    action_description: str
    action_config: Any
    object_collections: tuple[Any, ...]

    # Result data (for post-action hooks)
    result: Any | None = None
    error: Exception | None = None

    # Additional context
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "session_id": self.session_id,
            "action_id": self.action_id,
            "action_type": self.action_type,
            "action_description": self.action_description,
            "has_error": self.error is not None,
            "success": self.result.success if self.result else False,
            "match_count": (
                len(self.result.match_list)
                if self.result and hasattr(self.result, "match_list")
                else 0
            ),
            **self.extra,
        }


# Type aliases for hook functions
PreActionHook = Callable[[DebugHookContext], None]
PostActionHook = Callable[[DebugHookContext], None]
ErrorHook = Callable[[DebugHookContext], None]
