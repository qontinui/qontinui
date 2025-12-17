"""Base action executor for the command pattern refactoring.

This module provides the base class and execution context for all specialized
action executors, replacing the monolithic ActionExecutor god class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config.schema import Action

if TYPE_CHECKING:
    from ..actions.action_result import ActionResult


@dataclass
class ExecutionContext:
    """Shared context for all action executors.

    This context provides access to all dependencies needed by action executors,
    including HAL components, configuration, state management, and event emission.

    The context stores complete ActionResult objects from executions, enabling
    subsequent actions to access all matches and execution details, not just
    a single location.
    """

    # Configuration access
    config: Any  # QontinuiConfig
    defaults: Any  # SimpleNamespace with timing defaults

    # HAL components (wrappers)
    mouse: Any  # Mouse wrapper
    keyboard: Any  # Keyboard wrapper
    screen: Any  # Screen wrapper
    time: Any  # TimeWrapper

    # Shared state - stores complete action results
    last_action_result: ActionResult | None
    variable_context: Any  # VariableContext
    state_executor: Any | None  # StateExecutor

    # Sub-executors for control flow and data operations
    control_flow_executor: Any  # ControlFlowExecutor
    data_operations_executor: Any  # DataOperationsExecutor

    # Workflow execution (for RUN_WORKFLOW action and navigation)
    workflow_executor: Any | None  # Reference to main workflow executor
    execute_action: Callable[[Action], bool]  # Callback to execute nested actions

    # Event emission functions
    emit_event: Callable[[str, dict], None]
    # emit_action_event signature: (action_type, action_id, success, data)
    emit_action_event: Callable[[str, str, bool, dict], None]
    emit_image_recognition_event: Callable[[dict], None]

    # Project configuration
    project_root: Path | None = field(default=None)  # Root directory for code loading

    # Multi-monitor support - tracks monitor offset for coordinate conversion
    # When automation targets a specific monitor, found coordinates need to be
    # adjusted to absolute screen coordinates for mouse operations
    monitor_index: int | None = field(default=None)  # Current target monitor (0-based)
    monitor_offset_x: int = field(
        default=0
    )  # Monitor's left position in virtual screen
    monitor_offset_y: int = field(default=0)  # Monitor's top position in virtual screen

    def update_last_action_result(self, result: ActionResult | None) -> None:
        """Store complete action result for subsequent actions to reference.

        This method stores the full ActionResult object, preserving all matches
        and execution details. Subsequent actions can access any match from the
        result, not just the first/best one.

        Pass None to clear the last result (e.g., when FIND fails, to prevent
        subsequent CLICK from using stale data).

        Args:
            result: Complete ActionResult with all matches and execution data,
                    or None to clear.

        Example:
            # FIND action stores result with multiple matches
            result = ActionResultBuilder()
                .add_match(match1)
                .add_match(match2)
                .with_success(True)
                .build()
            context.update_last_action_result(result)

            # Later, CLICK action can access any match:
            # - result.matches[0] for best match
            # - result.matches[1] for second match
            # - result.matches for all matches

            # When FIND fails, clear the result:
            context.update_last_action_result(None)
        """
        self.last_action_result = result


class ActionExecutorBase(ABC):
    """Base class for all specialized action executors.

    Each executor handles one or more related action types (e.g., mouse actions,
    keyboard actions). Executors are focused classes that follow the Single
    Responsibility Principle.

    Attributes:
        context: Shared execution context with HAL, config, and state
    """

    def __init__(self, context: ExecutionContext) -> None:
        """Initialize with shared execution context.

        Args:
            context: Execution context containing all dependencies
        """
        self.context = context

    @abstractmethod
    def execute(self, action: Action, typed_config: Any) -> bool:
        """Execute the action with validated configuration.

        This method contains the core logic for executing a specific action type.
        It should use the context to access HAL components and emit events.

        Args:
            action: Pydantic Action model with type, config, etc.
            typed_config: Type-specific validated configuration object

        Returns:
            bool: True if action succeeded, False otherwise

        Raises:
            ActionExecutionError: If action execution fails critically
        """
        pass

    @abstractmethod
    def get_supported_action_types(self) -> list[str]:
        """Get list of action types this executor handles.

        Returns:
            List of action type strings (e.g., ["CLICK", "DOUBLE_CLICK"])
        """
        pass

    def handles_action_type(self, action_type: str) -> bool:
        """Check if this executor handles the given action type.

        Args:
            action_type: Action type string (e.g., "CLICK")

        Returns:
            bool: True if this executor handles the action type
        """
        return action_type in self.get_supported_action_types()

    # Helper methods for common operations

    def _emit_action_start(self, action: Action) -> None:
        """Emit action start event.

        Args:
            action: Action being executed
        """
        # Note: emit_action_event signature is (action_type, action_id, success, data)
        self.context.emit_action_event(
            action.type,
            action.id or "unknown",
            True,
            {"status": "started"},
        )

    def _emit_action_success(self, action: Action, data: dict | None = None) -> None:
        """Emit action success event.

        Args:
            action: Action that succeeded
            data: Optional additional data
        """
        # Note: emit_action_event signature is (action_type, action_id, success, data)
        self.context.emit_action_event(
            action.type, action.id or "unknown", True, data or {}
        )

    def _emit_action_failure(
        self, action: Action, error: str, data: dict | None = None
    ) -> None:
        """Emit action failure event.

        Args:
            action: Action that failed
            error: Error message
            data: Optional additional data
        """
        failure_data = {"error": error}
        if data:
            failure_data.update(data)

        # Note: emit_action_event signature is (action_type, action_id, success, data)
        self.context.emit_action_event(
            action.type,
            action.id or "unknown",
            False,
            failure_data,
        )

    def _get_default_timing(
        self, category: str, key: str, default: float = 0.0
    ) -> float:
        """Get default timing value from configuration.

        Args:
            category: Category name (e.g., "mouse", "keyboard")
            key: Timing key (e.g., "click_hold_duration")
            default: Default value if not found

        Returns:
            Timing value in seconds
        """
        try:
            category_obj = getattr(self.context.defaults, category, None)
            if category_obj:
                return getattr(category_obj, key, default)
            return default
        except (AttributeError, TypeError):
            return default
