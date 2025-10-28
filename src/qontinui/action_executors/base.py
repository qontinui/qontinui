"""Base action executor for the command pattern refactoring.

This module provides the base class and execution context for all specialized
action executors, replacing the monolithic ActionExecutor god class.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..config.schema import Action
from ..exceptions import ActionExecutionError


@dataclass
class ExecutionContext:
    """Shared context for all action executors.

    This context provides access to all dependencies needed by action executors,
    including HAL components, configuration, state management, and event emission.
    """

    # Configuration access
    config: Any  # QontinuiConfig
    defaults: Any  # SimpleNamespace with timing defaults

    # HAL components (wrappers)
    mouse: Any  # Mouse wrapper
    keyboard: Any  # Keyboard wrapper
    screen: Any  # Screen wrapper
    time: Any  # TimeWrapper

    # Shared state
    last_find_location: Optional[tuple[int, int]]
    variable_context: Any  # VariableContext
    state_executor: Optional[Any]  # StateExecutor

    # Sub-executors for control flow and data operations
    control_flow_executor: Any  # ControlFlowExecutor
    data_operations_executor: Any  # DataOperationsExecutor

    # Workflow execution (for RUN_WORKFLOW action and navigation)
    workflow_executor: Optional[Any]  # Reference to main workflow executor
    execute_action: Callable[[Action], bool]  # Callback to execute nested actions

    # Event emission functions
    emit_event: Callable[[str, dict], None]
    emit_action_event: Callable[[str, str, bool, dict], None]
    emit_image_recognition_event: Callable[[dict], None]

    def update_last_find_location(self, location: Optional[tuple[int, int]]) -> None:
        """Update the last find location (shared state).

        Args:
            location: New location or None to clear
        """
        self.last_find_location = location


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
        self.context.emit_action_event(
            action_id=action.id or "unknown",
            action_type=action.type,
            success=True,
            data={"status": "started"}
        )

    def _emit_action_success(self, action: Action, data: dict | None = None) -> None:
        """Emit action success event.

        Args:
            action: Action that succeeded
            data: Optional additional data
        """
        self.context.emit_action_event(
            action_id=action.id or "unknown",
            action_type=action.type,
            success=True,
            data=data or {}
        )

    def _emit_action_failure(self, action: Action, error: str, data: dict | None = None) -> None:
        """Emit action failure event.

        Args:
            action: Action that failed
            error: Error message
            data: Optional additional data
        """
        failure_data = {"error": error}
        if data:
            failure_data.update(data)

        self.context.emit_action_event(
            action_id=action.id or "unknown",
            action_type=action.type,
            success=False,
            data=failure_data
        )

    def _get_default_timing(self, category: str, key: str, default: float = 0.0) -> float:
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
