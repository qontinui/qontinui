"""Registry for action executors using command pattern.

This module provides the registration and lookup mechanism for specialized
action executors. Each executor registers the action types it handles.
"""

from typing import Type

from ..exceptions import ActionExecutionError
from .base import ActionExecutorBase, ExecutionContext

# Global registry mapping action types to executor classes
_executor_registry: dict[str, Type[ActionExecutorBase]] = {}


def register_executor(executor_class: Type[ActionExecutorBase]) -> Type[ActionExecutorBase]:
    """Register an executor class for its supported action types.

    This can be used as a decorator on executor classes:

    @register_executor
    class ClickActionExecutor(ActionExecutorBase):
        ...

    Args:
        executor_class: Executor class to register

    Returns:
        The same executor class (for use as decorator)

    Raises:
        ValueError: If an action type is already registered to a different executor
    """
    # Get action types from a temporary instance
    # (We need an instance to call get_supported_action_types, but we pass None
    # for context since we're only querying metadata)
    try:
        # Create a minimal context for registration purposes
        temp_instance = executor_class.__new__(executor_class)
        action_types = temp_instance.get_supported_action_types()
    except Exception as e:
        raise ValueError(
            f"Failed to get supported action types from {executor_class.__name__}: {e}"
        ) from e

    # Register each action type
    for action_type in action_types:
        if action_type in _executor_registry:
            existing_class = _executor_registry[action_type]
            if existing_class != executor_class:
                raise ValueError(
                    f"Action type '{action_type}' is already registered to "
                    f"{existing_class.__name__}, cannot register to {executor_class.__name__}"
                )

        _executor_registry[action_type] = executor_class

    return executor_class


def get_executor_class(action_type: str) -> Type[ActionExecutorBase] | None:
    """Get the executor class for an action type.

    Args:
        action_type: Action type string (e.g., "CLICK")

    Returns:
        Executor class or None if not found
    """
    return _executor_registry.get(action_type)


def create_executor(action_type: str, context: ExecutionContext) -> ActionExecutorBase:
    """Create an executor instance for an action type.

    Args:
        action_type: Action type string (e.g., "CLICK")
        context: Execution context to pass to executor

    Returns:
        Executor instance ready to execute actions

    Raises:
        ActionExecutionError: If no executor is registered for the action type
    """
    executor_class = get_executor_class(action_type)

    if executor_class is None:
        raise ActionExecutionError(
            f"No executor registered for action type: {action_type}",
            {"action_type": action_type}
        )

    return executor_class(context)


def get_registered_action_types() -> list[str]:
    """Get all registered action types.

    Returns:
        List of action type strings
    """
    return list(_executor_registry.keys())


def clear_registry() -> None:
    """Clear the executor registry.

    This is primarily useful for testing.
    """
    _executor_registry.clear()
