"""Action executors using command pattern.

This package replaces the monolithic ActionExecutor god class with focused,
specialized executor classes. Each executor handles a specific group of
related actions (e.g., mouse actions, keyboard actions).

Architecture:
    - ActionExecutorBase: Abstract base class for all executors
    - ExecutionContext: Shared dependencies and state
    - Registry: Maps action types to executor classes
    - Specialized executors: One per action type group
    - DelegatingActionExecutor: Main executor that delegates to specialized executors

Usage:
    # High-level usage (recommended for most cases)
    from qontinui.action_executors import DelegatingActionExecutor

    # Create executor with config
    executor = DelegatingActionExecutor(config)

    # Execute action (internally delegates to specialized executors)
    success = executor.execute_action(action)

    # Low-level usage (for direct access to specialized executors)
    from qontinui.action_executors import ExecutionContext, create_executor

    # Create context with dependencies
    context = ExecutionContext(
        config=config,
        mouse=mouse_wrapper,
        ...
    )

    # Get executor for action type
    executor = create_executor("CLICK", context)

    # Execute action
    success = executor.execute(action, typed_config)
"""

# Import executor modules to trigger @register_executor decorator
from . import (
    code_executor,  # noqa: F401
    control_flow,  # noqa: F401
    data_operations,  # noqa: F401
    keyboard,  # noqa: F401
    mouse,  # noqa: F401
    navigation,  # noqa: F401
    shell,  # noqa: F401
    utility,  # noqa: F401
    vision,  # noqa: F401
)
from .base import ActionExecutorBase, ExecutionContext
from .delegating_executor import DelegatingActionExecutor
from .registry import (
    clear_registry,
    create_executor,
    get_executor_class,
    get_registered_action_types,
    register_executor,
)

__all__ = [
    # Main executor
    "DelegatingActionExecutor",
    # Base classes
    "ActionExecutorBase",
    "ExecutionContext",
    # Registry functions
    "register_executor",
    "create_executor",
    "get_executor_class",
    "get_registered_action_types",
    "clear_registry",
]
