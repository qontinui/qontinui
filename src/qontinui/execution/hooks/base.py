"""Base interface for execution hooks.

This module defines the abstract base class for all execution hooks.
"""

from abc import ABC, abstractmethod
from typing import Any

from ...config import Action


class ExecutionHook(ABC):
    """Abstract base class for execution hooks.

    Hooks can intercept execution at three points:
    - before_action: Called before action execution
    - after_action: Called after successful action execution
    - on_error: Called when action execution fails
    """

    @abstractmethod
    def before_action(self, action: Action, context: dict[str, Any]):
        """Called before action execution.

        Args:
            action: The action about to execute
            context: Current execution context
        """
        pass

    @abstractmethod
    def after_action(self, action: Action, context: dict[str, Any], result: dict[str, Any]):
        """Called after successful action execution.

        Args:
            action: The action that executed
            context: Current execution context
            result: Execution result
        """
        pass

    @abstractmethod
    def on_error(self, action: Action, context: dict[str, Any], error: Exception):
        """Called when action execution fails.

        Args:
            action: The action that failed
            context: Current execution context
            error: The exception that occurred
        """
        pass
