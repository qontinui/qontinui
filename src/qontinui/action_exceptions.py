"""Action execution exceptions.

This module contains exceptions for action-related errors including
execution failures, timeouts, and parameter validation.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .base_exceptions import QontinuiException


class ActionException(QontinuiException):
    """Base exception for action-related errors."""

    pass


class ActionFailedException(ActionException):
    """Raised when an action fails to execute."""

    def __init__(self, action_type: str, reason: str | None = None, **kwargs) -> None:
        """Initialize with action details."""
        message = f"Action '{action_type}' failed"
        if reason:
            message += f": {reason}"

        super().__init__(
            message,
            error_code="ACTION_FAILED",
            context={"action_type": action_type, "reason": reason, **kwargs},
        )


class ActionTimeoutException(ActionException):
    """Raised when an action times out."""

    def __init__(self, action_type: str, timeout: float, **kwargs) -> None:
        """Initialize with timeout details."""
        super().__init__(
            f"Action '{action_type}' timed out after {timeout} seconds",
            error_code="ACTION_TIMEOUT",
            context={"action_type": action_type, "timeout": timeout, **kwargs},
        )


class ActionNotRegisteredException(ActionException):
    """Raised when trying to use an unregistered action."""

    def __init__(self, action_name: str, **kwargs) -> None:
        """Initialize with action name."""
        super().__init__(
            f"Action '{action_name}' is not registered",
            error_code="ACTION_NOT_REGISTERED",
            context={"action_name": action_name, **kwargs},
        )


class InvalidActionParametersException(ActionException):
    """Raised when action parameters are invalid."""

    def __init__(self, action_type: str, parameter: str, reason: str, **kwargs) -> None:
        """Initialize with parameter details."""
        super().__init__(
            f"Invalid parameter '{parameter}' for action '{action_type}': {reason}",
            error_code="INVALID_PARAMETERS",
            context={
                "action_type": action_type,
                "parameter": parameter,
                "reason": reason,
                **kwargs,
            },
        )


class ActionExecutionError(QontinuiException):
    """Raised when action execution fails."""

    def __init__(self, action_type: str, reason: str, **kwargs) -> None:
        """Initialize with action execution details."""
        super().__init__(
            f"Failed to execute {action_type} action: {reason}",
            error_code="ACTION_EXECUTION_FAILED",
            context={"action_type": action_type, "reason": reason, **kwargs},
        )


@contextmanager
def action_error_context(action_type: str, **action_details: Any) -> Iterator[None]:
    """Context manager to add action context to exceptions.

    Usage:
        with action_error_context("click", x=100, y=200):
            perform_click(100, 200)

    Args:
        action_type: Type of action being performed
        **action_details: Additional details about the action

    Raises:
        ActionExecutionError: Wraps any QontinuiException with action context
    """
    try:
        yield
    except QontinuiException as e:
        raise ActionExecutionError(
            action_type=action_type, reason=str(e), **action_details
        ) from e
    except Exception as e:
        raise ActionExecutionError(
            action_type=action_type, reason=f"Unexpected error: {e}", **action_details
        ) from e
