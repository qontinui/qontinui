"""State and navigation exceptions.

This module contains exceptions for state management, transitions,
and navigation operations.
"""

from .base_exceptions import QontinuiException


class StateException(QontinuiException):
    """Base exception for state-related errors."""

    pass


class StateNotFoundException(StateException):
    """Raised when a state cannot be found."""

    def __init__(self, state_name: str, **kwargs) -> None:
        """Initialize with state name."""
        super().__init__(
            f"State '{state_name}' not found",
            error_code="STATE_NOT_FOUND",
            context={"state_name": state_name, **kwargs},
        )


class StateTransitionException(StateException):
    """Raised when state transition fails."""

    def __init__(self, from_state: str, to_state: str, reason: str | None = None, **kwargs) -> None:
        """Initialize with transition details."""
        message = f"Failed to transition from '{from_state}' to '{to_state}'"
        if reason:
            message += f": {reason}"

        super().__init__(
            message,
            error_code="TRANSITION_FAILED",
            context={
                "from_state": from_state,
                "to_state": to_state,
                "reason": reason,
                **kwargs,
            },
        )


class StateAlreadyExistsException(StateException):
    """Raised when attempting to create a duplicate state."""

    def __init__(self, state_name: str, **kwargs) -> None:
        """Initialize with state name."""
        super().__init__(
            f"State '{state_name}' already exists",
            error_code="STATE_EXISTS",
            context={"state_name": state_name, **kwargs},
        )


class InvalidStateException(StateException):
    """Raised when a state is invalid or corrupted."""

    def __init__(self, state_name: str, reason: str, **kwargs) -> None:
        """Initialize with state details."""
        super().__init__(
            f"State '{state_name}' is invalid: {reason}",
            error_code="INVALID_STATE",
            context={"state_name": state_name, "reason": reason, **kwargs},
        )
