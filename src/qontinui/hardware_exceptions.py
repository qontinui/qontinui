"""Hardware and HAL exceptions.

This module contains exceptions for hardware operations including
screen capture, mouse, keyboard, and HAL layer errors.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .base_exceptions import QontinuiException


class HardwareException(QontinuiException):
    """Base exception for hardware/system errors."""

    pass


class ScreenCaptureException(HardwareException):
    """Raised when screen capture fails."""

    def __init__(self, reason: str, monitor: int | None = None, **kwargs) -> None:
        """Initialize with capture details."""
        message = "Screen capture failed"
        if monitor is not None:
            message += f" on monitor {monitor}"
        message += f": {reason}"

        super().__init__(
            message,
            error_code="CAPTURE_FAILED",
            context={"reason": reason, "monitor": monitor, **kwargs},
        )


class MouseOperationException(HardwareException):
    """Raised when mouse operation fails."""

    def __init__(self, operation: str, reason: str, **kwargs) -> None:
        """Initialize with operation details."""
        super().__init__(
            f"Mouse operation '{operation}' failed: {reason}",
            error_code="MOUSE_FAILED",
            context={"operation": operation, "reason": reason, **kwargs},
        )


class KeyboardOperationException(HardwareException):
    """Raised when keyboard operation fails."""

    def __init__(self, operation: str, reason: str, **kwargs) -> None:
        """Initialize with operation details."""
        super().__init__(
            f"Keyboard operation '{operation}' failed: {reason}",
            error_code="KEYBOARD_FAILED",
            context={"operation": operation, "reason": reason, **kwargs},
        )


class HALError(QontinuiException):
    """Base exception for HAL layer errors."""

    pass


class ScreenCaptureError(HALError):
    """Raised when screen capture operation fails."""

    def __init__(self, reason: str, monitor: int | None = None, **kwargs) -> None:
        """Initialize with capture details."""
        message = "Screen capture failed"
        if monitor is not None:
            message += f" on monitor {monitor}"
        message += f": {reason}"

        super().__init__(
            message,
            error_code="SCREEN_CAPTURE_FAILED",
            context={"reason": reason, "monitor": monitor, **kwargs},
        )


class InputControlError(HALError):
    """Raised when input control operation fails."""

    def __init__(self, operation: str, reason: str, **kwargs) -> None:
        """Initialize with operation details."""
        super().__init__(
            f"Input control operation '{operation}' failed: {reason}",
            error_code="INPUT_CONTROL_FAILED",
            context={"operation": operation, "reason": reason, **kwargs},
        )


@contextmanager
def hal_error_context(operation: str, **details: Any) -> Iterator[None]:
    """Context manager to add HAL operation context to exceptions.

    Usage:
        with hal_error_context("screen_capture", monitor=0):
            capture_screen()

    Args:
        operation: HAL operation being performed
        **details: Additional details about the operation

    Raises:
        HALError: Wraps exceptions with HAL context
    """
    try:
        yield
    except QontinuiException:
        raise
    except Exception as e:
        raise HALError(f"{operation} failed: {e}") from e
