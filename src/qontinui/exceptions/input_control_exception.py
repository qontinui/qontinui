"""Input control exception.

Exception thrown when input control operations fail.
"""

from .qontinui_runtime_exception import QontinuiRuntimeException


class InputControlError(QontinuiRuntimeException):
    """Exception thrown when input control operations fail.

    Raised during mouse, keyboard, or other input control operations.
    """

    def __init__(
        self,
        message: str = "Input control failed",
        cause: Exception | None = None,
        operation: str | None = None,
    ):
        """Initialize input control exception.

        Args:
            message: Error message
            cause: Underlying exception that caused this error
            operation: Operation that failed (e.g., 'mouse_move', 'key_press')
        """
        super().__init__(message, cause)
        self.operation = operation
