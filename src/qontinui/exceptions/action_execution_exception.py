"""Action execution exception.

Exception thrown when action execution fails.
"""

from .qontinui_runtime_exception import QontinuiRuntimeException


class ActionExecutionError(QontinuiRuntimeException):
    """Exception thrown when action execution fails.

    Raised when an action cannot be executed successfully.
    """

    def __init__(
        self,
        message: str = "Action execution failed",
        cause: Exception | None = None,
        action_type: str | None = None,
    ):
        """Initialize action execution exception.

        Args:
            message: Error message
            cause: Underlying exception that caused this error
            action_type: Type of action that failed (if applicable)
        """
        super().__init__(message, cause)
        self.action_type = action_type
