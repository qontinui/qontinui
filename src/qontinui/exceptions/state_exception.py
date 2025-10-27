"""State exception.

Exception thrown when state operations fail.
"""

from .qontinui_runtime_exception import QontinuiRuntimeException


class StateException(QontinuiRuntimeException):
    """Exception thrown when state operations fail.

    Raised during state transitions, lookups, or validations.
    """

    def __init__(
        self,
        message: str = "State operation failed",
        cause: Exception | None = None,
        state_id: str | None = None,
    ):
        """Initialize state exception.

        Args:
            message: Error message
            cause: Underlying exception that caused this error
            state_id: ID of the state that caused the error (if applicable)
        """
        super().__init__(message, cause)
        self.state_id = state_id
