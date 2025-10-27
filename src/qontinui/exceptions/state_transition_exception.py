"""State transition exception.

Exception thrown when state transitions fail.
"""

from .state_exception import StateException


class StateTransitionException(StateException):
    """Exception thrown when state transitions fail.

    Raised when attempting an invalid state transition.
    """

    def __init__(
        self,
        message: str = "State transition failed",
        cause: Exception | None = None,
        from_state: str | None = None,
        to_state: str | None = None,
    ):
        """Initialize state transition exception.

        Args:
            message: Error message
            cause: Underlying exception that caused this error
            from_state: Source state ID
            to_state: Target state ID
        """
        super().__init__(message, cause)
        self.from_state = from_state
        self.to_state = to_state
