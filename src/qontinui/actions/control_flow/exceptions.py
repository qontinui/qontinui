"""Control flow exceptions for loop control.

This module defines custom exceptions used to control loop execution flow,
including breaking out of loops and continuing to the next iteration.
"""


class BreakLoop(Exception):
    """Exception raised to break out of a loop.

    This exception is used to signal that a loop should be terminated early,
    similar to a 'break' statement in traditional programming. When raised
    during loop execution, the loop will exit immediately and control will
    return to the code following the loop.

    Attributes:
        message: A descriptive message explaining why the loop was broken

    Example:
        >>> if error_condition:
        ...     raise BreakLoop("Error detected, stopping loop")
    """

    def __init__(self, message: str = "Loop break triggered") -> None:
        """Initialize the BreakLoop exception.

        Args:
            message: A descriptive message explaining the break reason.
                    Defaults to "Loop break triggered".
        """
        self.message = message
        super().__init__(self.message)


class ContinueLoop(Exception):
    """Exception raised to continue to the next loop iteration.

    This exception is used to signal that the current loop iteration should
    be skipped and execution should continue with the next iteration, similar
    to a 'continue' statement in traditional programming. Any remaining actions
    in the current iteration will be skipped.

    Attributes:
        message: A descriptive message explaining why the iteration was skipped

    Example:
        >>> if skip_condition:
        ...     raise ContinueLoop("Skipping invalid item")
    """

    def __init__(self, message: str = "Loop continue triggered") -> None:
        """Initialize the ContinueLoop exception.

        Args:
            message: A descriptive message explaining the continue reason.
                    Defaults to "Loop continue triggered".
        """
        self.message = message
        super().__init__(self.message)
