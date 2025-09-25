"""Qontinui runtime exception - ported from Qontinui framework.

Base exception for the framework.
"""


class QontinuiRuntimeException(RuntimeError):
    """Base runtime exception for all Qontinui framework exceptions.

    Port of QontinuiRuntimeException from Qontinui framework class.

    This is the root of the Qontinui exception hierarchy, providing a common base
    for all framework-specific runtime exceptions. Using runtime exceptions allows
    the framework to propagate errors up through multiple layers without forcing
    intermediate code to handle them, enabling centralized error handling at
    appropriate orchestration points.
    """

    def __init__(self, message: str | None = None, cause: Exception | None = None):
        """Construct a new runtime exception.

        Args:
            message: The detail message
            cause: The cause of the exception
        """
        if message and cause:
            super().__init__(f"{message}: {cause}")
            self.__cause__ = cause
        elif message:
            super().__init__(message)
        elif cause:
            super().__init__(str(cause))
            self.__cause__ = cause
        else:
            super().__init__()
