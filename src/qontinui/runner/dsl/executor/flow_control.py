"""Flow control exceptions for DSL execution.

Provides exception-based control flow for break, continue, and return statements.
"""

from typing import Any


class FlowControlException(Exception):
    """Base class for flow control exceptions.

    Flow control exceptions are used to implement control flow constructs
    (break, continue, return) in the DSL executor. These are not errors,
    but rather mechanisms to alter the normal sequential flow of execution.

    These exceptions should be caught and handled by the appropriate
    statement executors (loops for break/continue, functions for return).
    """

    pass


class BreakException(FlowControlException):
    """Exception raised to break out of a loop.

    When a break statement is executed within a loop (forEach), this exception
    is raised to immediately exit the innermost enclosing loop. The loop executor
    catches this exception and terminates the loop iteration.

    Example:
        ```python
        # In a forEach loop:
        for item in items:
            if item == "stop":
                raise BreakException()  # Exit the loop
            process(item)
        ```

    Note:
        This exception should only be raised within a loop context. Raising it
        outside a loop is a runtime error that will be caught by the executor.
    """

    def __init__(self, message: str = "Break statement executed"):
        """Initialize break exception.

        Args:
            message: Optional message describing the break
        """
        super().__init__(message)


class ContinueException(FlowControlException):
    """Exception raised to continue to the next iteration of a loop.

    When a continue statement is executed within a loop (forEach), this exception
    is raised to skip the rest of the current iteration and proceed to the next
    one. The loop executor catches this exception and moves to the next element.

    Example:
        ```python
        # In a forEach loop:
        for item in items:
            if item.skip:
                raise ContinueException()  # Skip to next item
            process(item)
        ```

    Note:
        This exception should only be raised within a loop context. Raising it
        outside a loop is a runtime error that will be caught by the executor.
    """

    def __init__(self, message: str = "Continue statement executed"):
        """Initialize continue exception.

        Args:
            message: Optional message describing the continue
        """
        super().__init__(message)


class ReturnException(FlowControlException):
    """Exception raised to return from a function.

    When a return statement is executed, this exception is raised to immediately
    exit the current function and return a value to the caller. The function
    executor catches this exception and returns the contained value.

    Example:
        ```python
        # In a function:
        def calculate(x):
            if x < 0:
                raise ReturnException(None)  # Early return
            result = x * 2
            raise ReturnException(result)  # Normal return
        ```

    Attributes:
        value: The value to return from the function (can be None for void returns)
    """

    def __init__(self, value: Any = None):
        """Initialize return exception.

        Args:
            value: The value to return from the function
        """
        super().__init__(f"Return statement executed with value: {value}")
        self.value = value


class ExecutionError(Exception):
    """Exception raised for errors during DSL execution.

    This exception is raised when an error occurs during the execution of
    DSL statements or expressions. It provides context about where the error
    occurred and what went wrong.

    Attributes:
        message: Description of the error
        statement_type: Type of statement that caused the error (if applicable)
        expression_type: Type of expression that caused the error (if applicable)
        context: Additional context information
    """

    def __init__(
        self,
        message: str,
        statement_type: str | None = None,
        expression_type: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize execution error.

        Args:
            message: Description of the error
            statement_type: Type of statement that caused the error
            expression_type: Type of expression that caused the error
            context: Additional context information
        """
        self.message = message
        self.statement_type = statement_type
        self.expression_type = expression_type
        self.context = context or {}

        # Build detailed error message
        parts = [message]
        if statement_type:
            parts.append(f"Statement type: {statement_type}")
        if expression_type:
            parts.append(f"Expression type: {expression_type}")
        if context:
            parts.append(f"Context: {context}")

        super().__init__(" | ".join(parts))
