"""Registry for custom functions.

This module provides registration and execution of custom Python functions
that can be called from CUSTOM_FUNCTION actions in workflows.

Functions can be registered via:
1. Decorator: @register_function("my_func")
2. Direct registration: register_function_code("my_func", code_string)
3. File loading: register_function_file("my_func", "/path/to/file.py")

Example:
    @register_function("calculate_total")
    def calculate_total(context: FunctionContext, items: list) -> dict:
        total = sum(item["price"] for item in items)
        return {"total": total}

    # In workflow:
    {
        "type": "CUSTOM_FUNCTION",
        "config": {
            "functionId": "calculate_total",
            "inputs": {"items": "${cart_items}"},
            "outputs": {"total": "order_total"}
        }
    }
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class FunctionContext:
    """Context passed to custom functions during execution.

    Provides access to workflow state and variables without exposing
    full execution internals.
    """

    variables: dict[str, Any] = field(default_factory=dict)
    workflow_state: dict[str, Any] = field(default_factory=dict)
    active_states: set[str] = field(default_factory=set)
    previous_result: dict[str, Any] | None = None


@dataclass
class RegisteredFunction:
    """Metadata and code for a registered custom function."""

    function_id: str
    name: str
    description: str
    code: str  # Python source code
    callable: Callable | None = None  # Compiled callable (if available)
    author: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    tags: list[str] = field(default_factory=list)

    # Execution metadata
    input_schema: dict[str, Any] | None = None  # Expected inputs
    output_schema: dict[str, Any] | None = None  # Expected outputs
    timeout: int = 30  # Default timeout in seconds


# Global registry mapping function IDs to RegisteredFunction
_function_registry: dict[str, RegisteredFunction] = {}


def register_function(
    function_id: str,
    name: str | None = None,
    description: str = "",
    tags: list[str] | None = None,
) -> Callable:
    """Decorator to register a Python function for use in workflows.

    Args:
        function_id: Unique identifier for the function
        name: Human-readable name (defaults to function_id)
        description: Description of what the function does
        tags: Optional tags for categorization

    Returns:
        Decorator function

    Example:
        @register_function("format_currency", description="Format number as currency")
        def format_currency(context: FunctionContext, amount: float, currency: str = "USD") -> str:
            return f"{currency} {amount:.2f}"
    """

    def decorator(func: Callable) -> Callable:
        # Get the function source code
        import inspect

        try:
            code = inspect.getsource(func)
        except (OSError, TypeError):
            # If source unavailable, store function name
            code = f"# Source unavailable for: {func.__name__}"

        registered = RegisteredFunction(
            function_id=function_id,
            name=name or function_id,
            description=description or func.__doc__ or "",
            code=code,
            callable=func,
            tags=tags or [],
        )

        _function_registry[function_id] = registered
        logger.info(f"Registered custom function: {function_id}")

        return func

    return decorator


def register_function_code(
    function_id: str,
    code: str,
    name: str | None = None,
    description: str = "",
    entry_point: str = "main",
    tags: list[str] | None = None,
) -> RegisteredFunction:
    """Register a function from Python source code.

    The code should define a function matching entry_point that accepts
    FunctionContext as first argument.

    Args:
        function_id: Unique identifier for the function
        code: Python source code
        name: Human-readable name
        description: Description of the function
        entry_point: Name of function to call in the code (default: "main")
        tags: Optional tags for categorization

    Returns:
        RegisteredFunction metadata

    Example:
        register_function_code(
            "my_func",
            '''
            def main(context, x, y):
                return {"sum": x + y}
            '''
        )
    """
    registered = RegisteredFunction(
        function_id=function_id,
        name=name or function_id,
        description=description,
        code=code,
        callable=None,  # Will be compiled on execution
        tags=tags or [],
    )

    # Store entry point in metadata
    if registered.input_schema is None:
        registered.input_schema = {}
    registered.input_schema["_entry_point"] = entry_point

    _function_registry[function_id] = registered
    logger.info(f"Registered custom function from code: {function_id}")

    return registered


def register_function_file(
    function_id: str,
    file_path: str,
    name: str | None = None,
    description: str = "",
    entry_point: str = "main",
    tags: list[str] | None = None,
) -> RegisteredFunction:
    """Register a function from a Python file.

    Args:
        function_id: Unique identifier for the function
        file_path: Path to Python file
        name: Human-readable name
        description: Description of the function
        entry_point: Name of function to call in the file
        tags: Optional tags for categorization

    Returns:
        RegisteredFunction metadata

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Function file not found: {file_path}")

    code = path.read_text()

    return register_function_code(
        function_id=function_id,
        code=code,
        name=name,
        description=description,
        entry_point=entry_point,
        tags=tags,
    )


def get_function(function_id: str) -> RegisteredFunction | None:
    """Get a registered function by ID.

    Args:
        function_id: Function identifier

    Returns:
        RegisteredFunction or None if not found
    """
    return _function_registry.get(function_id)


def get_all_functions() -> list[RegisteredFunction]:
    """Get all registered functions.

    Returns:
        List of all RegisteredFunction objects
    """
    return list(_function_registry.values())


def get_functions_by_tag(tag: str) -> list[RegisteredFunction]:
    """Get all functions with a specific tag.

    Args:
        tag: Tag to filter by

    Returns:
        List of functions with the tag
    """
    return [f for f in _function_registry.values() if tag in f.tags]


def unregister_function(function_id: str) -> bool:
    """Remove a function from the registry.

    Args:
        function_id: Function identifier

    Returns:
        True if function was removed, False if not found
    """
    if function_id in _function_registry:
        del _function_registry[function_id]
        logger.info(f"Unregistered custom function: {function_id}")
        return True
    return False


def clear_registry() -> None:
    """Clear all registered functions.

    Primarily useful for testing.
    """
    _function_registry.clear()


def execute_function(
    function_id: str,
    context: FunctionContext,
    inputs: dict[str, Any] | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Execute a registered function.

    Args:
        function_id: Function identifier
        context: Execution context
        inputs: Input parameters for the function
        timeout: Execution timeout in seconds (overrides function default)

    Returns:
        Function result as dictionary

    Raises:
        KeyError: If function not found
        TimeoutError: If execution exceeds timeout
        Exception: Any exception raised by the function
    """
    func = get_function(function_id)
    if func is None:
        raise KeyError(f"Function not found: {function_id}")

    inputs = inputs or {}
    exec_timeout = timeout or func.timeout

    # If we have a direct callable, use it
    if func.callable is not None:
        return _execute_callable(func.callable, context, inputs, exec_timeout)

    # Otherwise, compile and execute code
    return _execute_code(func, context, inputs, exec_timeout)


def _execute_callable(
    callable_func: Callable,
    context: FunctionContext,
    inputs: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    """Execute a callable function.

    Args:
        callable_func: Function to execute
        context: Execution context
        inputs: Input parameters
        timeout: Timeout in seconds

    Returns:
        Function result
    """
    import signal

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Function execution exceeded {timeout}s timeout")

    try:
        # Set timeout (Unix only)
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

        # Execute function
        result = callable_func(context, **inputs)

        # Cancel timeout
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

        # Ensure result is a dict
        if not isinstance(result, dict):
            result = {"result": result}

        return result

    except TimeoutError:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        raise

    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)


def _execute_code(
    func: RegisteredFunction,
    context: FunctionContext,
    inputs: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    """Execute function from source code.

    Args:
        func: RegisteredFunction with code
        context: Execution context
        inputs: Input parameters
        timeout: Timeout in seconds

    Returns:
        Function result
    """
    import signal

    # Get entry point
    entry_point = "main"
    if func.input_schema and "_entry_point" in func.input_schema:
        entry_point = func.input_schema["_entry_point"]

    # Restricted builtins
    builtins_dict = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
    restricted_builtins = {
        k: v
        for k, v in builtins_dict.items()
        if k
        not in {
            "eval",
            "exec",
            "compile",
            "open",
            "input",
            "help",
            "breakpoint",
            "exit",
            "quit",
        }
    }

    # Prepare execution namespace
    exec_globals: dict[str, Any] = {
        "__builtins__": restricted_builtins,
        "FunctionContext": FunctionContext,
    }
    exec_locals: dict[str, Any] = {}

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Function execution exceeded {timeout}s timeout")

    try:
        # Set timeout (Unix only)
        if hasattr(signal, "SIGALRM"):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

        # Compile and execute the code to define the function
        exec(func.code, exec_globals, exec_locals)

        # Get the entry point function
        if entry_point not in exec_locals:
            raise ValueError(f"Entry point '{entry_point}' not found in function code")

        callable_func = exec_locals[entry_point]

        # Call the function
        result = callable_func(context, **inputs)

        # Cancel timeout
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)

        # Ensure result is a dict
        if not isinstance(result, dict):
            result = {"result": result}

        return result

    except TimeoutError:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
        raise

    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
