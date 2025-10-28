"""Execution context for DSL statement execution.

Manages variable scopes and provides variable storage during DSL execution.
"""

from typing import Any


class ExecutionContext:
    """Execution context for DSL statement execution.

    The execution context manages variable scopes during DSL execution. It supports
    nested scopes for control structures like loops and functions, allowing variables
    to be declared in inner scopes that shadow outer scope variables.

    The context uses a stack of scope dictionaries. When a new scope is pushed,
    variables can be declared in that scope. When the scope is popped, those
    variables are removed. Variable lookups search from the innermost (top) scope
    outward to the outermost (bottom) scope.

    Example:
        ```python
        context = ExecutionContext()

        # Global scope
        context.set_variable("x", 10)
        print(context.get_variable("x"))  # 10

        # Push new scope for a loop
        context.push_scope()
        context.set_variable("y", 20)  # Local to this scope
        context.set_variable("x", 15)  # Shadows outer x
        print(context.get_variable("x"))  # 15
        print(context.get_variable("y"))  # 20

        # Pop scope
        context.pop_scope()
        print(context.get_variable("x"))  # 10 (outer x restored)
        print(context.get_variable("y"))  # Raises KeyError
        ```

    Attributes:
        scopes: Stack of scope dictionaries, with the global scope at index 0
        _external_context: Optional external context for method calls and objects
    """

    def __init__(self, external_context: dict[str, Any] | None = None) -> None:
        """Initialize execution context.

        Args:
            external_context: Optional external context for method calls and objects.
                            This can contain objects, services, or other resources
                            that DSL code can interact with.
        """
        self.scopes: list[dict[str, Any]] = [{}]  # Start with global scope
        self._external_context = external_context or {}

    def push_scope(self) -> None:
        """Push a new scope onto the stack.

        Creates a new empty scope and pushes it onto the scope stack. Variables
        declared after this call will be in the new scope and will shadow any
        variables with the same name in outer scopes.

        This should be called when entering a new block that requires its own
        scope, such as:
        - forEach loops (for the loop variable)
        - If statement blocks
        - Function bodies

        Example:
            ```python
            context.set_variable("x", 1)
            context.push_scope()
            context.set_variable("x", 2)  # Shadows outer x
            print(context.get_variable("x"))  # 2
            context.pop_scope()
            print(context.get_variable("x"))  # 1
            ```
        """
        self.scopes.append({})

    def pop_scope(self) -> dict[str, Any]:
        """Pop the current scope from the stack.

        Removes the topmost scope from the stack and returns it. All variables
        declared in this scope are removed. The previous scope becomes active.

        Returns:
            The scope dictionary that was popped

        Raises:
            RuntimeError: If attempting to pop the global scope (last scope)

        Example:
            ```python
            context.push_scope()
            context.set_variable("temp", 42)
            scope = context.pop_scope()
            print(scope)  # {"temp": 42}
            # "temp" is no longer accessible
            ```
        """
        if len(self.scopes) <= 1:
            raise RuntimeError("Cannot pop global scope")
        return self.scopes.pop()

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the current scope.

        Creates or updates a variable in the current (topmost) scope. This will
        shadow any variable with the same name in outer scopes.

        Args:
            name: Variable name
            value: Variable value

        Example:
            ```python
            context.set_variable("count", 0)
            context.set_variable("message", "Hello")
            ```
        """
        self.scopes[-1][name] = value

    def update_variable(self, name: str, value: Any) -> None:
        """Update an existing variable in any scope.

        Searches for the variable from the innermost scope outward and updates
        the first occurrence found. This allows assignment statements to modify
        variables declared in outer scopes.

        Args:
            name: Variable name
            value: New variable value

        Raises:
            KeyError: If variable not found in any scope

        Example:
            ```python
            context.set_variable("x", 10)  # Global scope
            context.push_scope()
            context.update_variable("x", 20)  # Updates global x
            context.pop_scope()
            print(context.get_variable("x"))  # 20
            ```
        """
        # Search from innermost to outermost scope
        for scope in reversed(self.scopes):
            if name in scope:
                scope[name] = value
                return

        raise KeyError(f"Variable '{name}' not found in any scope")

    def get_variable(self, name: str) -> Any:
        """Get a variable value from the context.

        Searches for the variable from the innermost scope outward and returns
        the first occurrence found. This implements proper variable shadowing.

        Args:
            name: Variable name

        Returns:
            Variable value

        Raises:
            KeyError: If variable not found in any scope

        Example:
            ```python
            context.set_variable("x", 10)
            context.push_scope()
            context.set_variable("x", 20)  # Shadows outer x
            print(context.get_variable("x"))  # 20
            context.pop_scope()
            print(context.get_variable("x"))  # 10
            ```
        """
        # Search from innermost to outermost scope
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]

        raise KeyError(f"Variable '{name}' not found in context")

    def has_variable(self, name: str) -> bool:
        """Check if a variable exists in any scope.

        Args:
            name: Variable name

        Returns:
            True if variable exists, False otherwise

        Example:
            ```python
            context.set_variable("x", 10)
            print(context.has_variable("x"))  # True
            print(context.has_variable("y"))  # False
            ```
        """
        for scope in reversed(self.scopes):
            if name in scope:
                return True
        return False

    def delete_variable(self, name: str) -> None:
        """Delete a variable from the context.

        Searches for the variable from the innermost scope outward and deletes
        the first occurrence found.

        Args:
            name: Variable name

        Raises:
            KeyError: If variable not found in any scope

        Example:
            ```python
            context.set_variable("x", 10)
            context.delete_variable("x")
            # context.get_variable("x") raises KeyError
            ```
        """
        for scope in reversed(self.scopes):
            if name in scope:
                del scope[name]
                return

        raise KeyError(f"Variable '{name}' not found in context")

    def get_all_variables(self) -> dict[str, Any]:
        """Get all variables from all scopes merged.

        Returns a dictionary containing all variables from all scopes, with
        inner scope variables shadowing outer scope variables.

        Returns:
            Dictionary of all variables

        Example:
            ```python
            context.set_variable("x", 10)
            context.set_variable("y", 20)
            context.push_scope()
            context.set_variable("x", 15)  # Shadows outer x
            context.set_variable("z", 30)

            all_vars = context.get_all_variables()
            # {"x": 15, "y": 20, "z": 30}
            ```
        """
        result: dict[str, Any] = {}
        # Merge from outermost to innermost so inner scopes override
        for scope in self.scopes:
            result.update(scope)
        return result

    def get_current_scope(self) -> dict[str, Any]:
        """Get the current (topmost) scope.

        Returns:
            The current scope dictionary

        Example:
            ```python
            context.push_scope()
            context.set_variable("x", 10)
            scope = context.get_current_scope()
            print(scope)  # {"x": 10}
            ```
        """
        return self.scopes[-1]

    def get_external_context(self) -> dict[str, Any]:
        """Get the external context.

        The external context contains objects, services, and other resources
        that DSL code can interact with (e.g., logger, database, API clients).

        Returns:
            The external context dictionary

        Example:
            ```python
            context = ExecutionContext({"logger": logger_instance})
            logger = context.get_external_context()["logger"]
            ```
        """
        return self._external_context

    def set_external_object(self, name: str, obj: Any) -> None:
        """Add an object to the external context.

        Args:
            name: Object name
            obj: Object instance

        Example:
            ```python
            context.set_external_object("database", db_instance)
            # DSL code can now call methods on "database"
            ```
        """
        self._external_context[name] = obj

    def get_external_object(self, name: str) -> Any:
        """Get an object from the external context.

        Args:
            name: Object name

        Returns:
            Object instance

        Raises:
            KeyError: If object not found in external context

        Example:
            ```python
            db = context.get_external_object("database")
            ```
        """
        if name not in self._external_context:
            raise KeyError(f"Object '{name}' not found in external context")
        return self._external_context[name]

    def __repr__(self) -> str:
        """Return string representation of context.

        Returns:
            String representation showing all scopes
        """
        return f"ExecutionContext(scopes={len(self.scopes)}, variables={self.get_all_variables()})"
