"""Variable context management for data operations.

This module provides the VariableContext class for managing variables
across different scopes (local, process, global) with hierarchical resolution.
"""

import logging
from typing import Any

from .constants import VariableScope

logger = logging.getLogger(__name__)


class VariableContext:
    """Manages variables across different scopes.

    Provides a hierarchical variable storage system with three scopes:
    - Local: Action-level variables (highest priority)
    - Process: Process-level variables (medium priority)
    - Global: Application-level variables (lowest priority)

    Variables are resolved in order: local -> process -> global

    Example:
        >>> context = VariableContext()
        >>> context.set("user", "alice", scope="global")
        >>> context.set("user", "bob", scope="local")
        >>> context.get("user")  # Returns "bob" (local overrides global)
        'bob'
    """

    def __init__(self) -> None:
        """Initialize empty variable contexts for all scopes."""
        self.local_vars: dict[str, Any] = {}
        self.process_vars: dict[str, Any] = {}
        self.global_vars: dict[str, Any] = {}
        logger.debug("Initialized VariableContext with empty scopes")

    def set(self, name: str, value: Any, scope: str = "local") -> None:
        """Set a variable in the specified scope.

        Args:
            name: Variable name (must be non-empty)
            value: Variable value (any JSON-serializable type)
            scope: Target scope (local, global, or process). Defaults to "local"

        Raises:
            ValueError: If name is empty or scope is invalid

        Example:
            >>> context = VariableContext()
            >>> context.set("count", 42, scope="global")
            >>> context.set("temp", "value")  # Defaults to local scope
        """
        if not name:
            raise ValueError("Variable name cannot be empty")

        # Normalize scope
        scope = scope.lower() if scope else "local"

        # Truncate value for logging if it's too large
        value_str = str(value)
        if len(value_str) > 100:
            value_str = value_str[:97] + "..."

        try:
            scope_enum = VariableScope(scope)
        except ValueError as err:
            raise ValueError(
                f"Invalid scope '{scope}'. Must be one of: local, global, process"
            ) from err

        if scope_enum == VariableScope.LOCAL:
            self.local_vars[name] = value
            logger.debug(f"Set local variable '{name}' = {value_str}")
        elif scope_enum == VariableScope.PROCESS:
            self.process_vars[name] = value
            logger.debug(f"Set process variable '{name}' = {value_str}")
        elif scope_enum == VariableScope.GLOBAL:
            self.global_vars[name] = value
            logger.debug(f"Set global variable '{name}' = {value_str}")

    def get(self, name: str, default: Any = None) -> Any:
        """Get a variable value from any scope.

        Searches scopes in order of priority: local -> process -> global

        Args:
            name: Variable name
            default: Default value if variable not found. Defaults to None

        Returns:
            Variable value or default if not found

        Example:
            >>> context = VariableContext()
            >>> context.set("x", 10, scope="global")
            >>> context.set("x", 20, scope="local")
            >>> context.get("x")  # Returns 20 (local takes precedence)
            20
            >>> context.get("y", default=0)  # Returns default
            0
        """
        if not name:
            logger.warning("Attempted to get variable with empty name")
            return default

        # Search local scope first
        if name in self.local_vars:
            value = self.local_vars[name]
            logger.debug(f"Retrieved local variable '{name}'")
            return value

        # Then process scope
        if name in self.process_vars:
            value = self.process_vars[name]
            logger.debug(f"Retrieved process variable '{name}'")
            return value

        # Finally global scope
        if name in self.global_vars:
            value = self.global_vars[name]
            logger.debug(f"Retrieved global variable '{name}'")
            return value

        logger.debug(f"Variable '{name}' not found, returning default: {default}")
        return default

    def exists(self, name: str) -> bool:
        """Check if a variable exists in any scope.

        Args:
            name: Variable name

        Returns:
            True if variable exists in any scope, False otherwise

        Example:
            >>> context = VariableContext()
            >>> context.set("x", 10)
            >>> context.exists("x")
            True
            >>> context.exists("y")
            False
        """
        return name in self.local_vars or name in self.process_vars or name in self.global_vars

    def delete(self, name: str, scope: str | None = None) -> bool:
        """Delete a variable from specified scope or all scopes.

        Args:
            name: Variable name
            scope: Target scope (None = all scopes, "local", "process", or "global")

        Returns:
            True if at least one variable was deleted, False otherwise

        Example:
            >>> context = VariableContext()
            >>> context.set("x", 10, scope="local")
            >>> context.set("x", 20, scope="global")
            >>> context.delete("x", scope="local")  # Delete only from local
            True
            >>> context.delete("x")  # Delete from all scopes
            True
        """
        deleted = False

        if scope is None or scope == "local":
            if name in self.local_vars:
                del self.local_vars[name]
                deleted = True
                logger.debug(f"Deleted local variable '{name}'")

        if scope is None or scope == "process":
            if name in self.process_vars:
                del self.process_vars[name]
                deleted = True
                logger.debug(f"Deleted process variable '{name}'")

        if scope is None or scope == "global":
            if name in self.global_vars:
                del self.global_vars[name]
                deleted = True
                logger.debug(f"Deleted global variable '{name}'")

        if not deleted:
            logger.warning(f"Variable '{name}' not found for deletion")

        return deleted

    def clear_scope(self, scope: str) -> None:
        """Clear all variables in a specific scope.

        Args:
            scope: Scope to clear (local, global, or process)

        Raises:
            None, but logs warning if scope is invalid

        Example:
            >>> context = VariableContext()
            >>> context.set("x", 1, scope="local")
            >>> context.set("y", 2, scope="local")
            >>> context.clear_scope("local")
            >>> context.get("x")  # Returns None
        """
        if scope == "local":
            count = len(self.local_vars)
            self.local_vars.clear()
            logger.info(f"Cleared {count} local variables")
        elif scope == "process":
            count = len(self.process_vars)
            self.process_vars.clear()
            logger.info(f"Cleared {count} process variables")
        elif scope == "global":
            count = len(self.global_vars)
            self.global_vars.clear()
            logger.info(f"Cleared {count} global variables")
        else:
            logger.warning(f"Invalid scope '{scope}' for clear operation")

    def get_all_variables(self) -> dict[str, Any]:
        """Get all variables from all scopes merged with proper precedence.

        Returns a dictionary with local overriding process overriding global.
        This is useful for expression evaluation where all variables need to
        be in scope.

        Returns:
            Dictionary of all variables with proper precedence applied

        Example:
            >>> context = VariableContext()
            >>> context.set("x", 1, scope="global")
            >>> context.set("y", 2, scope="process")
            >>> context.set("z", 3, scope="local")
            >>> context.set("x", 10, scope="local")  # Override global
            >>> context.get_all_variables()
            {'x': 10, 'y': 2, 'z': 3}
        """
        # Merge with proper precedence: global < process < local
        merged = {}
        merged.update(self.global_vars)
        merged.update(self.process_vars)
        merged.update(self.local_vars)
        return merged
