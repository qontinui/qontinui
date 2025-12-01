"""Enhanced variable context with three-tier storage and persistence.

This module provides the EnhancedVariableContext class for managing variables
across three distinct tiers:
1. Execution variables (in-memory, temporary, cleared after execution)
2. Workflow variables (persistent within workflow, optionally saved to file)
3. Global variables (project-scoped, persistent, optionally saved to file)

For standalone library use (no database), the system uses:
- In-memory storage for all tiers during execution
- Optional file-based persistence (JSON) for workflow/global vars
- Thread-safe operations for concurrent workflow execution
"""

import json
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

VariableScope = Literal["execution", "workflow", "global"]


class EnhancedVariableContext:
    """Three-tier variable storage with optional file persistence.

    This context manages variables across three scopes with different
    lifetime and persistence characteristics:

    - Execution: Temporary variables for current action execution only
    - Workflow: Persistent variables for the workflow session
    - Global: Project-scoped variables shared across workflows

    Variables are resolved with scope precedence:
    execution -> workflow -> global (highest to lowest priority)

    Thread-safe for concurrent workflow execution.

    Example:
        >>> context = EnhancedVariableContext()
        >>> context.set("api_key", "secret", scope="global")
        >>> context.set("user_id", "12345", scope="workflow")
        >>> context.set("temp_result", {"status": "ok"}, scope="execution")
        >>> context.get("user_id")  # Returns "12345"
        '12345'
        >>> context.clear("execution")  # Clear execution vars
        >>> context.save_to_file("workflow")  # Save workflow vars
    """

    def __init__(
        self,
        workflow_file: Path | str | None = None,
        global_file: Path | str | None = None,
        auto_save: bool = False,
        change_callback: Callable[[str, str, Any], None] | None = None,
    ) -> None:
        """Initialize enhanced variable context.

        Args:
            workflow_file: Optional JSON file path for workflow variables
            global_file: Optional JSON file path for global variables
            auto_save: If True, automatically save to file on set operations
            change_callback: Optional callback(scope, name, value) on variable changes
        """
        # Three-tier storage
        self._execution_vars: dict[str, Any] = {}
        self._workflow_vars: dict[str, Any] = {}
        self._global_vars: dict[str, Any] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Persistence configuration
        self._workflow_file = Path(workflow_file) if workflow_file else None
        self._global_file = Path(global_file) if global_file else None
        self._auto_save = auto_save

        # Change tracking
        self._change_callback = change_callback
        self._change_history: list[dict[str, Any]] = []

        # Load from files if they exist
        if self._workflow_file and self._workflow_file.exists():
            self.load_from_file("workflow")
        if self._global_file and self._global_file.exists():
            self.load_from_file("global")

        logger.debug(
            f"Initialized EnhancedVariableContext "
            f"(workflow_file={workflow_file}, global_file={global_file}, "
            f"auto_save={auto_save})"
        )

    def set(
        self,
        name: str,
        value: Any,
        scope: VariableScope = "execution",
    ) -> None:
        """Set a variable in the specified scope.

        Args:
            name: Variable name (must be non-empty)
            value: Variable value (should be JSON-serializable for persistence)
            scope: Target scope (execution, workflow, or global)

        Raises:
            ValueError: If name is empty or scope is invalid

        Example:
            >>> context = EnhancedVariableContext()
            >>> context.set("count", 42, scope="workflow")
            >>> context.set("temp", "value")  # Defaults to execution scope
        """
        if not name:
            raise ValueError("Variable name cannot be empty")

        if scope not in ("execution", "workflow", "global"):
            raise ValueError(
                f"Invalid scope '{scope}'. Must be one of: "
                "execution, workflow, global"
            )

        with self._lock:
            # Get target storage
            if scope == "execution":
                storage = self._execution_vars
            elif scope == "workflow":
                storage = self._workflow_vars
            else:  # global
                storage = self._global_vars

            # Store value
            storage[name] = value

            # Track change
            self._track_change("set", scope, name, value)

            # Call change callback
            if self._change_callback:
                try:
                    self._change_callback(scope, name, value)
                except Exception as e:
                    logger.warning(f"Change callback failed: {e}")

            # Auto-save if enabled and scope is persistent
            if self._auto_save and scope in ("workflow", "global"):
                try:
                    self.save_to_file(scope)  # type: ignore[arg-type]
                except Exception as e:
                    logger.warning(f"Auto-save failed for {scope} scope: {e}")

            # Log (truncate value for large objects)
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:97] + "..."
            logger.debug(f"Set {scope} variable '{name}' = {value_str}")

    def get(
        self,
        name: str,
        default: Any = None,
        scope: VariableScope | None = None,
    ) -> Any:
        """Get a variable value from specified scope or with precedence.

        If scope is None, searches in order: execution -> workflow -> global
        If scope is specified, only searches that scope.

        Args:
            name: Variable name
            default: Default value if variable not found
            scope: Optional scope to search (None = search all with precedence)

        Returns:
            Variable value or default if not found

        Example:
            >>> context = EnhancedVariableContext()
            >>> context.set("x", 10, scope="global")
            >>> context.set("x", 20, scope="execution")
            >>> context.get("x")  # Returns 20 (execution takes precedence)
            20
            >>> context.get("x", scope="global")  # Returns 10
            10
            >>> context.get("y", default=0)  # Returns default
            0
        """
        if not name:
            logger.warning("Attempted to get variable with empty name")
            return default

        with self._lock:
            # If specific scope requested, only check that scope
            if scope is not None:
                if scope == "execution":
                    storage = self._execution_vars
                elif scope == "workflow":
                    storage = self._workflow_vars
                elif scope == "global":
                    storage = self._global_vars
                else:
                    logger.warning(f"Invalid scope '{scope}', returning default")
                    return default

                if name in storage:
                    logger.debug(f"Retrieved {scope} variable '{name}'")
                    return storage[name]
                else:
                    logger.debug(
                        f"Variable '{name}' not found in {scope} scope, "
                        f"returning default"
                    )
                    return default

            # Otherwise, search with precedence: execution -> workflow -> global
            if name in self._execution_vars:
                logger.debug(f"Retrieved execution variable '{name}'")
                return self._execution_vars[name]

            if name in self._workflow_vars:
                logger.debug(f"Retrieved workflow variable '{name}'")
                return self._workflow_vars[name]

            if name in self._global_vars:
                logger.debug(f"Retrieved global variable '{name}'")
                return self._global_vars[name]

            logger.debug(f"Variable '{name}' not found in any scope, returning default")
            return default

    def get_all(self, scope: VariableScope | None = None) -> dict[str, Any]:
        """Get all variables from specified scope or all scopes merged.

        If scope is None, returns merged dict with proper precedence:
        global < workflow < execution (later scopes override earlier)

        Args:
            scope: Optional scope to get (None = all scopes merged)

        Returns:
            Dictionary of variables

        Example:
            >>> context = EnhancedVariableContext()
            >>> context.set("x", 1, scope="global")
            >>> context.set("y", 2, scope="workflow")
            >>> context.set("x", 10, scope="execution")
            >>> context.get_all()  # Returns {'x': 10, 'y': 2}
            {'x': 10, 'y': 2}
            >>> context.get_all(scope="workflow")  # Returns {'y': 2}
            {'y': 2}
        """
        with self._lock:
            if scope == "execution":
                return dict(self._execution_vars)
            elif scope == "workflow":
                return dict(self._workflow_vars)
            elif scope == "global":
                return dict(self._global_vars)
            elif scope is None:
                # Merge with precedence: global < workflow < execution
                merged = {}
                merged.update(self._global_vars)
                merged.update(self._workflow_vars)
                merged.update(self._execution_vars)
                return merged
            else:
                logger.warning(f"Invalid scope '{scope}', returning empty dict")
                return {}

    def exists(self, name: str, scope: VariableScope | None = None) -> bool:
        """Check if a variable exists in specified scope or any scope.

        Args:
            name: Variable name
            scope: Optional scope to check (None = check all scopes)

        Returns:
            True if variable exists, False otherwise

        Example:
            >>> context = EnhancedVariableContext()
            >>> context.set("x", 10, scope="workflow")
            >>> context.exists("x")
            True
            >>> context.exists("x", scope="execution")
            False
            >>> context.exists("y")
            False
        """
        with self._lock:
            if scope == "execution":
                return name in self._execution_vars
            elif scope == "workflow":
                return name in self._workflow_vars
            elif scope == "global":
                return name in self._global_vars
            elif scope is None:
                return (
                    name in self._execution_vars
                    or name in self._workflow_vars
                    or name in self._global_vars
                )
            else:
                return False

    def delete(
        self,
        name: str,
        scope: VariableScope | None = None,
    ) -> bool:
        """Delete a variable from specified scope or all scopes.

        Args:
            name: Variable name
            scope: Target scope (None = all scopes)

        Returns:
            True if at least one variable was deleted, False otherwise

        Example:
            >>> context = EnhancedVariableContext()
            >>> context.set("x", 10, scope="execution")
            >>> context.set("x", 20, scope="workflow")
            >>> context.delete("x", scope="execution")
            True
            >>> context.delete("x")  # Delete from all scopes
            True
        """
        deleted = False

        with self._lock:
            if scope is None or scope == "execution":
                if name in self._execution_vars:
                    del self._execution_vars[name]
                    deleted = True
                    self._track_change("delete", "execution", name, None)
                    logger.debug(f"Deleted execution variable '{name}'")

            if scope is None or scope == "workflow":
                if name in self._workflow_vars:
                    del self._workflow_vars[name]
                    deleted = True
                    self._track_change("delete", "workflow", name, None)
                    logger.debug(f"Deleted workflow variable '{name}'")

                    # Auto-save if enabled
                    if self._auto_save:
                        try:
                            self.save_to_file("workflow")
                        except Exception as e:
                            logger.warning(f"Auto-save failed after delete: {e}")

            if scope is None or scope == "global":
                if name in self._global_vars:
                    del self._global_vars[name]
                    deleted = True
                    self._track_change("delete", "global", name, None)
                    logger.debug(f"Deleted global variable '{name}'")

                    # Auto-save if enabled
                    if self._auto_save:
                        try:
                            self.save_to_file("global")
                        except Exception as e:
                            logger.warning(f"Auto-save failed after delete: {e}")

        if not deleted:
            logger.warning(f"Variable '{name}' not found for deletion")

        return deleted

    def clear(self, scope: VariableScope) -> int:
        """Clear all variables in a specific scope.

        Args:
            scope: Scope to clear (execution, workflow, or global)

        Returns:
            Number of variables cleared

        Raises:
            ValueError: If scope is invalid

        Example:
            >>> context = EnhancedVariableContext()
            >>> context.set("x", 1, scope="execution")
            >>> context.set("y", 2, scope="execution")
            >>> context.clear("execution")
            2
            >>> context.get("x")  # Returns None
        """
        if scope not in ("execution", "workflow", "global"):
            raise ValueError(
                f"Invalid scope '{scope}'. Must be one of: "
                "execution, workflow, global"
            )

        with self._lock:
            if scope == "execution":
                count = len(self._execution_vars)
                self._execution_vars.clear()
            elif scope == "workflow":
                count = len(self._workflow_vars)
                self._workflow_vars.clear()

                # Auto-save if enabled
                if self._auto_save:
                    try:
                        self.save_to_file("workflow")
                    except Exception as e:
                        logger.warning(f"Auto-save failed after clear: {e}")
            else:  # global
                count = len(self._global_vars)
                self._global_vars.clear()

                # Auto-save if enabled
                if self._auto_save:
                    try:
                        self.save_to_file("global")
                    except Exception as e:
                        logger.warning(f"Auto-save failed after clear: {e}")

            self._track_change("clear", scope, None, None)
            logger.info(f"Cleared {count} {scope} variables")
            return count

    def save_to_file(self, scope: Literal["workflow", "global"]) -> None:
        """Save variables to JSON file for persistence.

        Only workflow and global scopes support file persistence.
        Execution variables are intentionally ephemeral.

        Args:
            scope: Scope to save (workflow or global)

        Raises:
            ValueError: If scope is invalid or no file configured
            IOError: If file write fails

        Example:
            >>> context = EnhancedVariableContext(
            ...     workflow_file="workflow_vars.json"
            ... )
            >>> context.set("user_id", "123", scope="workflow")
            >>> context.save_to_file("workflow")
        """
        if scope not in ("workflow", "global"):
            raise ValueError(
                f"Invalid scope '{scope}'. Only workflow and global support "
                "file persistence"
            )

        with self._lock:
            # Get target file
            if scope == "workflow":
                target_file = self._workflow_file
                storage = self._workflow_vars
            else:  # global
                target_file = self._global_file
                storage = self._global_vars

            if target_file is None:
                raise ValueError(
                    f"No file configured for {scope} scope. "
                    f"Set {scope}_file in constructor."
                )

            # Ensure parent directory exists
            target_file.parent.mkdir(parents=True, exist_ok=True)

            # Write JSON
            try:
                with open(target_file, "w", encoding="utf-8") as f:
                    json.dump(storage, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(storage)} {scope} variables to {target_file}")
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Failed to serialize {scope} variables to JSON: {e}"
                ) from e
            except OSError as e:
                raise OSError(
                    f"Failed to write {scope} variables to {target_file}: {e}"
                ) from e

    def load_from_file(self, scope: Literal["workflow", "global"]) -> int:
        """Load variables from JSON file.

        Only workflow and global scopes support file persistence.

        Args:
            scope: Scope to load (workflow or global)

        Returns:
            Number of variables loaded

        Raises:
            ValueError: If scope is invalid or no file configured
            IOError: If file read fails

        Example:
            >>> context = EnhancedVariableContext(
            ...     workflow_file="workflow_vars.json"
            ... )
            >>> count = context.load_from_file("workflow")
        """
        if scope not in ("workflow", "global"):
            raise ValueError(
                f"Invalid scope '{scope}'. Only workflow and global support "
                "file persistence"
            )

        with self._lock:
            # Get target file
            if scope == "workflow":
                target_file = self._workflow_file
                storage = self._workflow_vars
            else:  # global
                target_file = self._global_file
                storage = self._global_vars

            if target_file is None:
                raise ValueError(
                    f"No file configured for {scope} scope. "
                    f"Set {scope}_file in constructor."
                )

            if not target_file.exists():
                logger.warning(
                    f"{scope.capitalize()} variables file not found: " f"{target_file}"
                )
                return 0

            # Read JSON
            try:
                with open(target_file, encoding="utf-8") as f:
                    loaded_vars = json.load(f)

                if not isinstance(loaded_vars, dict):
                    raise ValueError(
                        f"Invalid format in {target_file}: expected dict, "
                        f"got {type(loaded_vars).__name__}"
                    )

                # Update storage
                storage.clear()
                storage.update(loaded_vars)

                logger.info(
                    f"Loaded {len(loaded_vars)} {scope} variables from "
                    f"{target_file}"
                )
                return len(loaded_vars)

            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON from {target_file}: {e}") from e
            except OSError as e:
                raise OSError(
                    f"Failed to read {scope} variables from {target_file}: {e}"
                ) from e

    def get_change_history(self) -> list[dict[str, Any]]:
        """Get history of variable changes.

        Returns:
            List of change records with operation, scope, name, value, timestamp

        Example:
            >>> context = EnhancedVariableContext()
            >>> context.set("x", 10, scope="workflow")
            >>> context.set("x", 20, scope="workflow")
            >>> history = context.get_change_history()
            >>> len(history)
            2
        """
        with self._lock:
            return list(self._change_history)

    def clear_change_history(self) -> None:
        """Clear the change history.

        Example:
            >>> context = EnhancedVariableContext()
            >>> context.set("x", 10)
            >>> context.clear_change_history()
            >>> len(context.get_change_history())
            0
        """
        with self._lock:
            self._change_history.clear()
            logger.debug("Cleared change history")

    def _track_change(
        self,
        operation: str,
        scope: str,
        name: str | None,
        value: Any,
    ) -> None:
        """Track a variable change in history (internal method).

        Args:
            operation: Operation type (set, delete, clear)
            scope: Variable scope
            name: Variable name (None for clear operation)
            value: Variable value (None for delete operation)
        """
        import time

        change_record = {
            "operation": operation,
            "scope": scope,
            "name": name,
            "value": value,
            "timestamp": time.time(),
        }
        self._change_history.append(change_record)

        # Keep history bounded (last 1000 changes)
        if len(self._change_history) > 1000:
            self._change_history = self._change_history[-1000:]

    # Backward compatibility methods for existing VariableContext interface

    @property
    def variables(self) -> dict[str, Any]:
        """Get all variables (merged from all scopes) for backward compatibility.

        This property provides compatibility with the old VariableContext
        interface that exposed a single 'variables' dict.

        Returns:
            Merged dictionary with scope precedence applied
        """
        return self.get_all()

    def get_all_variables(self) -> dict[str, Any]:
        """Get all variables merged for backward compatibility.

        Alias for get_all() with no scope argument.

        Returns:
            Merged dictionary with scope precedence applied
        """
        return self.get_all()

    def clear_scope(self, scope: str) -> None:
        """Clear scope for backward compatibility (no return value).

        Args:
            scope: Scope to clear
        """
        # Map old scope names to new ones
        scope_mapping = {
            "local": "execution",
            "process": "workflow",
            "global": "global",
        }

        mapped_scope = scope_mapping.get(scope, scope)

        if mapped_scope in ("execution", "workflow", "global"):
            self.clear(mapped_scope)  # type: ignore
        else:
            logger.warning(f"Invalid scope '{scope}' for clear operation")
