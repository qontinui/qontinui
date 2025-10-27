"""Data operations executor for variable management and collection operations.

This module provides executors for data-related actions including:
- Variable management (SET_VARIABLE, GET_VARIABLE)
- Collection operations (SORT, FILTER)
- Safe expression evaluation
- Multi-scope variable contexts (local, global, process)
"""

import ast
import json
import logging
import re
from datetime import datetime
from enum import Enum
from typing import Any

from ..exceptions import ActionExecutionError, ConfigurationError

try:
    import pyperclip

    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False

from qontinui.config import (
    Action,
    FilterActionConfig,
    GetVariableActionConfig,
    MapActionConfig,
    MathOperationActionConfig,
    ReduceActionConfig,
    SetVariableActionConfig,
    SortActionConfig,
    StringOperationActionConfig,
    get_typed_config,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================


class VariableScope(str, Enum):
    """Variable scope levels."""

    LOCAL = "local"
    GLOBAL = "global"
    PROCESS = "process"


class ComparatorType(str, Enum):
    """Types of comparators for sorting."""

    NUMERIC = "NUMERIC"
    ALPHABETIC = "ALPHABETIC"
    DATE = "DATE"
    CUSTOM = "CUSTOM"


class ComparisonOperator(str, Enum):
    """Comparison operators for filtering."""

    EQUAL = "=="
    NOT_EQUAL = "!="
    GREATER = ">"
    LESS = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    CONTAINS = "contains"
    MATCHES = "matches"


# ============================================================================
# Variable Context Management
# ============================================================================


class VariableContext:
    """Manages variables across different scopes.

    Provides a hierarchical variable storage system with three scopes:
    - Local: Action-level variables (highest priority)
    - Process: Process-level variables (medium priority)
    - Global: Application-level variables (lowest priority)

    Variables are resolved in order: local -> process -> global
    """

    def __init__(self):
        """Initialize empty variable contexts for all scopes."""
        self.local_vars: dict[str, Any] = {}
        self.process_vars: dict[str, Any] = {}
        self.global_vars: dict[str, Any] = {}
        logger.debug("Initialized VariableContext with empty scopes")

    def set(self, name: str, value: Any, scope: str = "local") -> None:
        """Set a variable in the specified scope.

        Args:
            name: Variable name
            value: Variable value (any JSON-serializable type)
            scope: Target scope (local, global, or process)

        Raises:
            ValueError: If scope is invalid
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

        Searches scopes in order: local -> process -> global

        Args:
            name: Variable name
            default: Default value if variable not found

        Returns:
            Variable value or default if not found
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
            True if variable exists in any scope
        """
        return name in self.local_vars or name in self.process_vars or name in self.global_vars

    def delete(self, name: str, scope: str | None = None) -> bool:
        """Delete a variable from specified scope or all scopes.

        Args:
            name: Variable name
            scope: Target scope (None = all scopes)

        Returns:
            True if variable was deleted
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
        """Get all variables from all scopes merged.

        Returns dict with local overriding process overriding global.

        Returns:
            Dictionary of all variables
        """
        # Merge with proper precedence
        merged = {}
        merged.update(self.global_vars)
        merged.update(self.process_vars)
        merged.update(self.local_vars)
        return merged


# ============================================================================
# Safe Expression Evaluation
# ============================================================================


class SafeEvaluator:
    """Safe expression evaluator with restricted capabilities.

    Allows basic arithmetic, comparisons, and variable access while
    preventing dangerous operations like file I/O, imports, etc.
    """

    # Allowed node types for safe evaluation
    ALLOWED_NODES = {
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Store,  # For list comprehensions
        ast.UnaryOp,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.IfExp,
        ast.Subscript,
        ast.Index,
        ast.Slice,
        ast.List,
        ast.Tuple,
        ast.Dict,
        ast.Set,
        ast.ListComp,
        ast.DictComp,
        ast.SetComp,
        ast.GeneratorExp,
        ast.comprehension,
        ast.Attribute,
        ast.Call,  # Allow function calls (validated separately)
    }

    # Safe built-in functions
    SAFE_FUNCTIONS = {
        "abs": abs,
        "bool": bool,
        "float": float,
        "int": int,
        "len": len,
        "max": max,
        "min": min,
        "range": range,
        "round": round,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "any": any,
        "all": all,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "set": set,
    }

    @classmethod
    def safe_eval(cls, expression: str, context: dict[str, Any]) -> Any:
        """Safely evaluate a Python expression.

        Args:
            expression: Python expression to evaluate
            context: Variable context for evaluation

        Returns:
            Result of evaluation

        Raises:
            ValueError: If expression contains unsafe operations
            SyntaxError: If expression has invalid syntax
        """
        if not expression or not isinstance(expression, str):
            raise ValueError("Expression must be a non-empty string")

        expression = expression.strip()
        logger.debug(f"Evaluating expression: {expression}")

        try:
            # Parse the expression
            tree = ast.parse(expression, mode="eval")

            # Validate all nodes are safe
            for node in ast.walk(tree):
                if type(node) not in cls.ALLOWED_NODES:
                    raise ValueError(f"Unsafe operation in expression: {type(node).__name__}")

                # Extra validation for function calls
                if isinstance(node, ast.Call):
                    # Only allow calls to functions in SAFE_FUNCTIONS
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name not in cls.SAFE_FUNCTIONS:
                            raise ValueError(f"Unsafe function call: {func_name}")
                    elif isinstance(node.func, ast.Attribute):
                        # Allow method calls on safe objects (like list.append, str.upper, etc.)
                        pass
                    else:
                        raise ValueError(f"Unsafe function call type: {type(node.func).__name__}")

            # Create safe namespace
            safe_namespace = {
                "__builtins__": cls.SAFE_FUNCTIONS,
            }
            safe_namespace.update(context)

            # Evaluate
            result = eval(compile(tree, "<string>", "eval"), safe_namespace)

            logger.debug(f"Expression evaluated to: {result}")
            return result

        except SyntaxError as e:
            logger.error(f"Syntax error in expression: {e}")
            raise
        except Exception as e:
            logger.error(f"Error evaluating expression: {e}")
            raise ValueError(f"Failed to evaluate expression: {e}") from e

    @classmethod
    def is_safe_expression(cls, expression: str) -> bool:
        """Check if an expression is safe to evaluate.

        Args:
            expression: Expression to check

        Returns:
            True if expression is safe
        """
        try:
            tree = ast.parse(expression, mode="eval")
            for node in ast.walk(tree):
                if type(node) not in cls.ALLOWED_NODES:
                    return False
            return True
        except (SyntaxError, ValueError) as e:
            # Invalid Python expression
            return False


# ============================================================================
# Type Coercion
# ============================================================================


class TypeCoercer:
    """Handles type coercion for variable values."""

    @staticmethod
    def coerce(value: Any, target_type: str | None) -> Any:
        """Coerce a value to the specified type.

        Args:
            value: Value to coerce
            target_type: Target type (string, number, boolean, array, object)

        Returns:
            Coerced value

        Raises:
            ValueError: If coercion fails
        """
        if target_type is None or value is None:
            return value

        target_type = target_type.lower()

        try:
            if target_type == "string":
                return str(value)

            elif target_type == "number":
                # Try int first, then float
                if isinstance(value, str):
                    if "." in value:
                        return float(value)
                    return int(value)
                return float(value)

            elif target_type == "boolean":
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.lower() in ("true", "yes", "1", "on")
                return bool(value)

            elif target_type == "array":
                if isinstance(value, list | tuple):
                    return list(value)
                if isinstance(value, str):
                    return json.loads(value)
                return [value]

            elif target_type == "object":
                if isinstance(value, dict):
                    return value
                if isinstance(value, str):
                    return json.loads(value)
                raise ValueError(f"Cannot coerce {type(value)} to object")

            else:
                logger.warning(f"Unknown target type '{target_type}', returning value as-is")
                return value

        except Exception as e:
            logger.error(f"Type coercion failed: {e}")
            raise ValueError(f"Failed to coerce value to {target_type}: {e}") from e


# ============================================================================
# Data Operations Executor
# ============================================================================


class DataOperationsExecutor:
    """Executor for data operations including variable management and collection operations.

    Provides execution methods for:
    - SET_VARIABLE: Set variables from various sources
    - GET_VARIABLE: Retrieve variable values
    - SORT: Sort collections by properties
    - FILTER: Filter collections by conditions
    """

    def __init__(self, variable_context: VariableContext | None = None):
        """Initialize the data operations executor.

        Args:
            variable_context: Variable context to use (creates new if None)
        """
        self.variable_context = variable_context or VariableContext()
        self.evaluator = SafeEvaluator()
        self.coercer = TypeCoercer()
        logger.info("Initialized DataOperationsExecutor")

    def execute_set_variable(self, action: Action, context: dict) -> dict:
        """Execute SET_VARIABLE action.

        Sets a variable from various sources:
        - Direct value
        - Expression evaluation
        - Target extraction (OCR, image matching)
        - Clipboard contents

        Args:
            action: Action configuration
            context: Execution context

        Returns:
            Dictionary with variable info

        Raises:
            ValueError: If configuration is invalid
        """
        try:
            config: SetVariableActionConfig = get_typed_config(action)

            logger.info(f"Setting variable '{config.variable_name}'")

            # Determine value source
            value = None

            if config.value is not None:
                # Direct value
                value = config.value
                logger.debug(f"Using direct value: {value}")

            elif config.value_source:
                # Value from source
                source_type = config.value_source.type

                if source_type == "expression":
                    # Evaluate expression
                    expression = config.value_source.expression
                    if not expression:
                        raise ValueError("Expression source requires 'expression' field")

                    value = self.evaluator.safe_eval(
                        expression, self.variable_context.get_all_variables()
                    )
                    logger.debug(f"Evaluated expression to: {value}")

                elif source_type == "clipboard":
                    # Read from clipboard
                    if not CLIPBOARD_AVAILABLE:
                        raise ValueError("Clipboard access requires 'pyperclip' package")

                    value = pyperclip.paste()
                    logger.debug(f"Read from clipboard: {value[:50]}...")

                elif source_type == "ocr":
                    # OCR extraction from target
                    # This would require integration with OCR engine
                    raise NotImplementedError(
                        "OCR value source not yet implemented. "
                        "Requires integration with OCR engine."
                    )

                elif source_type == "target":
                    # Extract value from screen target
                    # This would require integration with find/match operations
                    raise NotImplementedError(
                        "Target value source not yet implemented. "
                        "Requires integration with find operations."
                    )

                else:
                    raise ValueError(f"Unknown value source type: {source_type}")

            else:
                raise ValueError("SET_VARIABLE requires either 'value' or 'value_source'")

            # Type coercion if specified
            if config.type:
                value = self.coercer.coerce(value, config.type)

            # Store in appropriate scope
            scope = config.scope or "local"
            self.variable_context.set(config.variable_name, value, scope)

            # Store in context dict too for backwards compatibility
            context[config.variable_name] = value

            result = {
                "success": True,
                "variable_name": config.variable_name,
                "value": value,
                "scope": scope,
                "type": type(value).__name__,
            }

            logger.info(f"Successfully set variable '{config.variable_name}' in {scope} scope")
            return result

        except Exception as e:
            logger.error(f"SET_VARIABLE failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "variable_name": getattr(config, "variable_name", "unknown"),
            }

    def execute_get_variable(self, action: Action, context: dict) -> dict:
        """Execute GET_VARIABLE action.

        Retrieves a variable value and optionally stores it in another variable.

        Args:
            action: Action configuration
            context: Execution context

        Returns:
            Dictionary with variable value
        """
        try:
            config: GetVariableActionConfig = get_typed_config(action)

            logger.info(f"Getting variable '{config.variable_name}'")

            # Get variable value
            value = self.variable_context.get(config.variable_name, config.default_value)

            # Store in output variable if specified
            if config.output_variable:
                self.variable_context.set(config.output_variable, value)
                context[config.output_variable] = value
                logger.debug(f"Stored value in output variable '{config.output_variable}'")

            result = {
                "success": True,
                "variable_name": config.variable_name,
                "value": value,
                "found": self.variable_context.exists(config.variable_name),
            }

            logger.info(f"Successfully retrieved variable '{config.variable_name}'")
            return result

        except Exception as e:
            logger.error(f"GET_VARIABLE failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "variable_name": getattr(config, "variable_name", "unknown"),
            }

    def execute_sort(self, action: Action, context: dict) -> dict:
        """Execute SORT action.

        Sorts a collection by specified properties.

        Args:
            action: Action configuration
            context: Execution context

        Returns:
            Dictionary with sorted collection
        """
        try:
            config: SortActionConfig = get_typed_config(action)

            logger.info(f"Sorting collection from '{config.variable_name}'")

            # Get collection to sort
            collection = None

            if config.target == "variable":
                if not config.variable_name:
                    raise ValueError("SORT with target='variable' requires 'variable_name'")

                collection = self.variable_context.get(config.variable_name)
                if collection is None:
                    raise ValueError(f"Variable '{config.variable_name}' not found")

            elif config.target == "matches":
                # Get matches from context or find operation
                collection = context.get("matches", [])
                if not collection:
                    logger.warning("No matches found in context")

            elif config.target == "list":
                # Direct list from config (would need to be in config)
                collection = context.get("list", [])

            else:
                raise ValueError(f"Unknown sort target: {config.target}")

            if not isinstance(collection, list | tuple):
                raise ValueError(f"Cannot sort non-collection type: {type(collection)}")

            if len(collection) == 0:
                logger.warning("Empty collection, nothing to sort")
                sorted_collection = []
            else:
                # Perform sort
                sorted_collection = self._sort_collection(
                    list(collection),
                    config.sort_by,
                    config.order,
                    config.comparator,
                    config.custom_comparator,
                )

            # Store result
            if config.output_variable:
                self.variable_context.set(config.output_variable, sorted_collection)
                context[config.output_variable] = sorted_collection

            result = {
                "success": True,
                "sorted_collection": sorted_collection,
                "item_count": len(sorted_collection),
                "sort_by": config.sort_by,
                "order": config.order,
            }

            logger.info(f"Successfully sorted {len(sorted_collection)} items")
            return result

        except Exception as e:
            logger.error(f"SORT failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "sorted_collection": []}

    def _sort_collection(
        self,
        collection: list[Any],
        sort_by: str | list[str] | None,
        order: str,
        comparator: str | None,
        custom_comparator: str | None,
    ) -> list[Any]:
        """Sort a collection using specified parameters.

        Args:
            collection: Collection to sort
            sort_by: Property name(s) to sort by
            order: ASC or DESC
            comparator: Comparator type
            custom_comparator: Custom comparator expression

        Returns:
            Sorted collection
        """
        if not collection:
            return collection

        # Determine sort key function
        key_func = None

        if sort_by:
            # Sort by property
            properties = [sort_by] if isinstance(sort_by, str) else sort_by

            def extract_property(item):
                """Extract property value from item."""
                if isinstance(item, dict):
                    # Nested property access
                    value = item
                    for prop in properties:
                        if isinstance(value, dict):
                            value = value.get(prop)
                        else:
                            break
                    return value
                else:
                    # Try attribute access
                    value = item
                    for prop in properties:
                        try:
                            value = getattr(value, prop)
                        except AttributeError:
                            value = None
                            break
                    return value

            key_func = extract_property

        # Apply comparator transformation
        if comparator:
            comparator_enum = ComparatorType(comparator)

            if comparator_enum == ComparatorType.NUMERIC:
                original_key = key_func or (lambda x: x)

                def key_func(x):
                    return float(original_key(x) or 0)

            elif comparator_enum == ComparatorType.ALPHABETIC:
                original_key = key_func or (lambda x: x)

                def key_func(x):
                    return str(original_key(x) or "")

            elif comparator_enum == ComparatorType.DATE:
                original_key = key_func or (lambda x: x)

                def date_key(x):
                    value = original_key(x)
                    if isinstance(value, datetime):
                        return value
                    if isinstance(value, str):
                        # Try to parse as ISO format
                        try:
                            return datetime.fromisoformat(value)
                        except (ValueError, TypeError) as e:
                            # Invalid date format, use minimum date for sorting
                            return datetime.min
                    return datetime.min

                key_func = date_key

            elif comparator_enum == ComparatorType.CUSTOM:
                if not custom_comparator:
                    raise ValueError("CUSTOM comparator requires 'custom_comparator'")

                # Evaluate custom comparator as expression
                def custom_key(x):
                    try:
                        return self.evaluator.safe_eval(
                            custom_comparator,
                            {"item": x, **self.variable_context.get_all_variables()},
                        )
                    except (ValueError, SyntaxError, TypeError, NameError) as e:
                        # Expression evaluation failed, return default sort value
                        return 0

                key_func = custom_key

        # Perform sort
        reverse = order == "DESC"

        try:
            sorted_list = sorted(collection, key=key_func, reverse=reverse)
        except Exception as e:
            logger.error(f"Sort failed: {e}")
            # Return original on error
            sorted_list = collection

        return sorted_list

    def execute_filter(self, action: Action, context: dict) -> dict:
        """Execute FILTER action.

        Filters a collection based on conditions.

        Args:
            action: Action configuration
            context: Execution context

        Returns:
            Dictionary with filtered collection
        """
        try:
            config: FilterActionConfig = get_typed_config(action)

            logger.info(f"Filtering collection from '{config.variable_name}'")

            # Get collection to filter
            collection = self.variable_context.get(config.variable_name)

            if collection is None:
                raise ValueError(f"Variable '{config.variable_name}' not found")

            if not isinstance(collection, list | tuple):
                raise ValueError(f"Cannot filter non-collection type: {type(collection)}")

            # Perform filter
            filtered_collection = self._filter_collection(list(collection), config.condition)

            # Store result
            if config.output_variable:
                self.variable_context.set(config.output_variable, filtered_collection)
                context[config.output_variable] = filtered_collection

            result = {
                "success": True,
                "filtered_collection": filtered_collection,
                "original_count": len(collection),
                "filtered_count": len(filtered_collection),
            }

            logger.info(
                f"Successfully filtered collection: "
                f"{len(collection)} -> {len(filtered_collection)} items"
            )
            return result

        except Exception as e:
            logger.error(f"FILTER failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "filtered_collection": []}

    def _filter_collection(self, collection: list[Any], condition) -> list[Any]:
        """Filter a collection using specified condition.

        Args:
            collection: Collection to filter
            condition: Filter condition configuration

        Returns:
            Filtered collection
        """
        if not collection:
            return collection

        filtered = []

        for item in collection:
            try:
                if self._evaluate_filter_condition(item, condition):
                    filtered.append(item)
            except Exception as e:
                logger.warning(f"Filter condition evaluation failed for item: {e}")
                # Skip items that fail evaluation
                continue

        return filtered

    def _evaluate_filter_condition(self, item: Any, condition) -> bool:
        """Evaluate a filter condition for an item.

        Args:
            item: Item to evaluate
            condition: Filter condition

        Returns:
            True if item matches condition
        """
        condition_type = condition.type

        if condition_type == "expression":
            # Evaluate expression with item in context
            if not condition.expression:
                raise ValueError("Expression condition requires 'expression'")

            result = self.evaluator.safe_eval(
                condition.expression, {"item": item, **self.variable_context.get_all_variables()}
            )
            return bool(result)

        elif condition_type == "property":
            # Compare property value
            if not condition.property:
                raise ValueError("Property condition requires 'property'")

            # Extract property value
            if isinstance(item, dict):
                item_value = item.get(condition.property)
            else:
                item_value = getattr(item, condition.property, None)

            # Compare using operator
            return self._compare_values(item_value, condition.value, condition.operator)

        elif condition_type == "custom":
            # Evaluate custom function
            if not condition.custom_function:
                raise ValueError("Custom condition requires 'custom_function'")

            result = self.evaluator.safe_eval(
                condition.custom_function,
                {"item": item, **self.variable_context.get_all_variables()},
            )
            return bool(result)

        else:
            raise ValueError(f"Unknown condition type: {condition_type}")

    def execute_map(self, action: Action, context: dict) -> dict:
        """Execute MAP action.

        Transforms each item in a collection using an expression or function.

        Args:
            action: Action configuration
            context: Execution context

        Returns:
            Dictionary with transformed collection
        """
        try:
            config: MapActionConfig = get_typed_config(action)

            logger.info(f"Mapping over collection from '{config.variable_name}'")

            # Get collection to map
            collection = self.variable_context.get(config.variable_name)

            if collection is None:
                raise ValueError(f"Variable '{config.variable_name}' not found")

            if not isinstance(collection, list | tuple):
                raise ValueError(f"Cannot map over non-collection type: {type(collection)}")

            # Perform map
            mapped_collection = self._map_collection(list(collection), config.transform)

            # Store result
            if config.output_variable:
                self.variable_context.set(config.output_variable, mapped_collection)
                context[config.output_variable] = mapped_collection

            result = {
                "success": True,
                "mapped_collection": mapped_collection,
                "item_count": len(mapped_collection),
                "transform_type": config.transform.type,
            }

            logger.info(f"Successfully mapped {len(mapped_collection)} items")
            return result

        except Exception as e:
            logger.error(f"MAP failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "mapped_collection": []}

    def _map_collection(self, collection: list[Any], transform) -> list[Any]:
        """Map a collection using specified transform.

        Args:
            collection: Collection to map
            transform: Transform configuration

        Returns:
            Mapped collection
        """
        if not collection:
            return collection

        mapped = []
        transform_type = transform.type

        for item in collection:
            try:
                if transform_type == "expression":
                    # Evaluate expression with item in context
                    if not transform.expression:
                        raise ValueError("Expression transform requires 'expression'")

                    result = self.evaluator.safe_eval(
                        transform.expression,
                        {"item": item, **self.variable_context.get_all_variables()},
                    )
                    mapped.append(result)

                elif transform_type == "property":
                    # Extract property value
                    if not transform.property:
                        raise ValueError("Property transform requires 'property'")

                    if isinstance(item, dict):
                        value = item.get(transform.property)
                    else:
                        value = getattr(item, transform.property, None)

                    mapped.append(value)

                elif transform_type == "custom":
                    # Evaluate custom function
                    if not transform.custom_function:
                        raise ValueError("Custom transform requires 'custom_function'")

                    result = self.evaluator.safe_eval(
                        transform.custom_function,
                        {"item": item, **self.variable_context.get_all_variables()},
                    )
                    mapped.append(result)

                else:
                    raise ValueError(f"Unknown transform type: {transform_type}")

            except Exception as e:
                logger.warning(f"Map transform failed for item: {e}")
                # Add None for failed items
                mapped.append(None)

        return mapped

    def execute_reduce(self, action: Action, context: dict) -> dict:
        """Execute REDUCE action.

        Reduces a collection to a single value using an operation or function.

        Args:
            action: Action configuration
            context: Execution context

        Returns:
            Dictionary with reduced value
        """
        try:
            config: ReduceActionConfig = get_typed_config(action)

            logger.info(f"Reducing collection from '{config.variable_name}'")

            # Get collection to reduce
            collection = self.variable_context.get(config.variable_name)

            if collection is None:
                raise ValueError(f"Variable '{config.variable_name}' not found")

            if not isinstance(collection, list | tuple):
                raise ValueError(f"Cannot reduce non-collection type: {type(collection)}")

            # Perform reduce
            reduced_value = self._reduce_collection(
                list(collection), config.operation, config.initial_value, config.custom_reducer
            )

            # Store result
            if config.output_variable:
                self.variable_context.set(config.output_variable, reduced_value)
                context[config.output_variable] = reduced_value

            result = {
                "success": True,
                "reduced_value": reduced_value,
                "operation": config.operation,
                "item_count": len(collection),
            }

            logger.info(f"Successfully reduced {len(collection)} items to {reduced_value}")
            return result

        except Exception as e:
            logger.error(f"REDUCE failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "reduced_value": None}

    def _reduce_collection(
        self,
        collection: list[Any],
        operation: str,
        initial_value: Any,
        custom_reducer: str | None,
    ) -> Any:
        """Reduce a collection to a single value.

        Args:
            collection: Collection to reduce
            operation: Reduce operation (sum, average, min, max, count, custom)
            initial_value: Initial accumulator value
            custom_reducer: Custom reducer expression

        Returns:
            Reduced value
        """
        if not collection:
            return initial_value if initial_value is not None else 0

        if operation == "sum":
            return sum(collection, initial_value or 0)

        elif operation == "average":
            total = sum(collection, initial_value or 0)
            return total / len(collection)

        elif operation == "min":
            return min(collection)

        elif operation == "max":
            return max(collection)

        elif operation == "count":
            return len(collection)

        elif operation == "custom":
            if not custom_reducer:
                raise ValueError("Custom reduce operation requires 'custom_reducer'")

            # Custom reducer: accumulator and item variables
            accumulator = initial_value if initial_value is not None else collection[0]
            start_idx = 0 if initial_value is not None else 1

            for item in collection[start_idx:]:
                try:
                    accumulator = self.evaluator.safe_eval(
                        custom_reducer,
                        {
                            "accumulator": accumulator,
                            "acc": accumulator,  # Short alias
                            "item": item,
                            **self.variable_context.get_all_variables(),
                        },
                    )
                except Exception as e:
                    logger.warning(f"Custom reducer failed for item: {e}")
                    # Continue with current accumulator value

            return accumulator

        else:
            raise ValueError(f"Unknown reduce operation: {operation}")

    def execute_string_operation(self, action: Action, context: dict) -> dict:
        """Execute STRING_OPERATION action.

        Performs string manipulations like concat, substring, replace, etc.

        Args:
            action: Action configuration
            context: Execution context

        Returns:
            Dictionary with result string
        """
        try:
            config: StringOperationActionConfig = get_typed_config(action)

            logger.info(f"Executing string operation: {config.operation}")

            # Get input string(s)
            if isinstance(config.input, str):
                input_str = config.input
            elif isinstance(config.input, dict):
                # Variable reference
                var_name = config.input.get("variableName")
                if not var_name:
                    raise ValueError("Variable input requires 'variableName'")
                input_str = self.variable_context.get(var_name)
                if input_str is None:
                    raise ValueError(f"Variable '{var_name}' not found")
            else:
                input_str = str(config.input)

            # Perform operation
            result_str = self._perform_string_operation(
                input_str, config.operation, config.parameters
            )

            # Store result
            if config.output_variable:
                self.variable_context.set(config.output_variable, result_str)
                context[config.output_variable] = result_str

            result = {"success": True, "result": result_str, "operation": config.operation}

            logger.info(f"String operation completed: {config.operation}")
            return result

        except Exception as e:
            logger.error(f"STRING_OPERATION failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "result": None}

    def _perform_string_operation(self, input_str: str, operation: str, parameters) -> str:
        """Perform a string operation.

        Args:
            input_str: Input string
            operation: Operation type
            parameters: Operation parameters

        Returns:
            Result string
        """
        input_str = str(input_str)

        if operation == "CONCAT":
            # Concatenate with other strings
            if not parameters or not parameters.strings:
                return input_str
            return input_str + "".join(parameters.strings)

        elif operation == "SUBSTRING":
            # Extract substring
            if not parameters:
                return input_str
            start = parameters.start or 0
            end = parameters.end
            return input_str[start:end] if end else input_str[start:]

        elif operation == "REPLACE":
            # Replace text
            if not parameters or not parameters.search:
                return input_str
            search = parameters.search
            replacement = parameters.replacement or ""
            return input_str.replace(search, replacement)

        elif operation == "SPLIT":
            # Split string (returns list as JSON string)
            if not parameters:
                return json.dumps(input_str.split())
            delimiter = parameters.delimiter or " "
            return json.dumps(input_str.split(delimiter))

        elif operation == "TRIM":
            # Trim whitespace
            return input_str.strip()

        elif operation == "UPPERCASE":
            return input_str.upper()

        elif operation == "LOWERCASE":
            return input_str.lower()

        elif operation == "MATCH":
            # Regex match (returns match object as JSON)
            if not parameters or not parameters.pattern:
                raise ValueError("MATCH operation requires 'pattern'")
            match = re.search(parameters.pattern, input_str)
            if match:
                return json.dumps(
                    {"matched": True, "groups": match.groups(), "group_dict": match.groupdict()}
                )
            return json.dumps({"matched": False})

        elif operation == "PARSE_JSON":
            # Parse JSON string
            try:
                parsed = json.loads(input_str)
                return json.dumps(parsed)  # Return normalized JSON
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {e}") from e

        else:
            raise ValueError(f"Unknown string operation: {operation}")

    def execute_math_operation(self, action: Action, context: dict) -> dict:
        """Execute MATH_OPERATION action.

        Performs mathematical operations on operands.

        Args:
            action: Action configuration
            context: Execution context

        Returns:
            Dictionary with result value
        """
        try:
            config: MathOperationActionConfig = get_typed_config(action)

            logger.info(f"Executing math operation: {config.operation}")

            # Resolve operands (may be values or variable references)
            resolved_operands = []
            for operand in config.operands:
                if isinstance(operand, int | float):
                    resolved_operands.append(operand)
                elif isinstance(operand, dict):
                    # Variable reference
                    var_name = operand.get("variableName")
                    if not var_name:
                        raise ValueError("Variable operand requires 'variableName'")
                    value = self.variable_context.get(var_name)
                    if value is None:
                        raise ValueError(f"Variable '{var_name}' not found")
                    resolved_operands.append(float(value))
                else:
                    resolved_operands.append(float(operand))

            # Perform operation
            result_value = self._perform_math_operation(
                config.operation, resolved_operands, config.custom_expression
            )

            # Store result
            if config.output_variable:
                self.variable_context.set(config.output_variable, result_value)
                context[config.output_variable] = result_value

            result = {"success": True, "result": result_value, "operation": config.operation}

            logger.info(f"Math operation completed: {config.operation} = {result_value}")
            return result

        except Exception as e:
            logger.error(f"MATH_OPERATION failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "result": None}

    def _perform_math_operation(
        self, operation: str, operands: list[float], custom_expression: str | None
    ) -> float:
        """Perform a mathematical operation.

        Args:
            operation: Operation type
            operands: List of numeric operands
            custom_expression: Custom expression for CUSTOM operation

        Returns:
            Result value
        """
        if not operands:
            raise ValueError("Math operation requires at least one operand")

        if operation == "ADD":
            return sum(operands)

        elif operation == "SUBTRACT":
            if len(operands) < 2:
                raise ValueError("SUBTRACT requires at least 2 operands")
            result = operands[0]
            for val in operands[1:]:
                result -= val
            return result

        elif operation == "MULTIPLY":
            result = 1
            for val in operands:
                result *= val
            return result

        elif operation == "DIVIDE":
            if len(operands) < 2:
                raise ValueError("DIVIDE requires at least 2 operands")
            result = operands[0]
            for val in operands[1:]:
                if val == 0:
                    raise ValueError("Division by zero")
                result /= val
            return result

        elif operation == "MODULO":
            if len(operands) != 2:
                raise ValueError("MODULO requires exactly 2 operands")
            return operands[0] % operands[1]

        elif operation == "POWER":
            if len(operands) != 2:
                raise ValueError("POWER requires exactly 2 operands")
            return operands[0] ** operands[1]

        elif operation == "SQRT":
            if len(operands) != 1:
                raise ValueError("SQRT requires exactly 1 operand")
            import math

            return math.sqrt(operands[0])

        elif operation == "ABS":
            if len(operands) != 1:
                raise ValueError("ABS requires exactly 1 operand")
            return abs(operands[0])

        elif operation == "ROUND":
            if len(operands) < 1 or len(operands) > 2:
                raise ValueError("ROUND requires 1 or 2 operands")
            if len(operands) == 1:
                return round(operands[0])
            else:
                return round(operands[0], int(operands[1]))

        elif operation == "CUSTOM":
            if not custom_expression:
                raise ValueError("CUSTOM operation requires 'customExpression'")

            # Create context with operands
            eval_context = {
                "operands": operands,
                **{f"op{i}": val for i, val in enumerate(operands)},
                **self.variable_context.get_all_variables(),
            }

            result = self.evaluator.safe_eval(custom_expression, eval_context)
            return float(result)

        else:
            raise ValueError(f"Unknown math operation: {operation}")

    def _compare_values(self, left: Any, right: Any, op_str: str | None) -> bool:
        """Compare two values using an operator.

        Args:
            left: Left operand
            right: Right operand
            op_str: Operator string

        Returns:
            Comparison result
        """
        if op_str is None:
            op_str = "=="

        try:
            op_enum = ComparisonOperator(op_str)
        except ValueError as err:
            raise ValueError(f"Invalid operator: {op_str}") from err

        if op_enum == ComparisonOperator.EQUAL:
            return left == right
        elif op_enum == ComparisonOperator.NOT_EQUAL:
            return left != right
        elif op_enum == ComparisonOperator.GREATER:
            return left > right
        elif op_enum == ComparisonOperator.LESS:
            return left < right
        elif op_enum == ComparisonOperator.GREATER_EQUAL:
            return left >= right
        elif op_enum == ComparisonOperator.LESS_EQUAL:
            return left <= right
        elif op_enum == ComparisonOperator.CONTAINS:
            return right in left if hasattr(left, "__contains__") else False
        elif op_enum == ComparisonOperator.MATCHES:
            # Regex match
            if isinstance(left, str) and isinstance(right, str):
                return bool(re.search(right, left))
            return False
        else:
            raise ValueError(f"Unsupported operator: {op_enum}")


# ============================================================================
# Module-level convenience functions
# ============================================================================

# Global executor instance
_global_executor: DataOperationsExecutor | None = None


def get_data_operations_executor() -> DataOperationsExecutor:
    """Get the global data operations executor instance.

    Returns:
        Global DataOperationsExecutor instance
    """
    global _global_executor
    if _global_executor is None:
        _global_executor = DataOperationsExecutor()
    return _global_executor


def execute_set_variable(action: Action, context: dict) -> dict:
    """Convenience function to execute SET_VARIABLE action.

    Args:
        action: Action configuration
        context: Execution context

    Returns:
        Result dictionary
    """
    return get_data_operations_executor().execute_set_variable(action, context)


def execute_get_variable(action: Action, context: dict) -> dict:
    """Convenience function to execute GET_VARIABLE action.

    Args:
        action: Action configuration
        context: Execution context

    Returns:
        Result dictionary
    """
    return get_data_operations_executor().execute_get_variable(action, context)


def execute_sort(action: Action, context: dict) -> dict:
    """Convenience function to execute SORT action.

    Args:
        action: Action configuration
        context: Execution context

    Returns:
        Result dictionary
    """
    return get_data_operations_executor().execute_sort(action, context)


def execute_filter(action: Action, context: dict) -> dict:
    """Convenience function to execute FILTER action.

    Args:
        action: Action configuration
        context: Execution context

    Returns:
        Result dictionary
    """
    return get_data_operations_executor().execute_filter(action, context)


def execute_map(action: Action, context: dict) -> dict:
    """Convenience function to execute MAP action.

    Args:
        action: Action configuration
        context: Execution context

    Returns:
        Result dictionary
    """
    return get_data_operations_executor().execute_map(action, context)


def execute_reduce(action: Action, context: dict) -> dict:
    """Convenience function to execute REDUCE action.

    Args:
        action: Action configuration
        context: Execution context

    Returns:
        Result dictionary
    """
    return get_data_operations_executor().execute_reduce(action, context)


def execute_string_operation(action: Action, context: dict) -> dict:
    """Convenience function to execute STRING_OPERATION action.

    Args:
        action: Action configuration
        context: Execution context

    Returns:
        Result dictionary
    """
    return get_data_operations_executor().execute_string_operation(action, context)


def execute_math_operation(action: Action, context: dict) -> dict:
    """Convenience function to execute MATH_OPERATION action.

    Args:
        action: Action configuration
        context: Execution context

    Returns:
        Result dictionary
    """
    return get_data_operations_executor().execute_math_operation(action, context)
