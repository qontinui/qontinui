"""Data operations executor - re-export from modular structure.

This module provides backward-compatible access to the refactored data operations
functionality. The original monolithic DataOperationsExecutor has been split into
specialized executors:

- VariableExecutor: Variable get/set operations
- CollectionExecutor: Sort, filter, map, reduce operations
- StringExecutor: String manipulation operations
- MathExecutor: Mathematical operations

The DataOperationsExecutor class here wraps these specialized executors to maintain
API compatibility with existing code.

Example:
    >>> from qontinui.actions.data_operations import DataOperationsExecutor, VariableContext
    >>> context = VariableContext()
    >>> executor = DataOperationsExecutor(context)
    >>> result = executor.execute_set_variable(action, {})
"""

import logging
from typing import Any

from qontinui.config import Action

from .coercer import TypeCoercer
from .collection_executor import CollectionExecutor
from .context import VariableContext
from .evaluator import SafeEvaluator
from .math_executor import MathExecutor
from .string_executor import StringExecutor
from .variable_executor import VariableExecutor

logger = logging.getLogger(__name__)

__all__ = ["DataOperationsExecutor", "VariableContext"]


class DataOperationsExecutor:
    """Unified executor for data operations (backward compatibility wrapper).

    This class maintains the original API by delegating to specialized executors:
    - VariableExecutor: execute_set_variable, execute_get_variable
    - CollectionExecutor: execute_sort, execute_filter, execute_map, execute_reduce
    - StringExecutor: execute_string_operation
    - MathExecutor: execute_math_operation

    The executor accepts a VariableContext and creates the necessary specialized
    executors internally.

    Args:
        variable_context: Variable context for multi-scope variable storage.
                         If None, a new context will be created.

    Example:
        >>> context = VariableContext()
        >>> executor = DataOperationsExecutor(context)
        >>> result = executor.execute_set_variable(action, {})
    """

    def __init__(self, variable_context: VariableContext | None = None) -> None:
        """Initialize the data operations executor.

        Args:
            variable_context: Variable context to use (creates new if None)
        """
        self.variable_context = variable_context or VariableContext()

        # Create shared components
        evaluator = SafeEvaluator()
        coercer = TypeCoercer()

        # Initialize specialized executors
        self._variable_executor = VariableExecutor(self.variable_context, evaluator, coercer)
        self._collection_executor = CollectionExecutor(self.variable_context, evaluator)
        self._string_executor = StringExecutor(self.variable_context)
        self._math_executor = MathExecutor(self.variable_context, evaluator)

        logger.info("Initialized DataOperationsExecutor with specialized executors")

    # ============================================================================
    # Variable Operations
    # ============================================================================

    def execute_set_variable(self, action: Action, context: dict) -> dict:
        """Execute SET_VARIABLE action.

        Delegates to VariableExecutor.

        Args:
            action: Action configuration
            context: Execution context

        Returns:
            Dictionary with variable info
        """
        return self._variable_executor.execute_set_variable(action, context)

    def execute_get_variable(self, action: Action, context: dict) -> dict:
        """Execute GET_VARIABLE action.

        Delegates to VariableExecutor.

        Args:
            action: Action configuration
            context: Execution context

        Returns:
            Dictionary with variable value
        """
        return self._variable_executor.execute_get_variable(action, context)

    # ============================================================================
    # Collection Operations
    # ============================================================================

    def execute_sort(self, action: Action, context: dict) -> dict:
        """Execute SORT action.

        Delegates to CollectionExecutor.

        Args:
            action: Action configuration
            context: Execution context

        Returns:
            Dictionary with sorted collection
        """
        from qontinui.config import SortActionConfig, get_typed_config

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
                collection = context.get("matches", [])
                if not collection:
                    logger.warning("No matches found in context")

            elif config.target == "list":
                collection = context.get("list", [])

            else:
                raise ValueError(f"Unknown sort target: {config.target}")

            if not isinstance(collection, list | tuple):
                raise ValueError(f"Cannot sort non-collection type: {type(collection)}")

            if len(collection) == 0:
                logger.warning("Empty collection, nothing to sort")
                sorted_collection = []
            else:
                # Perform sort using CollectionExecutor
                sorted_collection = self._collection_executor.sort_collection(
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

        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"SORT failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "sorted_collection": []}

    def execute_filter(self, action: Action, context: dict) -> dict:
        """Execute FILTER action.

        Delegates to CollectionExecutor.

        Args:
            action: Action configuration
            context: Execution context

        Returns:
            Dictionary with filtered collection
        """
        from qontinui.config import FilterActionConfig, get_typed_config

        try:
            config: FilterActionConfig = get_typed_config(action)
            logger.info(f"Filtering collection from '{config.variable_name}'")

            # Get collection to filter
            collection = self.variable_context.get(config.variable_name)

            if collection is None:
                raise ValueError(f"Variable '{config.variable_name}' not found")

            if not isinstance(collection, list | tuple):
                raise ValueError(f"Cannot filter non-collection type: {type(collection)}")

            # Perform filter using CollectionExecutor
            filtered_collection = self._collection_executor.filter_collection(
                list(collection), config.condition
            )

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

        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"FILTER failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "filtered_collection": []}

    def execute_map(self, action: Action, context: dict) -> dict:
        """Execute MAP action.

        Delegates to CollectionExecutor.

        Args:
            action: Action configuration
            context: Execution context

        Returns:
            Dictionary with transformed collection
        """
        from qontinui.config import MapActionConfig, get_typed_config

        try:
            config: MapActionConfig = get_typed_config(action)
            logger.info(f"Mapping over collection from '{config.variable_name}'")

            # Get collection to map
            collection = self.variable_context.get(config.variable_name)

            if collection is None:
                raise ValueError(f"Variable '{config.variable_name}' not found")

            if not isinstance(collection, list | tuple):
                raise ValueError(f"Cannot map over non-collection type: {type(collection)}")

            # Perform map using CollectionExecutor
            mapped_collection = self._collection_executor.map_collection(
                list(collection), config.transform
            )

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

        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"MAP failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "mapped_collection": []}

    def execute_reduce(self, action: Action, context: dict) -> dict:
        """Execute REDUCE action.

        Delegates to CollectionExecutor.

        Args:
            action: Action configuration
            context: Execution context

        Returns:
            Dictionary with reduced value
        """
        from qontinui.config import ReduceActionConfig, get_typed_config

        try:
            config: ReduceActionConfig = get_typed_config(action)
            logger.info(f"Reducing collection from '{config.variable_name}'")

            # Get collection to reduce
            collection = self.variable_context.get(config.variable_name)

            if collection is None:
                raise ValueError(f"Variable '{config.variable_name}' not found")

            if not isinstance(collection, list | tuple):
                raise ValueError(f"Cannot reduce non-collection type: {type(collection)}")

            # Perform reduce using CollectionExecutor
            reduced_value = self._collection_executor.reduce_collection(
                list(collection),
                config.operation,
                config.initial_value,
                config.custom_reducer,
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

        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"REDUCE failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "reduced_value": None}

    # ============================================================================
    # String Operations
    # ============================================================================

    def execute_string_operation(self, action: Action, context: dict) -> dict:
        """Execute STRING_OPERATION action.

        Delegates to StringExecutor.

        Args:
            action: Action configuration
            context: Execution context

        Returns:
            Dictionary with result string
        """
        from qontinui.config import StringOperationActionConfig, get_typed_config

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

            # Perform operation using StringExecutor
            result_str = self._string_executor.execute(
                config.operation, input_str, config.parameters
            )

            # Store result
            if config.output_variable:
                self.variable_context.set(config.output_variable, result_str)
                context[config.output_variable] = result_str

            result = {"success": True, "result": result_str, "operation": config.operation}

            logger.info(f"String operation completed: {config.operation}")
            return result

        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"STRING_OPERATION failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "result": None}

    # ============================================================================
    # Math Operations
    # ============================================================================

    def execute_math_operation(self, action: Action, context: dict) -> dict:
        """Execute MATH_OPERATION action.

        Delegates to MathExecutor.

        Args:
            action: Action configuration
            context: Execution context

        Returns:
            Dictionary with result value
        """
        from qontinui.config import MathOperationActionConfig, get_typed_config

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

            # Perform operation using MathExecutor
            result_value = self._math_executor.execute(
                config.operation, resolved_operands, config.custom_expression
            )

            # Store result
            if config.output_variable:
                self.variable_context.set(config.output_variable, result_value)
                context[config.output_variable] = result_value

            result = {"success": True, "result": result_value, "operation": config.operation}

            logger.info(f"Math operation completed: {config.operation} = {result_value}")
            return result

        except (ValueError, TypeError, AttributeError, ZeroDivisionError) as e:
            logger.error(f"MATH_OPERATION failed: {e}", exc_info=True)
            return {"success": False, "error": str(e), "result": None}
