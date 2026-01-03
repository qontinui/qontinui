"""Data operations action executor adapter.

This module provides an adapter that integrates the existing DataOperationsExecutor
into the new registry-based action executor system. It wraps the legacy executor
while conforming to the ActionExecutorBase interface.
"""

import logging
from typing import Any

from ..actions.data_operations import DataOperationsExecutor
from ..config.schema import Action
from ..exceptions import ActionExecutionError
from .base import ActionExecutorBase, ExecutionContext
from .registry import register_executor

logger = logging.getLogger(__name__)


@register_executor
class DataOperationsExecutorAdapter(ActionExecutorBase):
    """Adapter for integrating DataOperationsExecutor into the registry system.

    This adapter wraps the existing DataOperationsExecutor and bridges the gap
    between the old execution model (action + context dict) and the new model
    (ActionExecutorBase + ExecutionContext).

    Supported action types:
    - SET_VARIABLE: Set variables from various sources
    - GET_VARIABLE: Retrieve variable values
    - MAP: Transform each item in a collection
    - REDUCE: Reduce a collection to a single value
    - SORT: Sort collections by properties
    - FILTER: Filter collections by conditions
    - STRING_OPERATION: String manipulations (concat, substring, replace, etc.)
    - MATH_OPERATION: Mathematical operations on operands
    """

    def __init__(self, context: ExecutionContext) -> None:
        """Initialize the adapter with execution context.

        Args:
            context: Execution context containing dependencies
        """
        super().__init__(context)

        # Wrap the existing DataOperationsExecutor using the variable context
        # from the ExecutionContext
        self._wrapped_executor = DataOperationsExecutor(variable_context=context.variable_context)

        logger.debug("Initialized DataOperationsExecutorAdapter")

    def get_supported_action_types(self) -> list[str]:
        """Get list of data operation action types this executor handles.

        Returns:
            List of action type strings
        """
        return [
            "SET_VARIABLE",
            "GET_VARIABLE",
            "MAP",
            "REDUCE",
            "SORT",
            "FILTER",
            "STRING_OPERATION",
            "MATH_OPERATION",
        ]

    async def execute(self, action: Action, typed_config: Any) -> bool:
        """Execute data operation action with validated configuration.

        This method delegates to the wrapped DataOperationsExecutor, adapting
        between the new execution interface and the legacy interface.

        Args:
            action: Pydantic Action model
            typed_config: Type-specific validated configuration object (unused,
                         as wrapped executor uses action.config directly)

        Returns:
            bool: True if action succeeded

        Raises:
            ActionExecutionError: If action execution fails critically
        """
        action_type = action.type
        logger.debug(f"Executing data operation: {action_type}")

        try:
            # Execute via the wrapped executor
            context: dict[str, Any] = {}

            if action_type == "SET_VARIABLE":
                result = self._wrapped_executor.execute_set_variable(action, context)
            elif action_type == "GET_VARIABLE":
                result = self._wrapped_executor.execute_get_variable(action, context)
            elif action_type == "MAP":
                result = self._wrapped_executor.execute_map(action, context)
            elif action_type == "REDUCE":
                result = self._wrapped_executor.execute_reduce(action, context)
            elif action_type == "SORT":
                result = self._wrapped_executor.execute_sort(action, context)
            elif action_type == "FILTER":
                result = self._wrapped_executor.execute_filter(action, context)
            elif action_type == "STRING_OPERATION":
                result = self._wrapped_executor.execute_string_operation(action, context)
            elif action_type == "MATH_OPERATION":
                result = self._wrapped_executor.execute_math_operation(action, context)
            else:
                raise ActionExecutionError(
                    action_type=action_type,
                    reason=f"Unsupported action type: {action_type}",
                )

            # Check result and emit appropriate events
            success = result.get("success", False)

            if success:
                self._emit_action_success(action, result)
                logger.info(f"Data operation {action_type} succeeded")
            else:
                error_msg = result.get("error", "Unknown error")
                self._emit_action_failure(action, error_msg, result)
                logger.error(f"Data operation {action_type} failed: {error_msg}")

            return success  # type: ignore[no-any-return]

        except ActionExecutionError:
            # Re-raise ActionExecutionErrors as-is
            raise
        except Exception as e:
            # Wrap other exceptions in ActionExecutionError
            error_msg = f"Data operation {action_type} failed: {e}"
            logger.error(error_msg, exc_info=True)
            self._emit_action_failure(action, str(e))
            raise ActionExecutionError(action_type=action_type, reason=str(e)) from e
