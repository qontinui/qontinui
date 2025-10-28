"""Variable execution for GET_VARIABLE and SET_VARIABLE actions.

This module provides the VariableExecutor class that handles variable get/set
operations, including:
- Multiple value sources (direct, expression, clipboard, OCR, target)
- Type coercion
- Multi-scope storage (local, global, process)
- Safe expression evaluation
"""

import logging
from typing import Any

from qontinui.config import (
    Action,
    GetVariableActionConfig,
    SetVariableActionConfig,
    get_typed_config,
)

from .coercer import TypeCoercer
from .constants import VariableScope
from .context import VariableContext
from .evaluator import SafeEvaluator

# Optional clipboard support
try:
    import pyperclip

    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False

logger = logging.getLogger(__name__)


class VariableExecutor:
    """Executor for variable get/set operations.

    Handles SET_VARIABLE and GET_VARIABLE actions with support for multiple
    value sources, type coercion, and multi-scope variable management.

    This class is standalone (not inheriting from ActionExecutorBase) and
    accepts its dependencies through constructor injection.

    Example:
        >>> context = VariableContext()
        >>> evaluator = SafeEvaluator()
        >>> coercer = TypeCoercer()
        >>> executor = VariableExecutor(context, evaluator, coercer)
        >>> result = executor.execute_set_variable(action, {})
    """

    def __init__(
        self,
        variable_context: VariableContext,
        evaluator: SafeEvaluator,
        coercer: TypeCoercer,
    ) -> None:
        """Initialize the variable executor.

        Args:
            variable_context: Variable context for multi-scope storage
            evaluator: Safe evaluator for expression evaluation
            coercer: Type coercer for value conversion
        """
        self.variable_context = variable_context
        self.evaluator = evaluator
        self.coercer = coercer
        logger.debug("Initialized VariableExecutor")

    def execute_set_variable(self, action: Action, context: dict[str, Any]) -> dict[str, Any]:
        """Execute SET_VARIABLE action.

        Sets a variable from various sources:
        - Direct value: Explicit value in config
        - Expression: Evaluated Python expression
        - Clipboard: Contents from system clipboard (requires pyperclip)
        - OCR: Text extracted via OCR (placeholder for future)
        - Target: Value from screen target extraction (placeholder for future)

        Args:
            action: Action configuration
            context: Execution context dictionary (updated with variable)

        Returns:
            Dictionary with:
                - success (bool): True if operation succeeded
                - variable_name (str): Name of the variable
                - value (Any): The set value
                - scope (str): Scope where variable was set
                - type (str): Python type name of the value
                - error (str): Error message if success=False

        Raises:
            No exceptions raised - errors are captured in return dict

        Example:
            >>> action = Action(type="SET_VARIABLE", config={
            ...     "variableName": "count",
            ...     "value": 42,
            ...     "scope": "local"
            ... })
            >>> result = executor.execute_set_variable(action, {})
            >>> result["success"]
            True
        """
        try:
            config: SetVariableActionConfig = get_typed_config(action)

            logger.info(f"Setting variable '{config.variable_name}'")

            # Determine value from source
            value = self._get_value_from_source(config)

            # Type coercion if specified
            if config.type:
                value = self.coercer.coerce(value, config.type)

            # Store in appropriate scope
            scope = config.scope or "local"
            self.variable_context.set(config.variable_name, value, scope)

            # Store in context dict for backwards compatibility
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

        except (ValueError, NotImplementedError) as e:
            logger.error(f"SET_VARIABLE failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "variable_name": getattr(config, "variable_name", "unknown"),
            }

    def execute_get_variable(self, action: Action, context: dict[str, Any]) -> dict[str, Any]:
        """Execute GET_VARIABLE action.

        Retrieves a variable value from any scope (local -> process -> global)
        and optionally stores it in another variable.

        Args:
            action: Action configuration
            context: Execution context dictionary (updated if output_variable set)

        Returns:
            Dictionary with:
                - success (bool): True if operation succeeded
                - variable_name (str): Name of the requested variable
                - value (Any): The retrieved value (or default)
                - found (bool): True if variable exists in any scope
                - error (str): Error message if success=False

        Raises:
            No exceptions raised - errors are captured in return dict

        Example:
            >>> action = Action(type="GET_VARIABLE", config={
            ...     "variableName": "count",
            ...     "defaultValue": 0,
            ...     "outputVariable": "current_count"
            ... })
            >>> result = executor.execute_get_variable(action, {})
            >>> result["success"]
            True
        """
        try:
            config: GetVariableActionConfig = get_typed_config(action)

            logger.info(f"Getting variable '{config.variable_name}'")

            # Get variable value with optional default
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

        except (ValueError, AttributeError) as e:
            logger.error(f"GET_VARIABLE failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "variable_name": getattr(config, "variable_name", "unknown"),
            }

    def _get_value_from_source(self, config: SetVariableActionConfig) -> Any:
        """Get value from the configured source.

        Handles multiple value sources with priority:
        1. Direct value (config.value)
        2. Value source (expression, clipboard, OCR, target)

        Args:
            config: SET_VARIABLE action configuration

        Returns:
            Value from the appropriate source

        Raises:
            ValueError: If no valid source specified or source configuration invalid
            NotImplementedError: If OCR or target source requested
        """
        # Direct value has priority
        if config.value is not None:
            logger.debug(f"Using direct value: {config.value}")
            return config.value

        # Check value source
        if config.value_source:
            source_type = config.value_source.type

            if source_type == "expression":
                return self._get_value_from_expression(config)
            elif source_type == "clipboard":
                return self._get_value_from_clipboard()
            elif source_type == "ocr":
                return self._get_value_from_ocr(config)
            elif source_type == "target":
                return self._get_value_from_target(config)
            else:
                raise ValueError(f"Unknown value source type: {source_type}")

        raise ValueError("SET_VARIABLE requires either 'value' or 'value_source'")

    def _get_value_from_expression(self, config: SetVariableActionConfig) -> Any:
        """Get value by evaluating an expression.

        Args:
            config: Action configuration with expression in value_source

        Returns:
            Result of expression evaluation

        Raises:
            ValueError: If expression is missing or evaluation fails
        """
        expression = config.value_source.expression
        if not expression:
            raise ValueError("Expression source requires 'expression' field")

        value = self.evaluator.safe_eval(expression, self.variable_context.get_all_variables())
        logger.debug(f"Evaluated expression to: {value}")
        return value

    def _get_value_from_clipboard(self) -> str:
        """Get value from system clipboard.

        Returns:
            Clipboard contents as string

        Raises:
            ValueError: If pyperclip is not available
        """
        if not CLIPBOARD_AVAILABLE:
            raise ValueError("Clipboard access requires 'pyperclip' package")

        value = pyperclip.paste()
        # Truncate for logging
        value_preview = value[:50] + "..." if len(value) > 50 else value
        logger.debug(f"Read from clipboard: {value_preview}")
        return value

    def _get_value_from_ocr(self, config: SetVariableActionConfig) -> Any:
        """Get value from OCR extraction (placeholder).

        This is a placeholder for future integration with OCR engine.

        Args:
            config: Action configuration

        Returns:
            Never returns

        Raises:
            NotImplementedError: Always (feature not yet implemented)
        """
        raise NotImplementedError(
            "OCR value source not yet implemented. "
            "Requires integration with OCR engine."
        )

    def _get_value_from_target(self, config: SetVariableActionConfig) -> Any:
        """Get value from screen target extraction (placeholder).

        This is a placeholder for future integration with find/match operations.

        Args:
            config: Action configuration

        Returns:
            Never returns

        Raises:
            NotImplementedError: Always (feature not yet implemented)
        """
        raise NotImplementedError(
            "Target value source not yet implemented. "
            "Requires integration with find operations."
        )
