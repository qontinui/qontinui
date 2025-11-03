"""Condition evaluation for control flow operations.

This module provides the ConditionEvaluator class which evaluates various types
of conditions used in control flow operations (IF, WHILE, BREAK, CONTINUE).
"""

import logging
import re
from typing import Any

from qontinui.config import ConditionConfig
from qontinui.orchestration.execution_context import ExecutionContext

logger = logging.getLogger(__name__)


class ConditionEvaluator:
    """Evaluates conditions for control flow operations.

    This class handles evaluation of multiple condition types including:
    - variable: Compare variable values with operators
    - expression: Evaluate Python expressions
    - image_exists: Check if an image is found on screen
    - image_vanished: Check if an image is NOT found on screen
    - text_exists: Check if text is found on screen

    The evaluator uses an ExecutionContext to access variables and maintain
    execution state during condition evaluation.

    Example:
        >>> context = ExecutionContext({"counter": 5})
        >>> evaluator = ConditionEvaluator(context)
        >>> condition = ConditionConfig(
        ...     type="variable",
        ...     variable_name="counter",
        ...     operator=">",
        ...     expected_value=3
        ... )
        >>> result = evaluator.evaluate_condition(condition)
        >>> print(result)  # True
    """

    def __init__(self, context: ExecutionContext) -> None:
        """Initialize the condition evaluator.

        Args:
            context: ExecutionContext providing variable access and state management
        """
        self.context = context
        logger.debug("ConditionEvaluator initialized with context")

    def evaluate_condition(self, condition: ConditionConfig) -> bool:
        """Evaluate a condition and return boolean result.

        Supports multiple condition types:
        - image_exists: Check if image is found on screen
        - image_vanished: Check if image is NOT found on screen
        - text_exists: Check if text is found on screen
        - variable: Compare variable value
        - expression: Evaluate Python expression

        Args:
            condition: Condition configuration specifying type and parameters

        Returns:
            True if condition is met, False otherwise

        Raises:
            ValueError: If condition type is unknown or configuration is invalid
        """
        logger.debug("Evaluating condition: type=%s", condition.type)

        if condition.type == "variable":
            return self._evaluate_variable_condition(condition)

        elif condition.type == "expression":
            return self._evaluate_expression_condition(condition)

        elif condition.type == "image_exists":
            return self._evaluate_image_exists_condition(condition)

        elif condition.type == "image_vanished":
            return not self._evaluate_image_exists_condition(condition)

        elif condition.type == "text_exists":
            return self._evaluate_text_exists_condition(condition)

        else:
            raise ValueError(f"Unknown condition type: {condition.type}")

    def _evaluate_variable_condition(self, condition: ConditionConfig) -> bool:
        """Evaluate a variable-based condition.

        Compares a variable's value against an expected value using a specified
        operator. Supports standard comparison operators and special operators
        like 'contains' and 'matches'.

        Args:
            condition: Condition configuration with variable_name, operator,
                      and expected_value

        Returns:
            Boolean result of the comparison

        Raises:
            ValueError: If variable_name is not specified
        """
        if not condition.variable_name:
            raise ValueError("Variable condition requires 'variable_name'")

        var_name = condition.variable_name
        if not self.context.has_variable(var_name):
            logger.warning("Variable '%s' not found in context, treating as None", var_name)
            var_value = None
        else:
            var_value = self.context.get_variable(var_name)

        expected = condition.expected_value
        operator = condition.operator or "=="

        logger.debug(
            "Variable condition: %s %s %s (actual=%s)", var_name, operator, expected, var_value
        )

        return self._compare_values(var_value, operator, expected)

    def _evaluate_expression_condition(self, condition: ConditionConfig) -> bool:
        """Evaluate a Python expression condition.

        Executes a Python expression in a restricted context with access to
        workflow variables. The expression should return a truthy/falsy value.

        SECURITY WARNING:
        This method uses eval() to execute Python expressions. It is designed for
        TRUSTED INPUT ONLY (automation scripts written by developers).

        DO NOT use this with:
        - User-provided input from web forms, APIs, or command line
        - Data from external/untrusted sources
        - Configuration files from untrusted locations

        Security mitigations in place:
        - Empty __builtins__ prevents access to dangerous functions
        - No access to __import__, open(), exec(), compile()
        - Expressions limited to variable references and basic operations
        - Intended for workflow conditions, not arbitrary code execution

        For untrusted scenarios:
        - Run Qontinui in isolated containers/VMs
        - Implement additional validation layers
        - Use alternative condition types (variable, image_exists)

        See docs/SECURITY.md for detailed security model.

        Args:
            condition: Condition configuration with expression string

        Returns:
            Boolean result of expression evaluation

        Raises:
            ValueError: If expression is not specified or evaluation fails
        """
        if not condition.expression:
            raise ValueError("Expression condition requires 'expression'")

        expression = condition.expression
        logger.debug("Evaluating expression: %s", expression)

        try:
            # Create safe evaluation context with variables
            # Include both namespaced and direct access to variables
            variables = self.context.variables
            eval_context = {"variables": variables, **variables}

            # Evaluate with restricted builtins for safety
            result = eval(expression, {"__builtins__": {}}, eval_context)
            logger.debug("Expression result: %s", result)
            return bool(result)

        except NameError as e:
            logger.error("Expression references undefined variable: %s", str(e))
            raise ValueError(f"Invalid expression '{expression}': {e}") from e

        except SyntaxError as e:
            logger.error("Expression has invalid syntax: %s", str(e))
            raise ValueError(f"Invalid expression '{expression}': {e}") from e

        except TypeError as e:
            logger.error("Expression type error: %s", str(e))
            raise ValueError(f"Invalid expression '{expression}': {e}") from e

        except ZeroDivisionError as e:
            logger.error("Expression division by zero: %s", str(e))
            raise ValueError(f"Invalid expression '{expression}': {e}") from e

    def _evaluate_image_exists_condition(self, condition: ConditionConfig) -> bool:
        """Evaluate image-exists condition using FindAction.

        Args:
            condition: Condition configuration with image_id

        Returns:
            True if image found, False otherwise

        Raises:
            ValueError: If image_id is not specified or image not found in registry
        """
        from ...actions.find import FindAction

        if not condition.image_id:
            raise ValueError("Image condition requires 'image_id'")

        logger.debug("Image exists check: image_id=%s", condition.image_id)

        # Import here to avoid circular dependencies
        from qontinui import registry

        # Get image and metadata from registry
        image = registry.get_image(condition.image_id)
        if image is None:
            error_msg = f"Image '{condition.image_id}' not found in registry"
            logger.error(error_msg)
            raise ValueError(error_msg)

        metadata = registry.get_image_metadata(condition.image_id)
        if metadata is None:
            error_msg = f"Image metadata for '{condition.image_id}' not found in registry"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Get file path from metadata
        file_path = metadata.get("file_path")
        if not file_path:
            error_msg = f"Image file_path for '{condition.image_id}' not found in metadata"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Create Pattern from file using factory method
        from ...model.element import Pattern

        pattern = Pattern.from_file(
            img_path=file_path,
            name=metadata.get("name", condition.image_id),
        )
        # Set similarity threshold for IF conditions
        pattern = pattern.with_similarity(0.8)

        # Use FindAction (single entry point)
        action = FindAction()
        exists = action.exists(pattern)

        logger.debug(
            "Image exists check result: image_id=%s, exists=%s",
            condition.image_id,
            exists,
        )

        return exists

    async def evaluate_multiple_image_exists_async(
        self,
        conditions: list[tuple[str, ConditionConfig]],
    ) -> dict[str, bool]:
        """Evaluate multiple image-exists conditions asynchronously.

        This method searches for multiple images in parallel for improved performance.
        Useful when checking if ANY of multiple state images are visible.

        Args:
            conditions: List of (action_id, condition_config) tuples to evaluate

        Returns:
            Dictionary mapping action_id to condition result (True/False)

        Raises:
            ValueError: If any image_id is missing or image not found in registry
        """

        from ...actions.find import FindAction
        from ...model.element import Pattern

        logger.debug(f"Async image exists check: {len(conditions)} conditions")

        # Import here to avoid circular dependencies
        from qontinui import registry

        # Build list of patterns to search for
        patterns = []
        action_ids = []
        image_ids = []

        for action_id, condition in conditions:
            if not condition.image_id:
                raise ValueError(f"Image condition requires 'image_id' (action_id={action_id})")

            image_ids.append(condition.image_id)
            action_ids.append(action_id)

            # Get image and metadata from registry
            image = registry.get_image(condition.image_id)
            if image is None:
                error_msg = f"Image '{condition.image_id}' not found in registry"
                logger.error(error_msg)
                raise ValueError(error_msg)

            metadata = registry.get_image_metadata(condition.image_id)
            if metadata is None:
                error_msg = f"Image metadata for '{condition.image_id}' not found in registry"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Get file path from metadata
            file_path = metadata.get("file_path")
            if not file_path:
                error_msg = f"Image file_path for '{condition.image_id}' not found in metadata"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Create Pattern from file
            pattern = Pattern.from_file(
                img_path=file_path,
                name=metadata.get("name", condition.image_id),
            )
            # Set similarity threshold for IF conditions
            pattern = pattern.with_similarity(0.8)
            patterns.append(pattern)

        # Use FindAction async method to search all patterns concurrently
        action = FindAction()
        results_dict = await action.exists_async(patterns, similarity=0.8)

        # Map results back to action IDs
        results = {}
        for _i, (action_id, image_id, pattern) in enumerate(
            zip(action_ids, image_ids, patterns, strict=False)
        ):
            exists = results_dict.get(pattern.name, False)
            results[action_id] = exists
            logger.debug(
                f"Image exists check result: action_id={action_id}, image_id={image_id}, exists={exists}"
            )

        return results

    def _evaluate_text_exists_condition(self, condition: ConditionConfig) -> bool:
        """Evaluate a text_exists condition.

        This is a placeholder implementation. In a full system, this would
        integrate with OCR capabilities to search for the specified text
        on screen.

        Args:
            condition: Condition configuration with text to search for

        Returns:
            True if text exists on screen, False otherwise

        Raises:
            ValueError: If text is not specified
        """
        if not condition.text:
            raise ValueError("Text condition requires 'text'")

        logger.debug("Text exists check: text=%s", condition.text)

        # Placeholder: OCR integration pending
        # Integration point: Use OCR action when implemented to search for text on screen
        # Requires OCR capabilities from perception module
        logger.warning("Text finding not implemented, returning False")
        return False

    def _compare_values(self, actual: Any, operator: str, expected: Any) -> bool:
        """Compare two values using the specified operator.

        Supports standard comparison operators (==, !=, >, <, >=, <=) as well
        as special operators:
        - contains: Check if expected is in actual (for strings/lists)
        - matches: Check if actual matches expected regex pattern

        Args:
            actual: Actual value to compare
            operator: Comparison operator string
            expected: Expected value to compare against

        Returns:
            Result of comparison as boolean

        Raises:
            ValueError: If operator is unknown
            TypeError: If comparison is invalid for the given types
        """
        try:
            if operator == "==":
                return actual == expected
            elif operator == "!=":
                return actual != expected
            elif operator == ">":
                return actual > expected
            elif operator == "<":
                return actual < expected
            elif operator == ">=":
                return actual >= expected
            elif operator == "<=":
                return actual <= expected
            elif operator == "contains":
                return expected in actual
            elif operator == "matches":
                pattern = str(expected)
                text = str(actual)
                return bool(re.match(pattern, text))
            else:
                raise ValueError(f"Unknown operator: {operator}")

        except TypeError as e:
            logger.error(
                "Type error comparing values: %s %s %s - %s", actual, operator, expected, str(e)
            )
            raise ValueError(
                f"Cannot compare {type(actual).__name__} and {type(expected).__name__} "
                f"with operator '{operator}'"
            ) from e
