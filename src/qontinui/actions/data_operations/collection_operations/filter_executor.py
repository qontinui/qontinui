"""Filter operations executor for collections.

This module provides the FilterExecutor class for filtering collections
based on various condition types.
"""

import logging
import re
from typing import Any, cast

from ..constants import ComparisonOperator
from ..context import VariableContext
from ..evaluator import SafeEvaluator

logger = logging.getLogger(__name__)


class FilterExecutor:
    """Executor for filtering collections.

    Handles filtering operations with support for:
    - Expression-based conditions (Python expressions)
    - Property-based conditions (compare item properties)
    - Custom function conditions
    - Multiple comparison operators (==, !=, >, <, >=, <=, contains, matches)

    Example:
        >>> from types import SimpleNamespace
        >>> context = VariableContext()
        >>> evaluator = SafeEvaluator()
        >>> executor = FilterExecutor(context, evaluator)
        >>> condition = SimpleNamespace(
        ...     type="property", property="age", operator=">=", value=30
        ... )
        >>> data = [{"age": 25}, {"age": 35}, {"age": 40}]
        >>> filtered = executor.filter_collection(data, condition)
        >>> [item["age"] for item in filtered]
        [35, 40]
    """

    def __init__(self, variable_context: VariableContext, evaluator: SafeEvaluator) -> None:
        """Initialize the filter executor.

        Args:
            variable_context: Variable context for accessing variables in expressions
            evaluator: Safe evaluator for custom expressions
        """
        self.variable_context = variable_context
        self.evaluator = evaluator
        logger.debug("Initialized FilterExecutor")

    def filter_collection(self, collection: list[Any], condition: Any) -> list[Any]:
        """Filter a collection using specified condition.

        Supports three condition types:
        - expression: Python expression with 'item' variable
        - property: Compare item property to value using operator
        - custom: Custom function expression

        Args:
            collection: Collection to filter
            condition: Filter condition configuration with attributes:
                - type: Condition type (expression, property, custom)
                - expression: Expression to evaluate (for type=expression)
                - property: Property name (for type=property)
                - value: Expected value (for type=property)
                - operator: Comparison operator (for type=property)
                - custom_function: Custom function expression (for type=custom)

        Returns:
            New list containing only items matching the condition

        Example:
            >>> from types import SimpleNamespace
            >>> condition = SimpleNamespace(
            ...     type="property", property="age", operator=">=", value=30
            ... )
            >>> data = [{"age": 25}, {"age": 35}, {"age": 40}]
            >>> filtered = executor.filter_collection(data, condition)
            >>> [item["age"] for item in filtered]
            [35, 40]
        """
        if not collection:
            logger.debug("Empty collection, nothing to filter")
            return collection

        logger.debug(
            f"Filtering collection of {len(collection)} items (condition_type={condition.type})"
        )

        filtered = []

        for item in collection:
            try:
                if self._evaluate_filter_condition(item, condition):
                    filtered.append(item)
            except Exception as e:
                logger.warning(f"Filter condition evaluation failed for item: {e}")
                # Skip items that fail evaluation
                continue

        logger.debug(f"Filtered {len(collection)} items to {len(filtered)} items")
        return filtered

    def _evaluate_filter_condition(self, item: Any, condition: Any) -> bool:
        """Evaluate a filter condition for an item.

        Args:
            item: Item to evaluate
            condition: Filter condition with type and parameters

        Returns:
            True if item matches condition, False otherwise

        Raises:
            ValueError: If condition type is unknown or required parameters are missing
        """
        condition_type = condition.type

        if condition_type == "expression":
            return self._evaluate_expression_condition(item, condition)

        elif condition_type == "property":
            return self._evaluate_property_condition(item, condition)

        elif condition_type == "custom":
            return self._evaluate_custom_condition(item, condition)

        else:
            raise ValueError(f"Unknown condition type: {condition_type}")

    def _evaluate_expression_condition(self, item: Any, condition: Any) -> bool:
        """Evaluate expression-based filter condition.

        Args:
            item: Item to evaluate
            condition: Condition with 'expression' attribute

        Returns:
            Boolean result of expression evaluation

        Raises:
            ValueError: If expression is not provided
        """
        if not condition.expression:
            raise ValueError("Expression condition requires 'expression'")

        result = self.evaluator.safe_eval(
            condition.expression,
            {"item": item, **self.variable_context.get_all_variables()},
        )
        return bool(result)

    def _evaluate_property_condition(self, item: Any, condition: Any) -> bool:
        """Evaluate property-based filter condition.

        Extracts property from item and compares to expected value using operator.

        Args:
            item: Item to evaluate
            condition: Condition with 'property', 'value', and 'operator' attributes

        Returns:
            Boolean result of property comparison

        Raises:
            ValueError: If property is not provided
        """
        if not condition.property:
            raise ValueError("Property condition requires 'property'")

        # Extract property value
        item_value = self._extract_item_property(item, condition.property)

        # Compare using operator
        return self._compare_values(item_value, condition.value, condition.operator)

    def _evaluate_custom_condition(self, item: Any, condition: Any) -> bool:
        """Evaluate custom function filter condition.

        Args:
            item: Item to evaluate
            condition: Condition with 'custom_function' attribute

        Returns:
            Boolean result of custom function evaluation

        Raises:
            ValueError: If custom_function is not provided
        """
        if not condition.custom_function:
            raise ValueError("Custom condition requires 'custom_function'")

        result = self.evaluator.safe_eval(
            condition.custom_function,
            {"item": item, **self.variable_context.get_all_variables()},
        )
        return bool(result)

    def _extract_item_property(self, item: Any, property_name: str) -> Any:
        """Extract a property value from an item.

        Supports both dictionary access and attribute access.

        Args:
            item: Item to extract from
            property_name: Name of property to extract

        Returns:
            Property value or None if not found
        """
        if isinstance(item, dict):
            return item.get(property_name)
        else:
            return getattr(item, property_name, None)

    def _compare_values(self, left: Any, right: Any, operator: str | None) -> bool:
        """Compare two values using an operator.

        Supports operators:
        - ==, !=, >, <, >=, <=: Standard comparisons
        - contains: Check if right is contained in left
        - matches: Regex match (left must be string)

        Args:
            left: Left operand
            right: Right operand
            operator: Comparison operator (defaults to "==")

        Returns:
            Boolean result of comparison

        Raises:
            ValueError: If operator is invalid or unsupported

        Example:
            >>> executor._compare_values(5, 3, ">")
            True
            >>> executor._compare_values("hello", "ell", "contains")
            True
            >>> executor._compare_values("test123", r"test\\d+", "matches")
            True
        """
        if operator is None:
            operator = "=="

        try:
            op_enum = ComparisonOperator(operator)
        except ValueError as err:
            raise ValueError(f"Invalid operator: {operator}") from err

        if op_enum == ComparisonOperator.EQUAL:
            return cast(bool, left == right)
        elif op_enum == ComparisonOperator.NOT_EQUAL:
            return cast(bool, left != right)
        elif op_enum == ComparisonOperator.GREATER:
            return cast(bool, left > right)
        elif op_enum == ComparisonOperator.LESS:
            return cast(bool, left < right)
        elif op_enum == ComparisonOperator.GREATER_EQUAL:
            return cast(bool, left >= right)
        elif op_enum == ComparisonOperator.LESS_EQUAL:
            return cast(bool, left <= right)
        elif op_enum == ComparisonOperator.CONTAINS:
            return right in left if hasattr(left, "__contains__") else False
        elif op_enum == ComparisonOperator.MATCHES:
            # Regex match
            if isinstance(left, str) and isinstance(right, str):
                return bool(re.search(right, left))
            return False
        else:
            raise ValueError(f"Unsupported operator: {op_enum}")
