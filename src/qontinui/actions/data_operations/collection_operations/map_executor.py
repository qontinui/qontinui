"""Map operations executor for collections.

This module provides the MapExecutor class for transforming collections
by applying operations to each item.
"""

import logging
from typing import Any

from ..context import VariableContext
from ..evaluator import SafeEvaluator

logger = logging.getLogger(__name__)


class MapExecutor:
    """Executor for mapping over collections.

    Handles transformation operations with support for:
    - Expression-based transforms (Python expressions)
    - Property extraction transforms
    - Custom function transforms

    Example:
        >>> from types import SimpleNamespace
        >>> context = VariableContext()
        >>> evaluator = SafeEvaluator()
        >>> executor = MapExecutor(context, evaluator)
        >>> transform = SimpleNamespace(type="expression", expression="item * 2")
        >>> data = [1, 2, 3, 4, 5]
        >>> mapped = executor.map_collection(data, transform)
        >>> mapped
        [2, 4, 6, 8, 10]
    """

    def __init__(
        self, variable_context: VariableContext, evaluator: SafeEvaluator
    ) -> None:
        """Initialize the map executor.

        Args:
            variable_context: Variable context for accessing variables in expressions
            evaluator: Safe evaluator for custom expressions
        """
        self.variable_context = variable_context
        self.evaluator = evaluator
        logger.debug("Initialized MapExecutor")

    def map_collection(self, collection: list[Any], transform: Any) -> list[Any]:
        """Map a collection using specified transform.

        Transforms each item in the collection using one of three transform types:
        - expression: Python expression with 'item' variable
        - property: Extract specific property from each item
        - custom: Custom function expression

        Args:
            collection: Collection to map
            transform: Transform configuration with attributes:
                - type: Transform type (expression, property, custom)
                - expression: Expression to evaluate (for type=expression)
                - property: Property name (for type=property)
                - custom_function: Custom function expression (for type=custom)

        Returns:
            New list with transformed items

        Example:
            >>> from types import SimpleNamespace
            >>> transform = SimpleNamespace(type="expression", expression="item * 2")
            >>> data = [1, 2, 3, 4, 5]
            >>> mapped = executor.map_collection(data, transform)
            >>> mapped
            [2, 4, 6, 8, 10]
        """
        if not collection:
            logger.debug("Empty collection, nothing to map")
            return collection

        logger.debug(
            f"Mapping collection of {len(collection)} items (transform_type={transform.type})"
        )

        mapped = []
        transform_type = transform.type

        for item in collection:
            try:
                if transform_type == "expression":
                    result = self._map_with_expression(item, transform)
                elif transform_type == "property":
                    result = self._map_with_property(item, transform)
                elif transform_type == "custom":
                    result = self._map_with_custom(item, transform)
                else:
                    raise ValueError(f"Unknown transform type: {transform_type}")

                mapped.append(result)

            except Exception as e:
                logger.warning(f"Map transform failed for item: {e}")
                # Add None for failed items
                mapped.append(None)

        logger.debug(f"Successfully mapped {len(mapped)} items")
        return mapped

    def _map_with_expression(self, item: Any, transform: Any) -> Any:
        """Transform item using expression.

        Args:
            item: Item to transform
            transform: Transform with 'expression' attribute

        Returns:
            Result of expression evaluation

        Raises:
            ValueError: If expression is not provided
        """
        if not transform.expression:
            raise ValueError("Expression transform requires 'expression'")

        return self.evaluator.safe_eval(
            transform.expression,
            {"item": item, **self.variable_context.get_all_variables()},
        )

    def _map_with_property(self, item: Any, transform: Any) -> Any:
        """Transform item by extracting property.

        Args:
            item: Item to extract from
            transform: Transform with 'property' attribute

        Returns:
            Property value

        Raises:
            ValueError: If property is not provided
        """
        if not transform.property:
            raise ValueError("Property transform requires 'property'")

        return self._extract_item_property(item, transform.property)

    def _map_with_custom(self, item: Any, transform: Any) -> Any:
        """Transform item using custom function.

        Args:
            item: Item to transform
            transform: Transform with 'custom_function' attribute

        Returns:
            Result of custom function evaluation

        Raises:
            ValueError: If custom_function is not provided
        """
        if not transform.custom_function:
            raise ValueError("Custom transform requires 'custom_function'")

        return self.evaluator.safe_eval(
            transform.custom_function,
            {"item": item, **self.variable_context.get_all_variables()},
        )

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
