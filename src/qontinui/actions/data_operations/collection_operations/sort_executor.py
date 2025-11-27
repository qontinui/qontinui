"""Sort operations executor for collections.

This module provides the SortExecutor class for sorting collections with
support for multiple comparator types and nested property access.
"""

import logging
from datetime import datetime
from typing import Any

from ..constants import ComparatorType
from ..context import VariableContext
from ..evaluator import SafeEvaluator

logger = logging.getLogger(__name__)


class SortExecutor:
    """Executor for sorting collections.

    Handles sorting operations with support for:
    - Multiple comparator types (numeric, alphabetic, date, custom)
    - Nested property access
    - Single or multiple sort keys
    - Ascending and descending order

    Example:
        >>> context = VariableContext()
        >>> evaluator = SafeEvaluator()
        >>> executor = SortExecutor(context, evaluator)
        >>> data = [{"price": "30"}, {"price": "100"}, {"price": "5"}]
        >>> sorted_data = executor.sort_collection(
        ...     data, sort_by="price", order="DESC", comparator="NUMERIC"
        ... )
        >>> [item["price"] for item in sorted_data]
        ["100", "30", "5"]
    """

    def __init__(self, variable_context: VariableContext, evaluator: SafeEvaluator) -> None:
        """Initialize the sort executor.

        Args:
            variable_context: Variable context for accessing variables in expressions
            evaluator: Safe evaluator for custom expressions
        """
        self.variable_context = variable_context
        self.evaluator = evaluator
        logger.debug("Initialized SortExecutor")

    def sort_collection(
        self,
        collection: list[Any],
        sort_by: str | list[str] | None,
        order: str = "ASC",
        comparator: str | None = None,
        custom_comparator: str | None = None,
    ) -> list[Any]:
        """Sort a collection using specified parameters.

        Supports sorting by:
        - Property names (including nested properties)
        - Multiple properties (applied in sequence)
        - Numeric, alphabetic, date, or custom comparators

        Args:
            collection: Collection to sort (list or tuple)
            sort_by: Property name(s) to sort by (None = sort items directly)
            order: Sort order ("ASC" or "DESC"). Defaults to "ASC"
            comparator: Comparator type (NUMERIC, ALPHABETIC, DATE, CUSTOM)
            custom_comparator: Custom comparator expression (required if comparator=CUSTOM)

        Returns:
            New sorted list

        Raises:
            ValueError: If comparator is CUSTOM but custom_comparator is not provided

        Example:
            >>> executor = SortExecutor(context, evaluator)
            >>> data = [{"price": "30"}, {"price": "100"}, {"price": "5"}]
            >>> sorted_data = executor.sort_collection(
            ...     data, sort_by="price", order="DESC", comparator="NUMERIC"
            ... )
            >>> [item["price"] for item in sorted_data]
            ["100", "30", "5"]
        """
        if not collection:
            logger.debug("Empty collection, nothing to sort")
            return collection

        logger.debug(
            f"Sorting collection of {len(collection)} items "
            f"(sort_by={sort_by}, order={order}, comparator={comparator})"
        )

        # Determine sort key function
        key_func = self._build_sort_key_function(sort_by, comparator, custom_comparator)

        # Perform sort
        reverse = order == "DESC"

        try:
            sorted_list = sorted(collection, key=key_func, reverse=reverse)
            logger.debug(f"Successfully sorted {len(sorted_list)} items")
            return sorted_list
        except (TypeError, ValueError, AttributeError) as e:
            logger.error(f"Sort failed: {e}")
            # Return original collection on error
            return collection

    def _build_sort_key_function(
        self,
        sort_by: str | list[str] | None,
        comparator: str | None,
        custom_comparator: str | None,
    ) -> Any:
        """Build the key function for sorting based on parameters.

        Args:
            sort_by: Property name(s) to sort by
            comparator: Comparator type
            custom_comparator: Custom comparator expression

        Returns:
            Key function for use with sorted()

        Raises:
            ValueError: If comparator is CUSTOM but custom_comparator is not provided
        """
        # Start with property extraction if sort_by is specified
        if sort_by:
            properties = [sort_by] if isinstance(sort_by, str) else sort_by
            base_key_func = lambda item: self._extract_property(item, properties)
        else:
            base_key_func = lambda item: item

        # Apply comparator transformation
        if not comparator:
            return base_key_func

        comparator_enum = ComparatorType(comparator)

        if comparator_enum == ComparatorType.NUMERIC:
            return lambda item: self._numeric_key(base_key_func(item))

        elif comparator_enum == ComparatorType.ALPHABETIC:
            return lambda item: self._alphabetic_key(base_key_func(item))

        elif comparator_enum == ComparatorType.DATE:
            return lambda item: self._date_key(base_key_func(item))

        elif comparator_enum == ComparatorType.CUSTOM:
            if not custom_comparator:
                raise ValueError("CUSTOM comparator requires 'custom_comparator'")
            return lambda item: self._custom_key(item, custom_comparator)

        return base_key_func

    def _extract_property(self, item: Any, properties: list[str]) -> Any:
        """Extract property value from item, supporting nested properties.

        Args:
            item: Item to extract from (dict or object)
            properties: List of property names for nested access

        Returns:
            Extracted property value or None if not found

        Example:
            >>> item = {"user": {"name": "Alice"}}
            >>> executor._extract_property(item, ["user", "name"])
            'Alice'
        """
        value = item

        for prop in properties:
            if isinstance(value, dict):
                value = value.get(prop)
            else:
                try:
                    value = getattr(value, prop)
                except AttributeError:
                    logger.debug(f"Property '{prop}' not found on {type(value)}")
                    return None

            if value is None:
                break

        return value

    def _numeric_key(self, value: Any) -> float:
        """Convert value to numeric for comparison.

        Args:
            value: Value to convert

        Returns:
            Float value (0.0 if conversion fails)
        """
        try:
            return float(value or 0)
        except (ValueError, TypeError):
            logger.debug(f"Could not convert '{value}' to numeric, using 0")
            return 0.0

    def _alphabetic_key(self, value: Any) -> str:
        """Convert value to string for alphabetic comparison.

        Args:
            value: Value to convert

        Returns:
            String value (empty string if None)
        """
        return str(value or "")

    def _date_key(self, value: Any) -> datetime:
        """Convert value to datetime for date comparison.

        Supports:
        - datetime objects (returned as-is)
        - ISO format date strings
        - Invalid dates return datetime.min for sorting

        Args:
            value: Value to convert

        Returns:
            Datetime object (datetime.min if conversion fails)
        """
        if isinstance(value, datetime):
            return value

        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except (ValueError, TypeError):
                logger.debug(f"Could not parse '{value}' as ISO date")
                return datetime.min

        return datetime.min

    def _custom_key(self, item: Any, custom_comparator: str) -> Any:
        """Evaluate custom comparator expression for sorting key.

        Args:
            item: Item being sorted
            custom_comparator: Expression to evaluate (has access to 'item' variable)

        Returns:
            Result of expression evaluation (0 if evaluation fails)
        """
        try:
            return self.evaluator.safe_eval(
                custom_comparator,
                {"item": item, **self.variable_context.get_all_variables()},
            )
        except (ValueError, SyntaxError, TypeError, NameError) as e:
            logger.warning(f"Custom comparator evaluation failed: {e}")
            return 0
