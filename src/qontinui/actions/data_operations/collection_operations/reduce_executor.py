"""Reduce operations executor for collections.

This module provides the ReduceExecutor class for reducing collections
to single values using various accumulation strategies.
"""

import logging
from typing import Any

from ..context import VariableContext
from ..evaluator import SafeEvaluator

logger = logging.getLogger(__name__)


class ReduceExecutor:
    """Executor for reducing collections.

    Handles reduction operations with support for:
    - Built-in operations (sum, average, min, max, count)
    - Custom reducer expressions
    - Initial value support

    Example:
        >>> context = VariableContext()
        >>> evaluator = SafeEvaluator()
        >>> executor = ReduceExecutor(context, evaluator)
        >>> data = [1, 2, 3, 4, 5]
        >>> executor.reduce_collection(data, "sum")
        15
        >>> executor.reduce_collection(data, "average")
        3.0
        >>> executor.reduce_collection(data, "max")
        5
    """

    def __init__(self, variable_context: VariableContext, evaluator: SafeEvaluator) -> None:
        """Initialize the reduce executor.

        Args:
            variable_context: Variable context for accessing variables in expressions
            evaluator: Safe evaluator for custom expressions
        """
        self.variable_context = variable_context
        self.evaluator = evaluator
        logger.debug("Initialized ReduceExecutor")

    def reduce_collection(
        self,
        collection: list[Any],
        operation: str,
        initial_value: Any = None,
        custom_reducer: str | None = None,
    ) -> Any:
        """Reduce a collection to a single value.

        Supports built-in operations:
        - sum: Sum all values
        - average: Calculate mean of values
        - min: Find minimum value
        - max: Find maximum value
        - count: Count items
        - custom: Apply custom reducer expression

        Args:
            collection: Collection to reduce
            operation: Reduce operation (sum, average, min, max, count, custom)
            initial_value: Initial accumulator value (used for sum and custom)
            custom_reducer: Custom reducer expression (required if operation=custom)

        Returns:
            Reduced value (type depends on operation)

        Raises:
            ValueError: If operation is custom but custom_reducer is not provided

        Example:
            >>> data = [1, 2, 3, 4, 5]
            >>> executor.reduce_collection(data, "sum")
            15
            >>> executor.reduce_collection(data, "average")
            3.0
            >>> executor.reduce_collection(data, "max")
            5
        """
        if not collection:
            logger.debug("Empty collection, returning initial value or default")
            return initial_value if initial_value is not None else 0

        logger.debug(f"Reducing collection of {len(collection)} items (operation={operation})")

        if operation == "sum":
            result = self._reduce_sum(collection, initial_value)
        elif operation == "average":
            result = self._reduce_average(collection, initial_value)
        elif operation == "min":
            result = self._reduce_min(collection)
        elif operation == "max":
            result = self._reduce_max(collection)
        elif operation == "count":
            result = self._reduce_count(collection)
        elif operation == "custom":
            result = self._reduce_custom(collection, initial_value, custom_reducer)
        else:
            raise ValueError(f"Unknown reduce operation: {operation}")

        logger.debug(f"Reduced {len(collection)} items to {result}")
        return result

    def _reduce_sum(self, collection: list[Any], initial_value: Any) -> float:
        """Sum all values in collection.

        Args:
            collection: Collection to sum
            initial_value: Initial value to add to sum

        Returns:
            Sum of all values
        """
        return sum(collection, initial_value or 0)

    def _reduce_average(self, collection: list[Any], initial_value: Any) -> float:
        """Calculate average of values in collection.

        Args:
            collection: Collection to average
            initial_value: Initial value to add before averaging

        Returns:
            Average (mean) of all values
        """
        total = sum(collection, initial_value or 0)
        return total / len(collection)

    def _reduce_min(self, collection: list[Any]) -> Any:
        """Find minimum value in collection.

        Args:
            collection: Collection to search

        Returns:
            Minimum value
        """
        return min(collection)

    def _reduce_max(self, collection: list[Any]) -> Any:
        """Find maximum value in collection.

        Args:
            collection: Collection to search

        Returns:
            Maximum value
        """
        return max(collection)

    def _reduce_count(self, collection: list[Any]) -> int:
        """Count items in collection.

        Args:
            collection: Collection to count

        Returns:
            Number of items
        """
        return len(collection)

    def _reduce_custom(
        self, collection: list[Any], initial_value: Any, custom_reducer: str | None
    ) -> Any:
        """Reduce collection using custom reducer expression.

        The custom reducer expression has access to:
        - accumulator (or acc): Current accumulated value
        - item: Current item being processed

        Args:
            collection: Collection to reduce
            initial_value: Initial accumulator value
            custom_reducer: Reducer expression

        Returns:
            Final accumulated value

        Raises:
            ValueError: If custom_reducer is not provided

        Example:
            >>> # Custom reducer to concatenate strings
            >>> data = ["a", "b", "c"]
            >>> executor.reduce_collection(
            ...     data, "custom", "", "accumulator + item + ','"
            ... )
            'a,b,c,'
        """
        if not custom_reducer:
            raise ValueError("Custom reduce operation requires 'custom_reducer'")

        # Initialize accumulator
        if initial_value is not None:
            accumulator = initial_value
            start_idx = 0
        else:
            accumulator = collection[0]
            start_idx = 1

        # Apply custom reducer to each item
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
