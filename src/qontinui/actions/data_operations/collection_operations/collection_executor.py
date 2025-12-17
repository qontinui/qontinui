"""Collection operations executor facade.

This module provides the CollectionExecutor facade class that delegates
to specialized executors for different collection operations.
"""

import logging
from typing import Any

from ..context import VariableContext
from ..evaluator import SafeEvaluator
from .filter_executor import FilterExecutor
from .map_executor import MapExecutor
from .reduce_executor import ReduceExecutor
from .sort_executor import SortExecutor

logger = logging.getLogger(__name__)


class CollectionExecutor:
    """Facade for collection operations (SORT, FILTER, MAP, REDUCE).

    This class delegates to specialized executors:
    - SortExecutor: Sort collections using various comparators
    - FilterExecutor: Filter collections based on conditions
    - MapExecutor: Transform each item in a collection
    - ReduceExecutor: Reduce collections to single values

    The facade provides a unified interface while maintaining separation
    of concerns through specialized executors.

    Example:
        >>> context = VariableContext()
        >>> evaluator = SafeEvaluator()
        >>> executor = CollectionExecutor(context, evaluator)
        >>> data = [{"age": 30}, {"age": 25}, {"age": 35}]
        >>> sorted_data = executor.sort_collection(
        ...     data, sort_by="age", order="ASC", comparator="NUMERIC"
        ... )
    """

    def __init__(
        self, variable_context: VariableContext, evaluator: SafeEvaluator
    ) -> None:
        """Initialize the collection executor facade.

        Args:
            variable_context: Variable context for accessing variables in expressions
            evaluator: Safe evaluator for custom expressions

        Example:
            >>> context = VariableContext()
            >>> evaluator = SafeEvaluator()
            >>> executor = CollectionExecutor(context, evaluator)
        """
        self._sort = SortExecutor(variable_context, evaluator)
        self._filter = FilterExecutor(variable_context, evaluator)
        self._map = MapExecutor(variable_context, evaluator)
        self._reduce = ReduceExecutor(variable_context, evaluator)
        logger.debug("Initialized CollectionExecutor facade")

    def sort_collection(
        self,
        collection: list[Any],
        sort_by: str | list[str] | None,
        order: str = "ASC",
        comparator: str | None = None,
        custom_comparator: str | None = None,
    ) -> list[Any]:
        """Sort a collection using specified parameters.

        Delegates to SortExecutor.

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
        """
        return self._sort.sort_collection(
            collection, sort_by, order, comparator, custom_comparator
        )

    def filter_collection(self, collection: list[Any], condition: Any) -> list[Any]:
        """Filter a collection using specified condition.

        Delegates to FilterExecutor.

        Args:
            collection: Collection to filter
            condition: Filter condition configuration

        Returns:
            New list containing only items matching the condition
        """
        return self._filter.filter_collection(collection, condition)

    def map_collection(self, collection: list[Any], transform: Any) -> list[Any]:
        """Map a collection using specified transform.

        Delegates to MapExecutor.

        Args:
            collection: Collection to map
            transform: Transform configuration

        Returns:
            New list with transformed items
        """
        return self._map.map_collection(collection, transform)

    def reduce_collection(
        self,
        collection: list[Any],
        operation: str,
        initial_value: Any = None,
        custom_reducer: str | None = None,
    ) -> Any:
        """Reduce a collection to a single value.

        Delegates to ReduceExecutor.

        Args:
            collection: Collection to reduce
            operation: Reduce operation (sum, average, min, max, count, custom)
            initial_value: Initial accumulator value (used for sum and custom)
            custom_reducer: Custom reducer expression (required if operation=custom)

        Returns:
            Reduced value (type depends on operation)

        Raises:
            ValueError: If operation is custom but custom_reducer is not provided
        """
        return self._reduce.reduce_collection(
            collection, operation, initial_value, custom_reducer
        )
