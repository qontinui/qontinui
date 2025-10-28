"""Collection operations module.

This module provides specialized executors for collection operations:
- SortExecutor: Sort collections using various comparators
- FilterExecutor: Filter collections based on conditions
- MapExecutor: Transform each item in a collection
- ReduceExecutor: Reduce collections to single values
- CollectionExecutor: Facade that delegates to specialized executors

The main entry point is the CollectionExecutor facade class.
"""

from .collection_executor import CollectionExecutor
from .filter_executor import FilterExecutor
from .map_executor import MapExecutor
from .reduce_executor import ReduceExecutor
from .sort_executor import SortExecutor

__all__ = [
    "CollectionExecutor",
    "SortExecutor",
    "FilterExecutor",
    "MapExecutor",
    "ReduceExecutor",
]
