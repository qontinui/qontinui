"""Collection operations executor - import redirect.

This module provides backward-compatible imports from the refactored
collection operations structure.

The original monolithic CollectionExecutor has been split into specialized
executors in the collection_operations package:
- SortExecutor: Sort operations and comparators
- FilterExecutor: Filter operations and conditions
- MapExecutor: Map operations and transforms
- ReduceExecutor: Reduce operations and accumulators
- CollectionExecutor: Facade that delegates to specialized executors

Import from this module to maintain backward compatibility.
"""

from .collection_operations import CollectionExecutor

__all__ = ["CollectionExecutor"]
