"""
Workflow execution infrastructure for qontinui.

This module provides graph-based workflow execution with support for:
- Connection routing based on action results
- Output resolution for different action types
- Execution tracking and debugging
- Merge nodes (multiple incoming connections)
- Parallel execution
- Multiple merge strategies
- Thread-safe context management
- Graph traversal and execution
- Connection resolution
- Execution state tracking
"""

from .connection_resolver import ConnectionResolver
from .connection_router import ConnectionRouter
from .execution_state import (
    ActionExecutionRecord,
    ActionStatus,
    ExecutionState,
    ExecutionStatus,
    PendingAction,
)
from .graph_traversal import (
    CycleDetectedError,
    GraphTraverser,
    InfiniteLoopError,
    OrphanedActionsError,
)
from .merge_context import MergeContext, VariableConflictResolution
from .merge_handler import MergeHandler, MergePoint, MergeStrategy
from .merge_strategies import (
    CustomStrategy,
    MajorityStrategy,
    TimeoutStrategy,
    WaitAllStrategy,
    WaitAnyStrategy,
    WaitFirstStrategy,
    create_strategy,
    get_available_strategies,
)
from .merge_strategies import (
    MergeStrategy as MergeStrategyBase,
)
from .output_resolver import OutputResolver, OutputTypeValidator
from .routing_context import PathSegment, RouteRecord, RoutingContext

__all__ = [
    # Connection routing
    "ConnectionRouter",
    "OutputResolver",
    "OutputTypeValidator",
    # Execution tracking
    "RoutingContext",
    "RouteRecord",
    "PathSegment",
    # Context management
    "MergeContext",
    "VariableConflictResolution",
    # Strategies
    "MergeStrategyBase",
    "WaitAllStrategy",
    "WaitAnyStrategy",
    "WaitFirstStrategy",
    "TimeoutStrategy",
    "MajorityStrategy",
    "CustomStrategy",
    "create_strategy",
    "get_available_strategies",
    # Handler
    "MergeHandler",
    "MergePoint",
    "MergeStrategy",
    # Connection resolution
    "ConnectionResolver",
    # Execution state
    "ExecutionState",
    "ExecutionStatus",
    "ActionStatus",
    "ActionExecutionRecord",
    "PendingAction",
    # Graph traversal
    "GraphTraverser",
    "CycleDetectedError",
    "InfiniteLoopError",
    "OrphanedActionsError",
]
