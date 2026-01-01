"""
Workflow execution infrastructure for qontinui.

This module provides graph-based workflow execution with support for:
- Connection routing based on action results
- Output resolution for different action types
- Execution tracking and debugging
- Sequential execution (appropriate for GUI automation)
- Graph traversal and execution
- Connection resolution
- Execution state tracking
- Enhanced three-tier variable context
- Variable utilities for persistence and manipulation
"""

from .connection_resolver import ConnectionResolver
from .connection_router import ConnectionRouter
from .enhanced_variable_context import EnhancedVariableContext
from .execution_controller import ExecutionController
from .execution_state import ExecutionState
from .execution_tracker import ExecutionTracker
from .execution_types import (
    ActionExecutionRecord,
    ActionStatus,
    ExecutionStatus,
    PendingAction,
)
from .graph_executor import GraphExecutor
from .graph_traversal import (
    CycleDetectedError,
    GraphTraverser,
    InfiniteLoopError,
    OrphanedActionsError,
)
from .output_resolver import OutputResolver, OutputTypeValidator
from .routing_context import PathSegment, RouteRecord, RoutingContext
from .success_criteria import (
    SuccessCriteria,
    SuccessCriteriaEvaluator,
    SuccessCriteriaType,
    WorkflowResult,
    evaluate_workflow_success,
)
from .variable_utils import (
    create_variable_snapshot,
    filter_variables_by_prefix,
    get_nested_variable,
    interpolate_variables,
    is_json_serializable,
    load_variables_from_json,
    merge_variable_scopes,
    resolve_variable_reference,
    restore_variable_snapshot,
    sanitize_for_persistence,
    save_variables_to_json,
    set_nested_variable,
    validate_variable_name,
)

__all__ = [
    # Graph execution
    "GraphExecutor",
    # Connection routing
    "ConnectionRouter",
    "OutputResolver",
    "OutputTypeValidator",
    # Execution tracking
    "RoutingContext",
    "RouteRecord",
    "PathSegment",
    # Connection resolution
    "ConnectionResolver",
    # Execution state
    "ExecutionState",
    "ExecutionController",
    "ExecutionTracker",
    "ExecutionStatus",
    "ActionStatus",
    "ActionExecutionRecord",
    "PendingAction",
    # Graph traversal
    "GraphTraverser",
    "CycleDetectedError",
    "InfiniteLoopError",
    "OrphanedActionsError",
    # Enhanced variable context
    "EnhancedVariableContext",
    # Variable utilities
    "load_variables_from_json",
    "save_variables_to_json",
    "merge_variable_scopes",
    "is_json_serializable",
    "validate_variable_name",
    "interpolate_variables",
    "resolve_variable_reference",
    "get_nested_variable",
    "set_nested_variable",
    "filter_variables_by_prefix",
    "sanitize_for_persistence",
    "create_variable_snapshot",
    "restore_variable_snapshot",
    # Success criteria
    "SuccessCriteria",
    "SuccessCriteriaType",
    "SuccessCriteriaEvaluator",
    "WorkflowResult",
    "evaluate_workflow_success",
]
