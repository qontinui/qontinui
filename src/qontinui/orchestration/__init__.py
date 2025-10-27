"""Workflow orchestration package.

Provides workflow execution, retry policies, and execution context management.
Extracted from ActionExecutor to follow Single Responsibility Principle.
"""

from .execution_context import ActionState, ExecutionContext, ExecutionStatistics
from .retry_policy import BackoffStrategy, RetryPolicy
from .workflow_orchestrator import (
    ActionExecutorProtocol,
    EventEmitterProtocol,
    WorkflowOrchestrator,
    WorkflowResult,
)

__all__ = [
    # Retry Policy
    "RetryPolicy",
    "BackoffStrategy",
    # Execution Context
    "ExecutionContext",
    "ExecutionStatistics",
    "ActionState",
    # Workflow Orchestrator
    "WorkflowOrchestrator",
    "WorkflowResult",
    "ActionExecutorProtocol",
    "EventEmitterProtocol",
]
