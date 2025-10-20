"""Execution Manager - Manages concurrent workflow executions.

This module provides the ExecutionManager class that handles:
- Starting and managing multiple concurrent workflow executions
- Execution lifecycle (pause, resume, cancel, step)
- Event streaming to WebSocket clients
- Execution status tracking
- Execution history
"""

import asyncio
import logging
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..config import Workflow
from ..execution.action_executor import ActionExecutor
from ..execution.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)


# ============================================================================
# Types and Enums
# ============================================================================


class ExecutionStatus(str, Enum):
    """Execution status enum."""

    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExecutionEventType(str, Enum):
    """Execution event types."""

    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_ERROR = "workflow_error"
    ACTION_START = "action_start"
    ACTION_COMPLETE = "action_complete"
    ACTION_ERROR = "action_error"
    ACTION_SKIP = "action_skip"
    BREAKPOINT = "breakpoint"
    VARIABLE_UPDATE = "variable_update"
    LOG = "log"


@dataclass
class ExecutionOptions:
    """Options for workflow execution."""

    initial_variables: dict[str, Any] = field(default_factory=dict)
    debug_mode: bool = False
    breakpoints: list[str] = field(default_factory=list)
    step_mode: bool = False
    timeout: int = 0  # 0 = unlimited
    continue_on_error: bool = False


@dataclass
class ExecutionEvent:
    """Execution event data."""

    event_id: str
    type: ExecutionEventType
    execution_id: str
    timestamp: datetime
    action_id: str | None = None
    action_type: str | None = None
    data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_id": self.event_id,
            "type": self.type.value,
            "execution_id": self.execution_id,
            "timestamp": self.timestamp.isoformat(),
            "action_id": self.action_id,
            "action_type": self.action_type,
            "data": self.data,
        }


@dataclass
class ExecutionContext:
    """Context for a running execution."""

    execution_id: str
    workflow: Workflow
    options: ExecutionOptions
    status: ExecutionStatus
    start_time: datetime
    end_time: datetime | None = None

    # Execution components
    executor: GraphExecutor | None = None
    action_executor: ActionExecutor | None = None
    execution_task: asyncio.Task | None = None

    # State tracking
    current_action: str | None = None
    action_states: dict[str, str] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)

    # Event streaming
    event_queue: deque = field(default_factory=deque)
    event_subscribers: set[Callable[[ExecutionEvent], None]] = field(default_factory=set)

    # Control
    pause_event: asyncio.Event = field(default_factory=asyncio.Event)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    step_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Statistics
    total_actions: int = 0
    completed_actions: int = 0
    failed_actions: int = 0
    skipped_actions: int = 0

    # Error tracking
    error: dict[str, Any] | None = None

    def __post_init__(self):
        """Initialize events."""
        if not hasattr(self, "pause_event") or self.pause_event is None:
            self.pause_event = asyncio.Event()
        if not hasattr(self, "cancel_event") or self.cancel_event is None:
            self.cancel_event = asyncio.Event()
        if not hasattr(self, "step_event") or self.step_event is None:
            self.step_event = asyncio.Event()

        # In non-step mode, step_event should always be set
        if not self.options.step_mode:
            self.step_event.set()


# ============================================================================
# Execution Manager
# ============================================================================


class ExecutionManager:
    """Manages concurrent workflow executions.

    The ExecutionManager handles:
    - Starting new executions
    - Tracking execution state
    - Controlling execution (pause, resume, cancel, step)
    - Streaming events to WebSocket clients
    - Maintaining execution history
    """

    def __init__(self):
        """Initialize execution manager."""
        self.executions: dict[str, ExecutionContext] = {}
        self.execution_history: list[dict[str, Any]] = []
        self.max_history_size = 1000

        logger.info("ExecutionManager initialized")

    # ========================================================================
    # Execution Control
    # ========================================================================

    async def start_execution(
        self, workflow: Workflow, options: ExecutionOptions | None = None
    ) -> str:
        """Start a new workflow execution.

        Args:
            workflow: Workflow to execute
            options: Execution options

        Returns:
            Execution ID
        """
        # Generate execution ID
        execution_id = str(uuid.uuid4())

        # Create execution context
        if options is None:
            options = ExecutionOptions()

        context = ExecutionContext(
            execution_id=execution_id,
            workflow=workflow,
            options=options,
            status=ExecutionStatus.STARTING,
            start_time=datetime.now(),
            total_actions=len(workflow.actions),
        )

        # Initialize action states
        for action in workflow.actions:
            context.action_states[action.id] = "idle"

        # Store context
        self.executions[execution_id] = context

        # Start execution task
        context.execution_task = asyncio.create_task(self._run_execution(context))

        # Emit start event
        await self._emit_event(
            context,
            ExecutionEventType.WORKFLOW_START,
            data={
                "workflow_id": workflow.id,
                "workflow_name": workflow.name,
                "total_actions": len(workflow.actions),
            },
        )

        logger.info(f"Started execution {execution_id} for workflow '{workflow.name}'")

        return execution_id

    async def pause_execution(self, execution_id: str) -> None:
        """Pause a running execution.

        Args:
            execution_id: Execution ID

        Raises:
            ValueError: If execution not found or cannot be paused
        """
        context = self.executions.get(execution_id)
        if not context:
            raise ValueError(f"Execution {execution_id} not found")

        if context.status != ExecutionStatus.RUNNING:
            raise ValueError(f"Cannot pause execution in state {context.status}")

        context.status = ExecutionStatus.PAUSED
        context.pause_event.clear()

        logger.info(f"Paused execution {execution_id}")

    async def resume_execution(self, execution_id: str) -> None:
        """Resume a paused execution.

        Args:
            execution_id: Execution ID

        Raises:
            ValueError: If execution not found or cannot be resumed
        """
        context = self.executions.get(execution_id)
        if not context:
            raise ValueError(f"Execution {execution_id} not found")

        if context.status != ExecutionStatus.PAUSED:
            raise ValueError(f"Cannot resume execution in state {context.status}")

        context.status = ExecutionStatus.RUNNING
        context.pause_event.set()

        logger.info(f"Resumed execution {execution_id}")

    async def step_execution(self, execution_id: str) -> None:
        """Execute next action in step mode.

        Args:
            execution_id: Execution ID

        Raises:
            ValueError: If execution not found or not in step mode
        """
        context = self.executions.get(execution_id)
        if not context:
            raise ValueError(f"Execution {execution_id} not found")

        if not context.options.step_mode:
            raise ValueError("Execution is not in step mode")

        # Trigger step
        context.step_event.set()

        logger.info(f"Stepped execution {execution_id}")

    async def cancel_execution(self, execution_id: str) -> None:
        """Cancel a running execution.

        Args:
            execution_id: Execution ID

        Raises:
            ValueError: If execution not found
        """
        context = self.executions.get(execution_id)
        if not context:
            raise ValueError(f"Execution {execution_id} not found")

        # Set cancel event
        context.cancel_event.set()
        context.status = ExecutionStatus.CANCELLED

        # Cancel task
        if context.execution_task:
            context.execution_task.cancel()

        # Emit event
        await self._emit_event(
            context, ExecutionEventType.WORKFLOW_COMPLETE, data={"status": "cancelled"}
        )

        logger.info(f"Cancelled execution {execution_id}")

    # ========================================================================
    # Status and History
    # ========================================================================

    def get_status(self, execution_id: str) -> dict[str, Any]:
        """Get execution status.

        Args:
            execution_id: Execution ID

        Returns:
            Status dictionary

        Raises:
            ValueError: If execution not found
        """
        context = self.executions.get(execution_id)
        if not context:
            raise ValueError(f"Execution {execution_id} not found")

        # Calculate progress
        progress = 0.0
        if context.total_actions > 0:
            completed = context.completed_actions + context.failed_actions + context.skipped_actions
            progress = (completed / context.total_actions) * 100

        status = {
            "execution_id": execution_id,
            "workflow_id": context.workflow.id,
            "status": context.status.value,
            "start_time": context.start_time.isoformat(),
            "end_time": context.end_time.isoformat() if context.end_time else None,
            "current_action": context.current_action,
            "progress": progress,
            "total_actions": context.total_actions,
            "completed_actions": context.completed_actions,
            "failed_actions": context.failed_actions,
            "skipped_actions": context.skipped_actions,
            "action_states": context.action_states,
            "error": context.error,
            "variables": context.variables,
        }

        return status

    def get_all_executions(self) -> list[dict[str, Any]]:
        """Get all active executions.

        Returns:
            List of execution status dictionaries
        """
        return [self.get_status(exec_id) for exec_id in self.executions.keys()]

    def get_execution_history(
        self, workflow_id: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Get execution history.

        Args:
            workflow_id: Filter by workflow ID (optional)
            limit: Maximum number of records to return

        Returns:
            List of execution records
        """
        history = self.execution_history

        # Filter by workflow ID
        if workflow_id:
            history = [r for r in history if r["workflow_id"] == workflow_id]

        # Apply limit
        if limit:
            history = history[:limit]

        return history

    # ========================================================================
    # Event Streaming
    # ========================================================================

    def subscribe_to_events(
        self, execution_id: str, callback: Callable[[ExecutionEvent], None]
    ) -> None:
        """Subscribe to execution events.

        Args:
            execution_id: Execution ID
            callback: Callback function for events

        Raises:
            ValueError: If execution not found
        """
        context = self.executions.get(execution_id)
        if not context:
            raise ValueError(f"Execution {execution_id} not found")

        context.event_subscribers.add(callback)

        logger.debug(f"Added event subscriber to execution {execution_id}")

    def unsubscribe_from_events(
        self, execution_id: str, callback: Callable[[ExecutionEvent], None]
    ) -> None:
        """Unsubscribe from execution events.

        Args:
            execution_id: Execution ID
            callback: Callback function to remove
        """
        context = self.executions.get(execution_id)
        if context:
            context.event_subscribers.discard(callback)
            logger.debug(f"Removed event subscriber from execution {execution_id}")

    # ========================================================================
    # Private Methods
    # ========================================================================

    async def _run_execution(self, context: ExecutionContext) -> None:
        """Run workflow execution.

        Args:
            context: Execution context
        """
        try:
            # Update status
            context.status = ExecutionStatus.RUNNING

            # Create action executor
            context.action_executor = ActionExecutor(context.variables)

            # Create graph executor
            context.executor = GraphExecutor(context.workflow, context.action_executor)

            # Add execution hook
            hook = ExecutionHook(self, context)
            context.executor.add_hook(hook)

            # Execute workflow
            result = context.executor.execute(context.options.initial_variables)

            # Update status
            if result["success"]:
                context.status = ExecutionStatus.COMPLETED
            else:
                context.status = ExecutionStatus.FAILED
                context.error = {
                    "message": "Workflow execution failed",
                    "details": result.get("summary", {}),
                }

            context.end_time = datetime.now()

            # Emit completion event
            await self._emit_event(
                context,
                ExecutionEventType.WORKFLOW_COMPLETE,
                data={
                    "status": context.status.value,
                    "summary": result.get("summary", {}),
                },
            )

            # Add to history
            self._add_to_history(context)

        except asyncio.CancelledError:
            logger.info(f"Execution {context.execution_id} cancelled")
            context.status = ExecutionStatus.CANCELLED
            context.end_time = datetime.now()
            self._add_to_history(context)

        except Exception as e:
            logger.error(f"Execution {context.execution_id} failed: {e}", exc_info=True)
            context.status = ExecutionStatus.FAILED
            context.end_time = datetime.now()
            context.error = {
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }

            # Emit error event
            await self._emit_event(
                context,
                ExecutionEventType.WORKFLOW_ERROR,
                data={
                    "error": str(e),
                },
            )

            self._add_to_history(context)

    async def _emit_event(
        self,
        context: ExecutionContext,
        event_type: ExecutionEventType,
        action_id: str | None = None,
        action_type: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit an execution event.

        Args:
            context: Execution context
            event_type: Event type
            action_id: Action ID (optional)
            action_type: Action type (optional)
            data: Event data (optional)
        """
        event = ExecutionEvent(
            event_id=str(uuid.uuid4()),
            type=event_type,
            execution_id=context.execution_id,
            timestamp=datetime.now(),
            action_id=action_id,
            action_type=action_type,
            data=data,
        )

        # Add to queue
        context.event_queue.append(event)

        # Notify subscribers
        for subscriber in context.event_subscribers:
            try:
                subscriber(event)
            except Exception as e:
                logger.error(f"Error in event subscriber: {e}", exc_info=True)

    def _add_to_history(self, context: ExecutionContext) -> None:
        """Add execution to history.

        Args:
            context: Execution context
        """
        duration = 0
        if context.end_time:
            duration = int((context.end_time - context.start_time).total_seconds() * 1000)

        record = {
            "execution_id": context.execution_id,
            "workflow_id": context.workflow.id,
            "workflow_name": context.workflow.name,
            "start_time": context.start_time.isoformat(),
            "end_time": context.end_time.isoformat() if context.end_time else None,
            "status": context.status.value,
            "duration": duration,
            "total_actions": context.total_actions,
            "completed_actions": context.completed_actions,
            "failed_actions": context.failed_actions,
            "error": context.error.get("message") if context.error else None,
        }

        self.execution_history.insert(0, record)

        # Limit history size
        if len(self.execution_history) > self.max_history_size:
            self.execution_history = self.execution_history[: self.max_history_size]


# ============================================================================
# Execution Hook
# ============================================================================


class ExecutionHook:
    """Execution hook for event streaming and control."""

    def __init__(self, manager: ExecutionManager, context: ExecutionContext):
        """Initialize hook.

        Args:
            manager: Execution manager
            context: Execution context
        """
        self.manager = manager
        self.context = context

    async def before_action(self, action, context_vars):
        """Called before action execution."""
        # Check for pause
        await self.context.pause_event.wait()

        # Check for cancellation
        if self.context.cancel_event.is_set():
            raise asyncio.CancelledError()

        # Check for step mode
        if self.context.options.step_mode:
            await self.context.step_event.wait()
            self.context.step_event.clear()

        # Check for breakpoint
        if action.id in self.context.options.breakpoints:
            await self.manager._emit_event(
                self.context,
                ExecutionEventType.BREAKPOINT,
                action_id=action.id,
                action_type=action.type,
            )
            self.context.status = ExecutionStatus.PAUSED
            self.context.pause_event.clear()
            await self.context.pause_event.wait()

        # Update state
        self.context.current_action = action.id
        self.context.action_states[action.id] = "running"

        # Emit event
        await self.manager._emit_event(
            self.context,
            ExecutionEventType.ACTION_START,
            action_id=action.id,
            action_type=action.type,
        )

    async def after_action(self, action, context_vars, result):
        """Called after successful action execution."""
        # Update state
        self.context.action_states[action.id] = "completed"
        self.context.completed_actions += 1
        self.context.variables = context_vars.copy()

        # Emit event
        await self.manager._emit_event(
            self.context,
            ExecutionEventType.ACTION_COMPLETE,
            action_id=action.id,
            action_type=action.type,
            data={
                "success": result.get("success", True),
                "result": result,
            },
        )

    async def on_error(self, action, context_vars, error):
        """Called when action execution fails."""
        # Update state
        self.context.action_states[action.id] = "failed"
        self.context.failed_actions += 1

        # Emit event
        await self.manager._emit_event(
            self.context,
            ExecutionEventType.ACTION_ERROR,
            action_id=action.id,
            action_type=action.type,
            data={
                "error": str(error),
            },
        )
