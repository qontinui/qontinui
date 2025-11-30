"""Execution Manager - Facade for managing concurrent workflow executions.

This module provides the ExecutionManager class that delegates to:
- ExecutionController - Lifecycle control (start, pause, resume, cancel, step)
- ExecutionStatusTracker - Status and progress tracking
- StateOperationsFacade - State API delegation
- ExecutionOrchestrator - Workflow execution coordination
- ExecutionRegistry - Active execution storage
- ExecutionHistory - Execution history tracking
- ExecutionEventBus - Event streaming
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..config import Workflow
from .execution_controller import ExecutionController
from .execution_event_bus import ExecutionEventBus
from .execution_history import ExecutionHistory
from .execution_orchestrator import ExecutionOrchestrator
from .execution_registry import ExecutionRegistry
from .execution_status_tracker import ExecutionStatusTracker
from .state_operations_facade import StateOperationsFacade

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
    executor: Any | None = None
    action_executor: Any | None = None
    execution_task: asyncio.Task | None = None

    # State tracking
    current_action: str | None = None
    action_states: dict[str, str] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)

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
# Execution Manager Facade
# ============================================================================


class ExecutionManager:
    """Facade for managing concurrent workflow executions.

    The ExecutionManager delegates to focused components:
    - ExecutionController: Lifecycle control
    - ExecutionStatusTracker: Status and progress tracking
    - StateOperationsFacade: State operation delegation
    - ExecutionOrchestrator: Workflow execution coordination
    - ExecutionRegistry: Active execution storage
    - ExecutionHistory: Execution history tracking
    - ExecutionEventBus: Event streaming

    This facade provides a unified API while maintaining clean separation
    of concerns through delegation.
    """

    def __init__(self) -> None:
        """Initialize execution manager."""
        # Core components
        self.registry = ExecutionRegistry()
        self.history = ExecutionHistory(max_history=1000)
        self.event_bus = ExecutionEventBus(max_events_per_execution=1000)

        # Execution orchestrator
        self.orchestrator = ExecutionOrchestrator(
            event_bus=self.event_bus,
            history=self.history,
        )

        # Focused components
        self.controller = ExecutionController(
            registry=self.registry,
            event_bus=self.event_bus,
            orchestrator=self.orchestrator,
            history=self.history,
        )

        self.status_tracker = ExecutionStatusTracker(
            registry=self.registry,
            history=self.history,
        )

        self.state_operations = StateOperationsFacade(
            state_apis=self.controller.state_apis,
        )

        logger.info("ExecutionManager initialized")

    # ========================================================================
    # Execution Control - Delegated to ExecutionController
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
        if options is None:
            options = ExecutionOptions()

        return await self.controller.start_execution(
            workflow=workflow,
            options=options,
            context_factory=ExecutionContext,
        )

    async def pause_execution(self, execution_id: str) -> None:
        """Pause a running execution.

        Args:
            execution_id: Execution ID

        Raises:
            ValueError: If execution not found or cannot be paused
        """
        await self.controller.pause_execution(execution_id)

    async def resume_execution(self, execution_id: str) -> None:
        """Resume a paused execution.

        Args:
            execution_id: Execution ID

        Raises:
            ValueError: If execution not found or cannot be resumed
        """
        await self.controller.resume_execution(execution_id)

    async def step_execution(self, execution_id: str) -> None:
        """Execute next action in step mode.

        Args:
            execution_id: Execution ID

        Raises:
            ValueError: If execution not found or not in step mode
        """
        await self.controller.step_execution(execution_id)

    async def cancel_execution(self, execution_id: str) -> None:
        """Cancel a running execution.

        Args:
            execution_id: Execution ID

        Raises:
            ValueError: If execution not found
        """
        await self.controller.cancel_execution(execution_id)

    # ========================================================================
    # Status and History - Delegated to ExecutionStatusTracker
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
        return self.status_tracker.get_status(execution_id)

    def get_all_executions(self) -> list[dict[str, Any]]:
        """Get all active executions.

        Returns:
            List of execution status dictionaries
        """
        return self.status_tracker.get_all_executions()

    async def get_execution_history(
        self, workflow_id: str | None = None, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Get execution history.

        Args:
            workflow_id: Filter by workflow ID (optional)
            limit: Maximum number of records to return

        Returns:
            List of execution records
        """
        return await self.status_tracker.get_execution_history(workflow_id=workflow_id, limit=limit)

    # ========================================================================
    # Event Streaming - Delegated to ExecutionEventBus
    # ========================================================================

    async def subscribe_to_events(
        self, execution_id: str, callback: Callable[[ExecutionEvent], None]
    ) -> None:
        """Subscribe to execution events.

        Args:
            execution_id: Execution ID
            callback: Callback function for events

        Raises:
            ValueError: If execution not found
        """
        # Verify execution exists
        if not self.registry.has(execution_id):
            raise ValueError(f"Execution {execution_id} not found")

        # Delegate to event bus
        await self.event_bus.subscribe(execution_id, callback)

    async def unsubscribe_from_events(
        self, execution_id: str, callback: Callable[[ExecutionEvent], None]
    ) -> None:
        """Unsubscribe from execution events.

        Args:
            execution_id: Execution ID
            callback: Callback function to remove
        """
        # Delegate to event bus
        await self.event_bus.unsubscribe(execution_id, callback)

    # ========================================================================
    # State Operations - Delegated to StateOperationsFacade
    # ========================================================================

    def execute_transition(self, execution_id: str, transition_id: str) -> dict[str, Any]:
        """Execute a transition via StateExecutionAPI.

        ExecutionManager NEVER touches state - all state operations delegated
        to StateExecutionAPI.

        Args:
            execution_id: Execution identifier
            transition_id: Transition identifier to execute

        Returns:
            Dictionary with transition execution result:
            - success: Whether transition succeeded
            - transition_id: Transition that was executed
            - activated_states: List of states activated
            - deactivated_states: List of states deactivated
            - error: Error message if failed

        Raises:
            ValueError: If execution not found
        """
        return self.state_operations.execute_transition(execution_id, transition_id)

    def navigate_to_states(self, execution_id: str, target_state_ids: list[str]) -> dict[str, Any]:
        """Navigate to target states via StateExecutionAPI.

        ExecutionManager NEVER touches state - all state operations delegated
        to StateExecutionAPI.

        Args:
            execution_id: Execution identifier
            target_state_ids: List of target state IDs

        Returns:
            Dictionary with navigation result:
            - success: Whether navigation succeeded
            - path: List of transition IDs executed
            - active_states: Currently active states
            - error: Error message if failed

        Raises:
            ValueError: If execution not found
        """
        return self.state_operations.navigate_to_states(execution_id, target_state_ids)

    def get_active_states(self, execution_id: str) -> list[str]:
        """Get active states via StateExecutionAPI.

        ExecutionManager NEVER touches state - all state operations delegated
        to StateExecutionAPI.

        Args:
            execution_id: Execution identifier

        Returns:
            List of currently active state IDs

        Raises:
            ValueError: If execution not found
        """
        return self.state_operations.get_active_states(execution_id)

    def get_available_transitions(self, execution_id: str) -> list[dict[str, Any]]:
        """Get available transitions via StateExecutionAPI.

        ExecutionManager NEVER touches state - all state operations delegated
        to StateExecutionAPI.

        Args:
            execution_id: Execution identifier

        Returns:
            List of available transition information dictionaries

        Raises:
            ValueError: If execution not found
        """
        return self.state_operations.get_available_transitions(execution_id)
