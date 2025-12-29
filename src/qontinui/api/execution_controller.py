"""Execution Controller - Manages execution lifecycle control.

This module provides the ExecutionController class that handles:
- Starting new workflow executions
- Pausing and resuming executions
- Stepping through executions in debug mode
- Cancelling running executions
- Coordinating with registry, event bus, and orchestrator
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING

from qontinui_schemas.common import utc_now

from ..config import Workflow
from ..state_execution_api import StateExecutionAPI
from .execution_event_bus import ExecutionEventBus
from .execution_history import ExecutionHistory
from .execution_orchestrator import ExecutionOrchestrator
from .execution_registry import ExecutionRegistry

if TYPE_CHECKING:
    from .execution_manager import ExecutionContext, ExecutionOptions

logger = logging.getLogger(__name__)


class ExecutionController:
    """Controls workflow execution lifecycle.

    The ExecutionController handles:
    - Starting new executions with proper initialization
    - Pausing/resuming execution flow
    - Stepping through actions in debug mode
    - Cancelling running executions
    - Coordinating between registry, event bus, and orchestrator

    This class focuses solely on lifecycle control operations.
    """

    def __init__(
        self,
        registry: ExecutionRegistry,
        event_bus: ExecutionEventBus,
        orchestrator: ExecutionOrchestrator,
        history: ExecutionHistory,
    ) -> None:
        """Initialize execution controller.

        Args:
            registry: Execution registry for storing contexts
            event_bus: Event bus for streaming events
            orchestrator: Orchestrator for workflow execution
            history: History tracker for execution records
        """
        self.registry = registry
        self.event_bus = event_bus
        self.orchestrator = orchestrator
        self.history = history

        # State execution APIs (per execution)
        self.state_apis: dict[str, StateExecutionAPI] = {}

        logger.info("ExecutionController initialized")

    async def start_execution(
        self,
        workflow: Workflow,
        options: ExecutionOptions,
        context_factory,
    ) -> str:
        """Start a new workflow execution.

        Args:
            workflow: Workflow to execute
            options: Execution options
            context_factory: Factory function to create ExecutionContext

        Returns:
            Execution ID
        """
        # Import here to avoid circular dependency
        from .execution_manager import ExecutionEventType, ExecutionStatus

        # Generate execution ID
        execution_id = str(uuid.uuid4())

        # Create execution context
        context = context_factory(
            execution_id=execution_id,
            workflow=workflow,
            options=options,
            status=ExecutionStatus.STARTING,
            start_time=utc_now(),
            total_actions=len(workflow.actions),
        )

        # Initialize action states
        for action in workflow.actions:
            context.action_states[action.id] = "idle"

        # Store context
        self.registry.add(context)

        # Register with event bus
        await self.event_bus.register_execution(execution_id)

        # Create StateExecutionAPI for this execution
        def event_callback(event_type: str, data: dict) -> None:
            """Callback for state execution events."""
            asyncio.create_task(
                self._emit_event(
                    context,
                    ExecutionEventType.LOG,
                    data={"event_type": event_type, "event_data": data},
                )
            )

        state_api = StateExecutionAPI(workflow, event_callback=event_callback)  # type: ignore[arg-type,call-arg]
        self.state_apis[execution_id] = state_api

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
        from .execution_manager import ExecutionStatus

        context = self.registry.get(execution_id)
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
        from .execution_manager import ExecutionStatus

        context = self.registry.get(execution_id)
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
        context = self.registry.get(execution_id)
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
        from .execution_manager import ExecutionEventType, ExecutionStatus

        context = self.registry.get(execution_id)
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

    def get_state_api(self, execution_id: str) -> StateExecutionAPI:
        """Get StateExecutionAPI for an execution.

        Args:
            execution_id: Execution ID

        Returns:
            StateExecutionAPI for the execution

        Raises:
            ValueError: If execution not found
        """
        state_api = self.state_apis.get(execution_id)
        if not state_api:
            raise ValueError(f"Execution {execution_id} not found")
        return state_api

    async def _run_execution(self, context: ExecutionContext) -> None:
        """Run workflow execution.

        Delegates to ExecutionOrchestrator for orchestration logic.

        Args:
            context: Execution context
        """
        try:
            # Delegate to orchestrator
            await self.orchestrator.execute(context)

        except asyncio.CancelledError:
            logger.info(f"Execution {context.execution_id} cancelled")

        except Exception as e:
            logger.error(f"Execution {context.execution_id} failed: {e}", exc_info=True)

    async def _emit_event(
        self,
        context: ExecutionContext,
        event_type,
        action_id: str | None = None,
        action_type: str | None = None,
        data: dict | None = None,
    ) -> None:
        """Emit an execution event.

        Args:
            context: Execution context
            event_type: Event type
            action_id: Action ID (optional)
            action_type: Action type (optional)
            data: Event data (optional)
        """
        from .execution_manager import ExecutionEvent

        event = ExecutionEvent(
            event_id=str(uuid.uuid4()),
            type=event_type,
            execution_id=context.execution_id,
            timestamp=utc_now(),
            action_id=action_id,
            action_type=action_type,
            data=data,
        )

        await self.event_bus.emit(context.execution_id, event)
