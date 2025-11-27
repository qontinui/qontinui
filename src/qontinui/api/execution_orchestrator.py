"""Execution Orchestrator - Orchestrates workflow execution coordination.

This module provides the ExecutionOrchestrator class that handles:
- Creating and configuring ActionExecutor and GraphExecutor
- Setting up execution hooks for event streaming
- Running workflow execution with error handling
- Managing execution status and lifecycle
- Coordinating event emission and history tracking
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any

from ..actions.internal.execution.action_executor import ActionExecutor
from ..execution.graph_executor import GraphExecutor
from .execution_event_bus import ExecutionEventBus
from .execution_history import ExecutionHistory
from .execution_manager import (
    ExecutionContext,
    ExecutionEvent,
    ExecutionEventType,
    ExecutionStatus,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Execution Result
# ============================================================================


class ExecutionResult:
    """Result of workflow execution orchestration."""

    def __init__(
        self,
        success: bool,
        context: dict[str, Any],
        summary: dict[str, Any],
        results: dict[str, dict[str, Any]],
    ) -> None:
        """Initialize execution result.

        Args:
            success: Whether execution succeeded
            context: Final execution context variables
            summary: Execution summary with statistics
            results: Individual action results
        """
        self.success = success
        self.context = context
        self.summary = summary
        self.results = results


# ============================================================================
# Execution Hook
# ============================================================================


class OrchestrationHook:
    """Execution hook for event streaming and execution control.

    This hook integrates with GraphExecutor to:
    - Stream execution events via ExecutionEventBus
    - Handle pause/resume/cancel/step control
    - Track action states and statistics
    - Manage breakpoints in debug mode
    """

    def __init__(
        self,
        context: ExecutionContext,
        event_bus: ExecutionEventBus,
    ) -> None:
        """Initialize orchestration hook.

        Args:
            context: Execution context
            event_bus: Event bus for emitting events
        """
        self.context = context
        self.event_bus = event_bus

    async def before_action(self, action: Any, context_vars: dict[str, Any]) -> None:
        """Called before action execution.

        Handles:
        - Pause/resume control
        - Cancellation checks
        - Step mode coordination
        - Breakpoint processing
        - State updates and event emission

        Args:
            action: Action to be executed
            context_vars: Current context variables

        Raises:
            asyncio.CancelledError: If execution is cancelled
        """
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
            await self._emit_event(
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
        await self._emit_event(
            ExecutionEventType.ACTION_START,
            action_id=action.id,
            action_type=action.type,
        )

    async def after_action(
        self, action: Any, context_vars: dict[str, Any], result: dict[str, Any]
    ) -> None:
        """Called after successful action execution.

        Args:
            action: Action that was executed
            context_vars: Updated context variables
            result: Action execution result
        """
        # Update state
        self.context.action_states[action.id] = "completed"
        self.context.completed_actions += 1
        self.context.variables = context_vars.copy()

        # Emit event
        await self._emit_event(
            ExecutionEventType.ACTION_COMPLETE,
            action_id=action.id,
            action_type=action.type,
            data={
                "success": result.get("success", True),
                "result": result,
            },
        )

    async def on_error(self, action: Any, context_vars: dict[str, Any], error: Exception) -> None:
        """Called when action execution fails.

        Args:
            action: Action that failed
            context_vars: Current context variables
            error: Exception that occurred
        """
        # Update state
        self.context.action_states[action.id] = "failed"
        self.context.failed_actions += 1

        # Emit event
        await self._emit_event(
            ExecutionEventType.ACTION_ERROR,
            action_id=action.id,
            action_type=action.type,
            data={
                "error": str(error),
            },
        )

    async def _emit_event(
        self,
        event_type: ExecutionEventType,
        action_id: str | None = None,
        action_type: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Emit an execution event.

        Args:
            event_type: Type of event to emit
            action_id: Action ID (optional)
            action_type: Action type (optional)
            data: Event data (optional)
        """
        event = ExecutionEvent(
            event_id=str(uuid.uuid4()),
            type=event_type,
            execution_id=self.context.execution_id,
            timestamp=datetime.now(),
            action_id=action_id,
            action_type=action_type,
            data=data,
        )

        await self.event_bus.emit(self.context.execution_id, event)


# ============================================================================
# Execution Orchestrator
# ============================================================================


class ExecutionOrchestrator:
    """Orchestrates workflow execution using GraphExecutor.

    The ExecutionOrchestrator is responsible for:
    1. Creating and configuring ActionExecutor
    2. Creating and configuring GraphExecutor
    3. Setting up execution hooks for event streaming
    4. Running workflow execution with proper error handling
    5. Managing execution status and lifecycle
    6. Coordinating event emission
    7. Recording execution history

    This class separates orchestration concerns from execution management,
    allowing ExecutionManager to focus on lifecycle control while the
    orchestrator handles the execution flow.
    """

    def __init__(
        self,
        event_bus: ExecutionEventBus,
        history: ExecutionHistory,
    ) -> None:
        """Initialize execution orchestrator.

        Args:
            event_bus: Event bus for emitting execution events
            history: History tracker for recording executions
        """
        self.event_bus = event_bus
        self.history = history

        logger.info("ExecutionOrchestrator initialized")

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute workflow with full orchestration.

        This method orchestrates the complete workflow execution flow:
        1. Creates ActionExecutor for executing individual actions
        2. Creates GraphExecutor for workflow traversal
        3. Sets up hooks for event streaming and control
        4. Runs the workflow execution
        5. Handles errors and updates status
        6. Emits completion/error events
        7. Records execution in history

        Args:
            context: Execution context with workflow and options

        Returns:
            ExecutionResult with success status and execution data

        Raises:
            asyncio.CancelledError: If execution is cancelled
            Exception: If execution encounters critical errors
        """
        try:
            # Step 1: Update status to running
            context.status = ExecutionStatus.RUNNING
            logger.info(f"Starting orchestration for execution {context.execution_id}")

            # Step 2: Create action executor
            context.action_executor = ActionExecutor(context.variables)
            logger.debug("Created ActionExecutor")

            # Step 3: Create graph executor
            context.executor = GraphExecutor(context.workflow, context.action_executor)
            logger.debug("Created GraphExecutor")

            # Step 4: Setup hook for event streaming and control
            hook = OrchestrationHook(context, self.event_bus)
            context.executor.add_hook(hook)
            logger.debug("Added orchestration hook")

            # Step 5: Execute workflow
            logger.info(f"Executing workflow '{context.workflow.name}'")
            result = context.executor.execute(context.options.initial_variables)

            # Step 6: Process execution result
            if result["success"]:
                context.status = ExecutionStatus.COMPLETED
                logger.info(f"Execution {context.execution_id} completed successfully")
            else:
                context.status = ExecutionStatus.FAILED
                context.error = {
                    "message": "Workflow execution failed",
                    "details": result.get("summary", {}),
                }
                logger.warning(f"Execution {context.execution_id} failed")

            context.end_time = datetime.now()

            # Step 7: Emit completion event
            await self._emit_event(
                context,
                ExecutionEventType.WORKFLOW_COMPLETE,
                data={
                    "status": context.status.value,
                    "summary": result.get("summary", {}),
                },
            )

            # Step 8: Add to history
            await self.history.add_record(context)
            logger.debug(f"Added execution {context.execution_id} to history")

            # Step 9: Return execution result
            return ExecutionResult(
                success=result["success"],
                context=result["context"],
                summary=result["summary"],
                results=result["results"],
            )

        except asyncio.CancelledError:
            # Handle cancellation
            logger.info(f"Execution {context.execution_id} cancelled")
            context.status = ExecutionStatus.CANCELLED
            context.end_time = datetime.now()
            await self.history.add_record(context)
            raise

        except Exception as e:
            # Handle execution errors
            logger.error(
                f"Execution {context.execution_id} failed with error: {e}",
                exc_info=True,
            )
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

            # Add to history
            await self.history.add_record(context)

            # Re-raise the exception
            raise

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

        await self.event_bus.emit(context.execution_id, event)
