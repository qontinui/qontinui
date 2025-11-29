"""Workflow orchestrator for action execution.

Orchestrates workflow execution with retry logic, error handling, and state management.
Extracted from ActionExecutor to follow Single Responsibility Principle.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .execution_context import ExecutionContext
from .retry_policy import RetryPolicy

logger = logging.getLogger(__name__)


class ActionExecutorProtocol(Protocol):
    """Protocol for action execution.

    Defines the interface that action executors must implement.
    """

    def execute(self, action: Any, target: Any | None = None) -> bool:
        """Execute an action.

        Args:
            action: Action to execute
            target: Optional target for action

        Returns:
            True if successful
        """
        ...


class EventEmitterProtocol(Protocol):
    """Protocol for event emission.

    Defines the interface for event emitters.
    """

    def emit(self, event_type: str, **kwargs: Any) -> None:
        """Emit an event.

        Args:
            event_type: Type of event
            kwargs: Event data
        """
        ...


@dataclass
class WorkflowResult:
    """Result of workflow execution."""

    success: bool
    """Whether the workflow completed successfully."""

    error: Exception | None = None
    """Error that occurred, if any."""

    context: ExecutionContext | None = None
    """Execution context with variables and statistics."""

    completed_actions: int = 0
    """Number of actions that were executed."""

    failed_action_index: int | None = None
    """Index of the action that failed, if any."""

    def __str__(self) -> str:
        """String representation of result."""
        status = "SUCCESS" if self.success else "FAILED"
        if self.context:
            return f"WorkflowResult({status}, {self.context.statistics})"
        return f"WorkflowResult({status}, actions={self.completed_actions})"


@dataclass
class WorkflowOrchestrator:
    """Orchestrates workflow execution with retry and error handling.

    This class is responsible for:
    - Executing sequences of actions in order
    - Applying retry policies to failed actions
    - Managing execution context and variables
    - Emitting events for monitoring
    - Handling errors and continue-on-error logic

    Uses dependency injection for testability and flexibility.
    """

    action_executor: ActionExecutorProtocol
    """Executor for individual actions."""

    retry_policy: RetryPolicy | None = None
    """Default retry policy for actions."""

    event_emitter: EventEmitterProtocol | None = None
    """Optional event emitter for monitoring."""

    def execute_workflow(
        self,
        actions: list[Any],
        context: ExecutionContext | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> WorkflowResult:
        """Execute a sequence of actions with error handling and retry.

        Automatically detects and batches consecutive IF-image_exists actions
        for parallel execution to improve performance.

        Args:
            actions: List of actions to execute
            context: Optional execution context (created if not provided)
            retry_policy: Optional retry policy override

        Returns:
            WorkflowResult with execution details
        """
        import asyncio

        context = context or ExecutionContext()
        retry_policy = retry_policy or self.retry_policy or RetryPolicy.no_retry()

        context.start_workflow()
        self._emit_event("workflow_started", action_count=len(actions))

        try:
            # Detect IF-image_exists batches for optimization
            batches = self._detect_if_image_exists_batches(actions)

            index = 0
            completed = 0

            while index < len(actions):
                # Check if current action is part of a batch
                batch_info = self._find_batch_at_index(batches, index)

                if batch_info and len(batch_info["actions"]) >= 2:
                    # Execute batch of IF actions in parallel
                    logger.info(
                        f"Auto-batching {len(batch_info['actions'])} IF-image_exists actions "
                        f"at indices {batch_info['start_index']}-{batch_info['end_index']}"
                    )

                    batch_success = asyncio.run(
                        self._execute_if_batch_optimized(batch_info, context)
                    )

                    if not batch_success and not retry_policy.continue_on_error:
                        context.complete_workflow()
                        return WorkflowResult(
                            success=False,
                            context=context,
                            completed_actions=completed,
                            failed_action_index=index,
                        )

                    # Skip past batched actions
                    completed += len(batch_info["actions"])
                    index = batch_info["end_index"] + 1
                else:
                    # Execute single action normally
                    action = actions[index]
                    result = self._execute_action_with_retry(
                        action=action, index=index, context=context, retry_policy=retry_policy
                    )

                    if not result.success:
                        # Action failed - check continue_on_error
                        if not retry_policy.continue_on_error:
                            context.complete_workflow()
                            return WorkflowResult(
                                success=False,
                                error=result.error,
                                context=context,
                                completed_actions=completed,
                                failed_action_index=index,
                            )

                    completed += 1
                    index += 1

            # All actions completed
            context.complete_workflow()
            self._emit_event("workflow_completed", statistics=context.statistics)

            return WorkflowResult(success=True, context=context, completed_actions=completed)

        except Exception as e:
            logger.error(f"Workflow execution failed with unexpected error: {e}")
            context.complete_workflow()
            self._emit_event("workflow_failed", error=str(e))

            return WorkflowResult(
                success=False,
                error=e,
                context=context,
                completed_actions=len(context.action_states),
            )

    def _execute_action_with_retry(
        self,
        action: Any,
        index: int,
        context: ExecutionContext,
        retry_policy: RetryPolicy,
    ) -> "ActionResult":
        """Execute a single action with retry logic.

        Args:
            action: Action to execute
            index: Action index in workflow
            context: Execution context
            retry_policy: Retry policy to apply

        Returns:
            ActionResult with execution details
        """
        action_name = self._get_action_name(action)
        action_state = context.start_action(index, action_name)

        self._emit_event("action_started", action=action_name, index=index)

        attempt = 0

        while True:
            try:
                # Execute the action
                success = self.action_executor.execute(action)

                if success:
                    # Action succeeded
                    context.complete_action(action_state, success=True)
                    self._emit_event(
                        "action_completed", action=action_name, index=index, attempts=attempt + 1
                    )
                    return ActionResult(success=True)

                # Action failed - check if we should retry
                if retry_policy.should_retry(attempt):
                    context.record_retry(action_state)
                    retry_policy.wait_for_retry(attempt)
                    attempt += 1
                    self._emit_event("action_retrying", action=action_name, attempt=attempt)
                    continue

                # No more retries
                context.complete_action(action_state, success=False)
                self._emit_event("action_failed", action=action_name, index=index)
                return ActionResult(success=False)

            except Exception as e:
                logger.error(f"Action {action_name} failed with error: {e}")

                # Check if we should retry based on exception
                if retry_policy.should_retry(attempt, e):
                    context.record_retry(action_state)
                    retry_policy.wait_for_retry(attempt)
                    attempt += 1
                    self._emit_event(
                        "action_retrying", action=action_name, attempt=attempt, error=str(e)
                    )
                    continue

                # No more retries
                context.complete_action(action_state, success=False, error=e)
                self._emit_event("action_failed", action=action_name, index=index, error=str(e))
                return ActionResult(success=False, error=e)

    def execute_with_condition(
        self,
        actions: list[Any],
        condition: Callable[[ExecutionContext], bool],
        context: ExecutionContext | None = None,
    ) -> WorkflowResult:
        """Execute workflow only if condition is met.

        Args:
            actions: Actions to execute
            condition: Predicate that takes context and returns True to execute
            context: Optional execution context

        Returns:
            WorkflowResult with execution details
        """
        context = context or ExecutionContext()

        if not condition(context):
            self._emit_event("workflow_skipped", reason="condition_not_met")
            return WorkflowResult(success=True, context=context, completed_actions=0)

        return self.execute_workflow(actions, context)

    def execute_parallel(
        self,
        action_groups: list[list[Any]],
        context: ExecutionContext | None = None,
    ) -> list[WorkflowResult]:
        """Execute multiple action sequences in parallel.

        Note: This is a placeholder for future parallel execution.
        Currently executes sequentially.

        Args:
            action_groups: List of action sequences
            context: Optional shared execution context

        Returns:
            List of WorkflowResult instances
        """
        context = context or ExecutionContext()
        results = []

        for actions in action_groups:
            result = self.execute_workflow(actions, context)
            results.append(result)

        return results

    def _get_action_name(self, action: Any) -> str:
        """Get a descriptive name for an action.

        Args:
            action: Action instance

        Returns:
            Action name
        """
        if hasattr(action, "name"):
            return str(action.name)  # type: ignore[no-any-return]
        if hasattr(action, "__class__"):
            return action.__class__.__name__  # type: ignore[no-any-return]
        return "UnknownAction"

    def _emit_event(self, event_type: str, **kwargs: Any) -> None:
        """Emit an event if emitter is configured.

        Args:
            event_type: Type of event
            kwargs: Event data
        """
        if self.event_emitter:
            try:
                self.event_emitter.emit(event_type, **kwargs)
            except Exception as e:
                logger.warning(f"Event emission failed: {e}")

    def _detect_if_image_exists_batches(self, actions: list[Any]) -> list[dict[str, Any]]:
        """Detect consecutive IF-image_exists actions for batching.

        Args:
            actions: List of actions to analyze

        Returns:
            List of batch information dictionaries
        """
        batches = []
        current_batch = []

        for i, action in enumerate(actions):
            # Check if this is an IF action with image_exists condition
            if self._is_if_image_exists_action(action):
                current_batch.append(
                    {
                        "index": i,
                        "action": action,
                    }
                )
            else:
                # Non-IF or non-image_exists action breaks the batch
                if len(current_batch) >= 2:  # Only batch if 2+ actions
                    batches.append(
                        {
                            "actions": current_batch,
                            "start_index": current_batch[0]["index"],
                            "end_index": current_batch[-1]["index"],
                        }
                    )
                current_batch = []

        # Handle final batch
        if len(current_batch) >= 2:
            batches.append(
                {
                    "actions": current_batch,
                    "start_index": current_batch[0]["index"],
                    "end_index": current_batch[-1]["index"],
                }
            )

        return batches

    def _is_if_image_exists_action(self, action: Any) -> bool:
        """Check if action is an IF with image_exists condition.

        Args:
            action: Action to check

        Returns:
            True if this is an IF-image_exists action
        """
        if not hasattr(action, "type") or action.type != "IF":
            return False

        # Check if it has image_exists condition
        if not hasattr(action, "config"):
            return False

        config = action.config
        if not hasattr(config, "condition"):
            return False

        condition = config.condition
        if not hasattr(condition, "type") or condition.type != "image_exists":
            return False

        return True

    def _find_batch_at_index(
        self, batches: list[dict[str, Any]], index: int
    ) -> dict[str, Any] | None:
        """Find batch that contains given index.

        Args:
            batches: List of batch information
            index: Action index to find

        Returns:
            Batch info or None if index not in any batch
        """
        for batch in batches:
            if batch["start_index"] <= index <= batch["end_index"]:
                return batch
        return None

    async def _execute_if_batch_optimized(
        self,
        batch_info: dict[str, Any],
        context: ExecutionContext,
    ) -> bool:
        """Execute batch of IF-image_exists actions in parallel.

        Args:
            batch_info: Batch information with actions
            context: Execution context

        Returns:
            True if batch executed successfully
        """
        from ..actions.control_flow.condition_evaluator import ConditionEvaluator

        actions_data = batch_info["actions"]
        logger.debug(f"Executing batch of {len(actions_data)} IF-image_exists actions")

        # Build list of conditions to evaluate
        conditions = []
        for action_data in actions_data:
            action = action_data["action"]
            conditions.append((action.id, action.config.condition))

        # Create condition evaluator
        evaluator = ConditionEvaluator(context)

        # Execute all conditions in parallel
        try:
            results = await evaluator.evaluate_multiple_image_exists_async(conditions)

            # Log results
            any_found = any(results.values())
            logger.info(
                f"Batch execution complete: {sum(results.values())} of {len(results)} images found"
            )

            # For state verification workflows, success if ANY image found
            return any_found

        except Exception as e:
            logger.error(f"Batch execution failed: {e}", exc_info=True)
            return False


class ActionResult:
    """Simple action result for internal use.

    This is a simplified version used internally by the orchestrator.
    The full ActionResult from actions package is used for action execution.
    """

    def __init__(self, success: bool = False, error: Exception | None = None) -> None:
        """Initialize action result.

        Args:
            success: Whether action succeeded
            error: Error that occurred, if any
        """
        self.success = success
        self.error = error
