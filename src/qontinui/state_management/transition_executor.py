"""Transition execution orchestration.

This module handles transition execution with validation, context preparation,
result building, and event emission coordination.

Architecture:
    - TransitionExecutor: Orchestrates transition execution
    - Delegates to EnhancedTransitionExecutor for actual execution
    - Builds rich result objects with state changes
    - Coordinates event emission

Key Features:
    1. Transition validation before execution
    2. Context preparation and state tracking
    3. Execution via EnhancedTransitionExecutor
    4. Result building with state changes
    5. Event emission coordination
    6. Error handling and logging

Example:
    >>> executor = TransitionExecutor(config, state_memory, transition_executor, event_emitter)
    >>> result = executor.execute_transition("login_transition", callback)
    >>> if result.success:
    ...     print(f"Activated: {result.activated_states}")
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from qontinui.exceptions import StateException

logger = logging.getLogger(__name__)


@dataclass
class TransitionExecutionResult:
    """Result of a transition execution.

    Attributes:
        success: Whether operation succeeded
        error_message: Error message if failed
        context: Additional context information
        transition_id: ID of executed transition
        transition_name: Name of executed transition
        activated_states: States activated by transition
        deactivated_states: States deactivated by transition
        active_states_after: All active states after transition
    """

    success: bool
    error_message: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    transition_id: str | None = None
    transition_name: str | None = None
    activated_states: set[int] = field(default_factory=set)
    deactivated_states: set[int] = field(default_factory=set)
    active_states_after: set[int] = field(default_factory=set)


class TransitionExecutor:
    """Orchestrates transition execution with validation and result building.

    Handles the complete transition execution flow including validation,
    context preparation, execution, result building, and event emission.

    Example:
        >>> executor = TransitionExecutor(config, state_memory, transition_executor, event_emitter)
        >>> result = executor.execute_transition("login", callback)
        >>> if result.success:
        ...     print(f"Active states: {result.active_states_after}")
    """

    def __init__(
        self,
        validator: Any,
        state_memory: Any,
        transition_executor: Any,
        event_emitter: Any,
    ) -> None:
        """Initialize TransitionExecutor.

        Args:
            validator: StateValidator for validation
            state_memory: EnhancedStateMemory for state tracking
            transition_executor: EnhancedTransitionExecutor for execution
            event_emitter: StateEventEmitter for event emission
        """
        self.validator = validator
        self.state_memory = state_memory
        self.transition_executor = transition_executor
        self.event_emitter = event_emitter

    def execute_transition(
        self,
        transition_id: str,
        emit_event_callback: Callable[[str, dict], None] | None = None,
    ) -> TransitionExecutionResult:
        """Execute a transition by ID.

        Executes the transition using EnhancedTransitionExecutor with phased
        execution (VALIDATE → OUTGOING → ACTIVATE → INCOMING → EXIT).

        Args:
            transition_id: ID of transition to execute
            emit_event_callback: Optional callback for emitting events

        Returns:
            TransitionExecutionResult with execution details

        Example:
            >>> result = executor.execute_transition("login_transition")
            >>> if result.success:
            ...     print(f"Activated: {result.activated_states}")
            >>> else:
            ...     print(f"Failed: {result.error_message}")
        """
        logger.info(f"Executing transition: {transition_id}")

        # Emit starting event
        self.event_emitter.emit_transition_start(transition_id, emit_event_callback)

        try:
            # Validate transition exists
            transition = self.validator.validate_transition(transition_id)
            if not transition:
                return self._handle_transition_not_found(
                    transition_id, emit_event_callback
                )

            # Prepare execution context
            context = self._prepare_context(transition_id)

            # Execute transition
            success = self._execute(transition)

            # Build result from execution
            result = self._build_result(transition_id, transition, context, success)

            # Emit completion events and log
            self.event_emitter.emit_transition_complete(result, emit_event_callback)
            self._log_result(result)

            return result

        except StateException as e:
            logger.error(f"State error executing transition '{transition_id}': {e}")
            self.event_emitter.emit_transition_failed(
                transition_id, str(e), emit_event_callback
            )
            return TransitionExecutionResult(
                success=False, transition_id=transition_id, error_message=str(e)
            )

        except Exception as e:
            logger.error(
                f"Unexpected error executing transition '{transition_id}': {e}",
                exc_info=True,
            )
            error_msg = f"Unexpected error: {e}"
            self.event_emitter.emit_transition_failed(
                transition_id, error_msg, emit_event_callback
            )
            return TransitionExecutionResult(
                success=False, transition_id=transition_id, error_message=error_msg
            )

    def _prepare_context(self, transition_id: str) -> dict[str, Any]:
        """Prepare execution context by capturing pre-execution state.

        Args:
            transition_id: ID of transition being executed

        Returns:
            Context dict with states_before
        """
        return {"states_before": set(self.state_memory.active_states)}

    def _execute(self, transition: Any) -> bool:
        """Execute the transition via EnhancedTransitionExecutor.

        Args:
            transition: Transition object to execute

        Returns:
            True if execution succeeded, False otherwise
        """
        return cast(bool, self.transition_executor.execute_transition(transition))

    def _build_result(
        self,
        transition_id: str,
        transition: Any,
        context: dict[str, Any],
        success: bool,
    ) -> TransitionExecutionResult:
        """Build result object from transition execution.

        Args:
            transition_id: ID of executed transition
            transition: Transition object
            context: Execution context with states_before
            success: Whether execution succeeded

        Returns:
            TransitionExecutionResult with execution details
        """
        states_before = context["states_before"]
        states_after = set(self.state_memory.active_states)

        activated = states_after - states_before
        deactivated = states_before - states_after

        return TransitionExecutionResult(
            success=success,
            transition_id=transition_id,
            transition_name=getattr(transition, "name", transition_id),
            activated_states=activated,
            deactivated_states=deactivated,
            active_states_after=states_after,
        )

    def _handle_transition_not_found(
        self,
        transition_id: str,
        emit_event_callback: Callable[[str, dict], None] | None,
    ) -> TransitionExecutionResult:
        """Handle case where transition is not found.

        Args:
            transition_id: ID of transition that was not found
            emit_event_callback: Optional callback for emitting events

        Returns:
            Failed TransitionExecutionResult
        """
        error_msg = f"Transition '{transition_id}' not found in configuration"
        self.event_emitter.emit_transition_failed(
            transition_id, error_msg, emit_event_callback
        )
        return TransitionExecutionResult(success=False, error_message=error_msg)

    def _log_result(self, result: TransitionExecutionResult) -> None:
        """Log transition execution result.

        Args:
            result: Transition execution result
        """
        if result.success:
            logger.info(
                f"Transition '{result.transition_id}' completed successfully. "
                f"Activated: {result.activated_states}, Deactivated: {result.deactivated_states}"
            )
        else:
            logger.warning(f"Transition '{result.transition_id}' failed")
