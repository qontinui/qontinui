"""Enhanced TransitionExecutor with MultiState phased execution.

Extends Qontinui's TransitionExecutor with:
- Phased execution (VALIDATE → OUTGOING → ACTIVATE → INCOMING → EXIT)
- Multi-state activation support
- Success policies for complex transitions
- Rollback capabilities
"""

import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Debug logging helper - writes to file to bypass disabled logging
_DEBUG_LOG_PATH = os.path.join(tempfile.gettempdir(), "qontinui_navigation_debug.log")


def _debug_print(msg: str) -> None:
    """Write debug message to file to ensure visibility when logging is disabled."""
    try:
        from datetime import datetime

        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            f.write(f"[{timestamp}] [TRANSITION_EXECUTOR] {msg}\n")
            f.flush()
    except Exception:
        pass


from qontinui.model.transition.enhanced_state_transition import (  # noqa: E402
    TaskSequenceStateTransition,
)
from qontinui.state_management.state_memory import StateMemory  # noqa: E402

from .multistate_adapter import MultiStateAdapter  # noqa: E402

logger = logging.getLogger(__name__)


class ExecutionPhase(Enum):
    """Phases of transition execution."""

    VALIDATE = "validate"
    OUTGOING = "outgoing"
    ACTIVATE = "activate"
    INCOMING = "incoming"
    EXIT = "exit"
    COMPLETE = "complete"
    ROLLBACK = "rollback"


class SuccessPolicy(Enum):
    """Policies for determining transition success."""

    STRICT = "strict"  # All incoming transitions must succeed (Brobot-like)
    LENIENT = "lenient"  # At least one incoming transition must succeed
    THRESHOLD = "threshold"  # Configurable percentage must succeed


@dataclass
class ExecutionContext:
    """Context for a transition execution."""

    transition: TaskSequenceStateTransition
    phase: ExecutionPhase
    activated_states: set[int] | None = None
    successful_incoming: set[int] | None = None
    failed_incoming: set[int] | None = None
    rollback_needed: bool = False
    error_message: str | None = None

    def __post_init__(self):
        if self.activated_states is None:
            self.activated_states = set()
        if self.successful_incoming is None:
            self.successful_incoming = set()
        if self.failed_incoming is None:
            self.failed_incoming = set()


class EnhancedTransitionExecutor:
    """TransitionExecutor enhanced with MultiState's phased execution model.

    This executor implements the formal transition execution model:
    VALIDATE → OUTGOING → ACTIVATE → INCOMING → EXIT

    Key features:
    1. Phased execution with proper rollback
    2. Multi-state activation (ALL states activated together)
    3. Individual incoming transitions for each activated state
    4. Success policies for complex scenarios
    5. Safe rollback without state loss
    """

    def __init__(
        self,
        state_memory: StateMemory,
        multistate_adapter: MultiStateAdapter | None = None,
        workflow_executor: Any | None = None,
        success_policy: SuccessPolicy = SuccessPolicy.STRICT,
        success_threshold: float = 0.7,
    ) -> None:
        """Initialize enhanced transition executor.

        Args:
            state_memory: Qontinui's state memory
            multistate_adapter: MultiState adapter for advanced features
            workflow_executor: Executor that can run workflows (e.g., ActionExecutor)
            success_policy: Policy for determining success
            success_threshold: Threshold for THRESHOLD policy (0.0-1.0)
        """
        self.state_memory = state_memory
        self.multistate_adapter = multistate_adapter or MultiStateAdapter(state_memory)
        self.workflow_executor = workflow_executor
        self.success_policy = success_policy
        self.success_threshold = success_threshold

        logger.info(
            f"EnhancedTransitionExecutor initialized with workflow_executor: {workflow_executor is not None} (type: {type(workflow_executor).__name__ if workflow_executor else 'None'})"
        )

        # Track execution history
        self.execution_history: list[ExecutionContext] = []

        # Track phased execution callbacks
        self.phase_callbacks: dict[ExecutionPhase, list[Callable[..., Any]]] = {
            phase: [] for phase in ExecutionPhase
        }

    def execute_transition(
        self, transition: TaskSequenceStateTransition, dry_run: bool = False
    ) -> bool:
        """Execute a transition with phased execution model.

        Args:
            transition: Transition to execute
            dry_run: If True, validate only without execution

        Returns:
            True if transition successful, False otherwise
        """
        context = ExecutionContext(transition=transition, phase=ExecutionPhase.VALIDATE)

        try:
            # Phase 1: VALIDATE
            if not self._execute_validate(context):
                logger.info(f"Validation failed for transition {transition.name}")
                return False

            if dry_run:
                logger.info(f"Dry run successful for transition {transition.name}")
                return True

            # Phase 2: OUTGOING
            context.phase = ExecutionPhase.OUTGOING
            if not self._execute_outgoing(context):
                logger.warning(f"Outgoing transitions failed for {transition.name}")
                return False

            # Phase 3: ACTIVATE (infallible - just memory operation)
            context.phase = ExecutionPhase.ACTIVATE
            self._execute_activate(context)

            # Phase 4: INCOMING (may partially fail)
            context.phase = ExecutionPhase.INCOMING
            success = self._execute_incoming(context)

            # Phase 5: EXIT (only if incoming succeeded per policy)
            if success:
                context.phase = ExecutionPhase.EXIT
                self._execute_exit(context)

                context.phase = ExecutionPhase.COMPLETE
                logger.info(f"Transition {transition.name} completed successfully")
            else:
                # Rollback activation if incoming failed
                context.rollback_needed = True
                context.phase = ExecutionPhase.ROLLBACK
                self._execute_rollback(context)
                logger.warning(f"Transition {transition.name} rolled back")

            # Record execution
            self.execution_history.append(context)
            return success

        except Exception as e:
            logger.error(f"Error executing transition {transition.name}: {e}")
            context.error_message = str(e)

            # Attempt rollback on error
            if context.phase in [ExecutionPhase.ACTIVATE, ExecutionPhase.INCOMING]:
                context.rollback_needed = True
                context.phase = ExecutionPhase.ROLLBACK
                self._execute_rollback(context)

            self.execution_history.append(context)
            return False

    def _execute_validate(self, context: ExecutionContext) -> bool:
        """Validate transition can be executed.

        Args:
            context: Execution context

        Returns:
            True if validation successful
        """
        transition = context.transition

        # Check preconditions
        if transition.from_states:
            current_states = self.state_memory.active_states
            if not any(state_id in current_states for state_id in transition.from_states):
                logger.debug("Precondition failed: no from_states active")
                return False

        # Check target states are reachable
        for state_id in transition.activate:
            if not self.state_memory.state_service:
                continue
            state = self.state_memory.state_service.get_state(state_id)
            if not state:
                logger.warning(f"Target state {state_id} not found")
                return False

        # Run phase callbacks
        for callback in self.phase_callbacks[ExecutionPhase.VALIDATE]:
            if not callback(context):
                return False

        return True

    def _execute_outgoing(self, context: ExecutionContext) -> bool:
        """Execute outgoing transitions from current states.

        This executes the workflows associated with the transition.

        Args:
            context: Execution context

        Returns:
            True if outgoing transitions successful
        """
        transition = context.transition

        _debug_print(
            f"_execute_outgoing called for transition '{transition.name}' (id={transition.id})"
        )
        _debug_print(f"  workflow_executor is None: {self.workflow_executor is None}")
        _debug_print(f"  transition has workflow_ids attr: {hasattr(transition, 'workflow_ids')}")
        if hasattr(transition, "workflow_ids"):
            _debug_print(f"  transition.workflow_ids: {transition.workflow_ids}")
        logger.debug(f"_execute_outgoing called for transition '{transition.name}'")
        logger.debug(f"  workflow_executor is None: {self.workflow_executor is None}")
        logger.debug(f"  transition has workflow_ids attr: {hasattr(transition, 'workflow_ids')}")
        if hasattr(transition, "workflow_ids"):
            logger.debug(f"  transition.workflow_ids: {transition.workflow_ids}")

        # Check if transition has workflows to execute
        if not hasattr(transition, "workflow_ids") or not transition.workflow_ids:
            _debug_print(
                f"ERROR: Transition '{transition.name}' (id={transition.id}) has no workflows to execute!"
            )
            _debug_print(
                "  This usually means the transition references a workflow ID that doesn't exist in the config."
            )
            _debug_print(
                "  Check the 'workflows' array in your config file and ensure all referenced IDs are defined."
            )
            logger.warning(
                f"Transition '{transition.name}' has no workflows to execute. "
                "Navigation will fail unless workflows are linked to this transition."
            )
            return False

        # Execute each workflow associated with this transition
        for workflow_id in transition.workflow_ids:
            logger.info(f"Executing workflow '{workflow_id}' for transition '{transition.name}'")

            try:
                if self.workflow_executor:
                    # Get state names for target states
                    target_state_names = []
                    if self.state_memory.state_service:
                        for state_id in transition.activate:
                            state = self.state_memory.state_service.get_state(state_id)
                            if state:
                                target_state_names.append(state.name)
                            else:
                                target_state_names.append(str(state_id))
                    else:
                        # Fallback to IDs if no state service
                        target_state_names = [str(s) for s in transition.activate]

                    # Build transition context for runner
                    transition_context = {
                        "transition_id": f"outgoing-{transition.name}-{workflow_id}",
                        "transition_name": "Outgoing Transition",  # Just the type, not the states
                        "transition_type": "outgoing",
                        "target_states": target_state_names,
                        "source_state": transition.name,
                    }

                    # Use the injected workflow executor with transition context
                    result = self.workflow_executor.execute_workflow(
                        workflow_id, transition_context
                    )
                    if not result.get("success", False):
                        logger.error(f"Workflow '{workflow_id}' failed")
                        return False
                else:
                    # No workflow executor available - cannot proceed
                    # Direct workflow execution requires ActionExecutor or similar
                    from qontinui import registry

                    workflow = registry.get_workflow(workflow_id)
                    if not workflow:
                        logger.error(f"Workflow '{workflow_id}' not found in registry")
                        return False

                    logger.error(
                        f"No workflow_executor provided. EnhancedTransitionExecutor requires "
                        f"a workflow_executor to execute workflows. Cannot execute workflow '{workflow_id}'"
                    )
                    return False

            except Exception as e:
                logger.error(f"Error executing workflow '{workflow_id}': {e}", exc_info=True)
                return False

        # Run phase callbacks
        for callback in self.phase_callbacks[ExecutionPhase.OUTGOING]:
            if not callback(context):
                return False

        logger.debug(f"Outgoing transitions executed successfully for {transition.name}")
        return True

    def _execute_activate(self, context: ExecutionContext) -> None:
        """Activate ALL target states atomically.

        This is infallible - just a memory operation.

        Args:
            context: Execution context
        """
        transition = context.transition

        # Activate ALL states together (atomic operation)
        if transition.activate is not None:
            for state_id in transition.activate:
                self.state_memory.active_states.add(state_id)
                if context.activated_states is not None:
                    context.activated_states.add(state_id)

        # Sync with MultiState
        if self.multistate_adapter:
            self.multistate_adapter.sync_with_state_memory()

        # Run phase callbacks
        for callback in self.phase_callbacks[ExecutionPhase.ACTIVATE]:
            callback(context)

        activated_count = len(context.activated_states) if context.activated_states else 0
        logger.debug(f"Activated {activated_count} states")

    def _execute_incoming(self, context: ExecutionContext) -> bool:
        """Execute incoming transitions for EACH activated state.

        Args:
            context: Execution context

        Returns:
            True if successful per policy, False otherwise
        """

        # Execute incoming transition for EACH activated state
        if context.activated_states is not None:
            for state_id in context.activated_states:
                if self._execute_single_incoming(state_id, context):
                    if context.successful_incoming is not None:
                        context.successful_incoming.add(state_id)
                else:
                    if context.failed_incoming is not None:
                        context.failed_incoming.add(state_id)

        # Run phase callbacks
        for callback in self.phase_callbacks[ExecutionPhase.INCOMING]:
            callback(context)

        # Determine success based on policy
        return self._evaluate_success(context)

    def _execute_single_incoming(self, state_id: int, context: ExecutionContext) -> bool:
        """Execute incoming transition workflows for a single state.

        IncomingTransitions are verification workflows that run when entering a state.
        They verify that the state should actually be activated.

        Args:
            state_id: State to execute incoming for
            context: Execution context

        Returns:
            True if all incoming transitions successful, False otherwise
        """
        if not self.state_memory.state_service:
            logger.warning("No state_service available")
            return True  # No way to verify, assume success

        state = self.state_memory.state_service.get_state(state_id)
        if not state:
            logger.error(f"State {state_id} not found")
            return False

        # Check if state has incoming transitions
        if not hasattr(state, "incoming_transitions") or not state.incoming_transitions:
            logger.debug(f"State {state.name} has no incoming transitions - verification skipped")
            return True  # No incoming transitions = no verification needed = success

        logger.info(
            f"Executing {len(state.incoming_transitions)} incoming transitions for state {state.name}"
        )

        # Execute all incoming transitions for this state
        for incoming_transition in state.incoming_transitions:
            if (
                not hasattr(incoming_transition, "workflow_ids")
                or not incoming_transition.workflow_ids
            ):
                logger.warning("IncomingTransition has no workflows - skipping")
                continue

            # Execute each workflow in the incoming transition
            for workflow_id in incoming_transition.workflow_ids:
                logger.debug(
                    f"About to execute incoming workflow '{workflow_id}' for state '{state.name}'"
                )

                if not self.workflow_executor:
                    logger.error("No workflow_executor available for incoming transition")
                    return False

                # Build transition context for runner
                transition_context = {
                    "transition_id": f"incoming-{state.name}-{workflow_id}",
                    "transition_name": "Incoming Transition",  # Just the type, not the state
                    "transition_type": "incoming",
                    "target_states": [state.name],
                    "source_state": None,  # Incoming transitions don't have a specific source
                }

                logger.debug(
                    f"Calling workflow_executor.execute_workflow('{workflow_id}') with transition context..."
                )
                result = self.workflow_executor.execute_workflow(workflow_id, transition_context)
                logger.debug(f"Workflow '{workflow_id}' returned: {result}")

                if not result.get("success", False):
                    logger.error(
                        f"Incoming workflow '{workflow_id}' failed for state '{state.name}'"
                    )
                    return False

        logger.info(f"All incoming transitions succeeded for state {state.name}")
        return True

    def _execute_exit(self, context: ExecutionContext) -> None:
        """Exit states that need to be deactivated.

        Only called if incoming transitions succeeded.

        Args:
            context: Execution context
        """
        transition = context.transition

        # Exit specified states
        for state_id in transition.exit:
            self.state_memory.active_states.discard(state_id)

        # Sync with MultiState
        if self.multistate_adapter:
            self.multistate_adapter.sync_with_state_memory()

        # Run phase callbacks
        for callback in self.phase_callbacks[ExecutionPhase.EXIT]:
            callback(context)

        logger.debug(f"Exited {len(transition.exit)} states")

    def _execute_rollback(self, context: ExecutionContext) -> None:
        """Rollback activation if incoming transitions failed.

        Args:
            context: Execution context
        """
        # Remove activated states from memory
        if context.activated_states is not None:
            for state_id in context.activated_states:
                self.state_memory.active_states.discard(state_id)

        # Sync with MultiState
        if self.multistate_adapter:
            self.multistate_adapter.sync_with_state_memory()

        activated_count = len(context.activated_states) if context.activated_states else 0
        logger.info(f"Rolled back activation of {activated_count} states")

    def _evaluate_success(self, context: ExecutionContext) -> bool:
        """Evaluate transition success based on policy.

        Args:
            context: Execution context

        Returns:
            True if successful per policy
        """
        total = len(context.activated_states) if context.activated_states else 0
        successful = len(context.successful_incoming) if context.successful_incoming else 0

        if total == 0:
            return True  # No states to activate is success

        if self.success_policy == SuccessPolicy.STRICT:
            # All must succeed (Brobot-like)
            return successful == total

        elif self.success_policy == SuccessPolicy.LENIENT:
            # At least one must succeed
            return successful > 0

        elif self.success_policy == SuccessPolicy.THRESHOLD:
            # Percentage must succeed
            ratio = successful / total
            return ratio >= self.success_threshold

        return False

    def register_phase_callback(self, phase: ExecutionPhase, callback) -> None:
        """Register callback for execution phase.

        Callbacks receive ExecutionContext and return bool for validation.

        Args:
            phase: Phase to register for
            callback: Callback function
        """
        self.phase_callbacks[phase].append(callback)

    def execute_group_transition(self, group_name: str, activate: bool = True) -> bool:
        """Execute transition for an entire state group.

        Args:
            group_name: Name of group
            activate: True to activate, False to deactivate

        Returns:
            True if successful
        """
        if not hasattr(self.state_memory, "state_groups"):
            logger.warning("State groups not supported in state memory")
            return False

        if group_name not in self.state_memory.state_groups:
            logger.warning(f"Unknown group: {group_name}")
            return False

        group_states = self.state_memory.state_groups[group_name]

        # Create synthetic transition
        transition = TaskSequenceStateTransition(
            name=f"{'Activate' if activate else 'Deactivate'} group {group_name}",
            activate=group_states if activate else set(),
            exit=group_states if not activate else set(),
            path_cost=1,  # Low cost for group operations
        )

        return self.execute_transition(transition)

    def get_execution_statistics(self) -> dict[str, Any]:
        """Get statistics about transition executions.

        Returns:
            Dictionary with execution statistics
        """
        total = len(self.execution_history)
        successful = sum(
            1 for ctx in self.execution_history if ctx.phase == ExecutionPhase.COMPLETE
        )
        rolled_back = sum(1 for ctx in self.execution_history if ctx.rollback_needed)

        phase_counts = {}
        for phase in ExecutionPhase:
            phase_counts[phase.value] = sum(
                1 for ctx in self.execution_history if ctx.phase == phase
            )

        return {
            "total_executions": total,
            "successful": successful,
            "failed": total - successful,
            "rolled_back": rolled_back,
            "success_rate": successful / total if total > 0 else 0,
            "phase_counts": phase_counts,
            "success_policy": self.success_policy.value,
        }
