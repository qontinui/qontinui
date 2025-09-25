"""Enhanced TransitionExecutor with MultiState phased execution.

Extends Qontinui's TransitionExecutor with:
- Phased execution (VALIDATE → OUTGOING → ACTIVATE → INCOMING → EXIT)
- Multi-state activation support
- Success policies for complex transitions
- Rollback capabilities
"""

import logging
from dataclasses import dataclass
from enum import Enum

from qontinui.model.transition.enhanced_state_transition import StateTransition
from qontinui.state_management.state_memory import StateMemory

from .multistate_adapter import MultiStateAdapter

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

    transition: StateTransition
    phase: ExecutionPhase
    activated_states: set[int] = None
    successful_incoming: set[int] = None
    failed_incoming: set[int] = None
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
        success_policy: SuccessPolicy = SuccessPolicy.STRICT,
        success_threshold: float = 0.7,
    ):
        """Initialize enhanced transition executor.

        Args:
            state_memory: Qontinui's state memory
            multistate_adapter: MultiState adapter for advanced features
            success_policy: Policy for determining success
            success_threshold: Threshold for THRESHOLD policy (0.0-1.0)
        """
        self.state_memory = state_memory
        self.multistate_adapter = multistate_adapter or MultiStateAdapter(state_memory)
        self.success_policy = success_policy
        self.success_threshold = success_threshold

        # Track execution history
        self.execution_history: list[ExecutionContext] = []

        # Track phased execution callbacks
        self.phase_callbacks: dict[ExecutionPhase, list] = {phase: [] for phase in ExecutionPhase}

    def execute_transition(self, transition: StateTransition, dry_run: bool = False) -> bool:
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

        Args:
            context: Execution context

        Returns:
            True if outgoing transitions successful
        """
        transition = context.transition

        # In Qontinui, outgoing transitions are the exit actions
        # This would execute any cleanup or state exit logic

        # Run phase callbacks
        for callback in self.phase_callbacks[ExecutionPhase.OUTGOING]:
            if not callback(context):
                return False

        logger.debug(f"Outgoing transitions executed for {transition.name}")
        return True

    def _execute_activate(self, context: ExecutionContext) -> None:
        """Activate ALL target states atomically.

        This is infallible - just a memory operation.

        Args:
            context: Execution context
        """
        transition = context.transition

        # Activate ALL states together (atomic operation)
        for state_id in transition.activate:
            self.state_memory.active_states.add(state_id)
            context.activated_states.add(state_id)

        # Sync with MultiState
        if self.multistate_adapter:
            self.multistate_adapter.sync_with_state_memory()

        # Run phase callbacks
        for callback in self.phase_callbacks[ExecutionPhase.ACTIVATE]:
            callback(context)

        logger.debug(f"Activated {len(context.activated_states)} states")

    def _execute_incoming(self, context: ExecutionContext) -> bool:
        """Execute incoming transitions for EACH activated state.

        Args:
            context: Execution context

        Returns:
            True if successful per policy, False otherwise
        """

        # Execute incoming transition for EACH activated state
        for state_id in context.activated_states:
            if self._execute_single_incoming(state_id, context):
                context.successful_incoming.add(state_id)
            else:
                context.failed_incoming.add(state_id)

        # Run phase callbacks
        for callback in self.phase_callbacks[ExecutionPhase.INCOMING]:
            callback(context)

        # Determine success based on policy
        return self._evaluate_success(context)

    def _execute_single_incoming(self, state_id: int, context: ExecutionContext) -> bool:
        """Execute incoming transition for a single state.

        Args:
            state_id: State to execute incoming for
            context: Execution context

        Returns:
            True if incoming successful
        """
        # In Qontinui, this would execute the state's entry actions
        # For now, we simulate with a success check

        if not self.state_memory.state_service:
            return True

        state = self.state_memory.state_service.get_state(state_id)
        if not state:
            return False

        # Execute state's incoming transition
        # This would run state.on_entry() or similar
        logger.debug(f"Executing incoming for state {state.name}")

        # Simulate execution (in real implementation, would call state methods)
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
        for state_id in context.activated_states:
            self.state_memory.active_states.discard(state_id)

        # Sync with MultiState
        if self.multistate_adapter:
            self.multistate_adapter.sync_with_state_memory()

        logger.info(f"Rolled back activation of {len(context.activated_states)} states")

    def _evaluate_success(self, context: ExecutionContext) -> bool:
        """Evaluate transition success based on policy.

        Args:
            context: Execution context

        Returns:
            True if successful per policy
        """
        total = len(context.activated_states)
        successful = len(context.successful_incoming)

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
        transition = StateTransition(
            id=-1,  # Synthetic ID
            name=f"{'Activate' if activate else 'Deactivate'} group {group_name}",
            activate=group_states if activate else set(),
            exit=group_states if not activate else set(),
            score=0.1,  # Low cost for group operations
        )

        return self.execute_transition(transition)

    def get_execution_statistics(self) -> dict:
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
