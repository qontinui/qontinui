"""TransitionExecutor - Orchestrates complex state transitions.

This module implements the complete transition execution flow including:
- Outgoing transition execution
- Multi-state activation
- Incoming transition execution for all activated states
- State visibility management
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from qontinui.model.transition.enhanced_joint_table import StateTransitionsJointTable
from qontinui.model.transition.enhanced_state_transition import (
    StateTransition,
    StaysVisible,
    TransitionContext,
    TransitionResult,
)

if TYPE_CHECKING:
    from qontinui.state_management.enhanced_active_state_set import (
        EnhancedActiveStateSet,
    )
    from qontinui.state_management.state_memory import StateMemory
    from qontinui.state_management.state_visibility_manager import (
        StateVisibilityManager,
    )

logger = logging.getLogger(__name__)


@dataclass
class TransitionExecutor:
    """Orchestrates complex state transitions with multi-state support.

    This is a key component that implements Brobot's transition execution
    pattern where:
    1. Outgoing transition is executed from current states
    2. ALL target states are activated together
    3. Incoming transitions execute for ALL activated states
    4. State visibility is properly managed

    This ensures complex GUI states (like workspaces with multiple panels)
    are properly managed as cohesive units.
    """

    joint_table: StateTransitionsJointTable
    active_states: "EnhancedActiveStateSet | None" = None  # Will be injected
    state_memory: "StateMemory | None" = None  # Will be injected
    visibility_manager: "StateVisibilityManager | None" = None  # Will be injected

    # Execution options
    execute_incoming: bool = True  # Execute incoming transitions for activated states
    batch_activation: bool = True  # Activate all states together
    track_timing: bool = True  # Track execution timing

    def execute_transition(
        self, transition: StateTransition, current_states: set[int] | None = None
    ) -> TransitionResult:
        """Execute a complete state transition.

        This is the main entry point that orchestrates the full transition:
        1. Execute outgoing transition from current states
        2. Activate ALL target states
        3. Execute incoming transitions for ALL activated states
        4. Update state visibility and memory

        Args:
            transition: The transition to execute
            current_states: Override current active states (optional)

        Returns:
            TransitionResult with complete execution details
        """
        start_time = time.time() if self.track_timing else 0

        # Get current states if not provided
        if current_states is None:
            current_states = self._get_current_active_states()

        # Create execution context
        context = TransitionContext(
            current_states=current_states,
            target_states=transition.activate,
            exit_states=transition.exit,
        )

        logger.info(f"Executing transition: {transition}")
        logger.debug(f"Current states: {current_states}")
        logger.debug(f"Target states: {transition.activate}")

        # Initialize result
        overall_result = TransitionResult(successful=True)

        try:
            # Phase 1: Execute outgoing transition
            outgoing_result = self._execute_outgoing(transition, context)
            overall_result.metadata["outgoing"] = outgoing_result

            if not outgoing_result.successful:
                overall_result.successful = False
                overall_result.errors.extend(outgoing_result.errors)
                logger.error(f"Outgoing transition failed: {outgoing_result.errors}")
                return overall_result

            # Phase 2: Activate all target states
            if self.batch_activation:
                activated_states = self._activate_states_batch(transition.activate)
            else:
                activated_states = self._activate_states_sequential(transition.activate)

            overall_result.activated_states = activated_states
            logger.info(f"Activated states: {activated_states}")

            # Phase 3: Deactivate exit states
            deactivated_states = self._deactivate_states(transition.exit)
            overall_result.deactivated_states = deactivated_states
            logger.info(f"Deactivated states: {deactivated_states}")

            # Phase 4: Execute incoming transitions for all activated states
            if self.execute_incoming and activated_states:
                incoming_results = self._execute_incoming_transitions(activated_states)
                overall_result.metadata["incoming"] = incoming_results

                # Check for any incoming failures
                for state_id, result in incoming_results.items():
                    if not result.successful:
                        logger.warning(
                            f"Incoming transition failed for state {state_id}: "
                            f"{result.errors}"
                        )

            # Phase 5: Update state visibility
            if self.active_states:
                hidden_states = self._update_state_visibility(transition, context)
                overall_result.hidden_states = hidden_states
                logger.debug(f"Hidden states: {hidden_states}")

            # Phase 6: Update state memory
            if self.state_memory:
                self._update_state_memory(overall_result)

            logger.info("Transition execution completed successfully")

        except Exception as e:
            overall_result.successful = False
            overall_result.add_error(f"Transition execution failed: {str(e)}")
            logger.exception("Transition execution failed with exception")

        finally:
            if self.track_timing:
                overall_result.execution_time = time.time() - start_time
                logger.debug(f"Execution time: {overall_result.execution_time:.3f}s")

        return overall_result

    def _execute_outgoing(
        self, transition: StateTransition, context: TransitionContext
    ) -> TransitionResult:
        """Execute the outgoing transition.

        Args:
            transition: The transition to execute
            context: Execution context

        Returns:
            Result of outgoing transition
        """
        logger.debug("Executing outgoing transition")

        try:
            # Check if transition can be executed
            if not transition.can_execute(context):
                result = TransitionResult(successful=False)
                result.add_error("Transition cannot be executed: preconditions not met")
                return result

            # Execute the transition
            result = transition.execute(context)

            # Track success/failure
            if result.successful:
                transition.record_success()
            else:
                transition.record_failure()

            return result

        except Exception as e:
            result = TransitionResult(successful=False)
            result.add_error(f"Outgoing transition exception: {str(e)}")
            transition.record_failure()
            return result

    def _activate_states_batch(self, state_ids: set[int]) -> set[int]:
        """Activate multiple states as a batch.

        Args:
            state_ids: Set of state IDs to activate

        Returns:
            Set of successfully activated state IDs
        """
        logger.debug(f"Batch activating states: {state_ids}")

        if not self.active_states:
            logger.warning("ActiveStateSet not available, cannot activate states")
            return set()

        activated = set()
        for state_id in state_ids:
            self.active_states.add_state(cast(Any, state_id))  # type: ignore[arg-type]
            activated.add(state_id)

        return activated

    def _activate_states_sequential(self, state_ids: set[int]) -> set[int]:
        """Activate states one by one.

        Args:
            state_ids: Set of state IDs to activate

        Returns:
            Set of successfully activated state IDs
        """
        logger.debug(f"Sequentially activating states: {state_ids}")

        if not self.active_states:
            logger.warning("ActiveStateSet not available, cannot activate states")
            return set()

        activated = set()
        for state_id in state_ids:
            # Small delay between activations if needed
            self.active_states.add_state(cast(Any, state_id))  # type: ignore[arg-type]
            activated.add(state_id)
            time.sleep(0.01)  # Small delay for sequential activation

        return activated

    def _deactivate_states(self, state_ids: set[int]) -> set[int]:
        """Deactivate specified states.

        Args:
            state_ids: Set of state IDs to deactivate

        Returns:
            Set of successfully deactivated state IDs
        """
        logger.debug(f"Deactivating states: {state_ids}")

        if not self.active_states:
            logger.warning("ActiveStateSet not available, cannot deactivate states")
            return set()

        deactivated = set()
        for state_id in state_ids:
            self.active_states.remove_state(cast(Any, state_id))  # type: ignore[arg-type]
            deactivated.add(state_id)

        return deactivated

    def _execute_incoming_transitions(
        self, activated_states: set[int]
    ) -> dict[int, TransitionResult]:
        """Execute incoming transitions for all activated states.

        This is a key feature from Brobot - after activating states,
        their incoming transitions are executed to properly initialize them.

        Args:
            activated_states: Set of newly activated state IDs

        Returns:
            Dictionary mapping state ID to incoming transition results
        """
        logger.debug(f"Executing incoming transitions for states: {activated_states}")

        results = {}

        # Get incoming transitions for each activated state
        incoming_map = self.joint_table.get_incoming_transitions(activated_states)

        for state_id, transitions in incoming_map.items():
            logger.debug(
                f"State {state_id} has {len(transitions)} incoming transitions"
            )

            # Execute first valid incoming transition
            for transition in transitions:
                # Create context for incoming execution
                context = TransitionContext(
                    current_states={state_id},
                    target_states={state_id},
                    exit_states=set(),
                )

                # Try to execute
                if transition.can_execute(context):
                    result = transition.execute(context)
                    results[state_id] = result

                    if result.successful:
                        logger.debug(
                            f"Incoming transition succeeded for state {state_id}"
                        )
                        break  # Only execute first successful incoming
                    else:
                        logger.warning(
                            f"Incoming transition failed for state {state_id}: "
                            f"{result.errors}"
                        )
                else:
                    logger.debug(
                        f"Incoming transition cannot execute for state {state_id}"
                    )

            # If no incoming transition executed, create a default result
            if state_id not in results:
                results[state_id] = TransitionResult(
                    successful=True,
                    metadata={"note": "No incoming transition executed"},
                )

        return results

    def _update_state_visibility(
        self, transition: StateTransition, context: TransitionContext
    ) -> set[int]:
        """Update state visibility based on transition settings.

        Args:
            transition: The executed transition
            context: Execution context

        Returns:
            Set of hidden state IDs
        """
        if not self.active_states:
            return set()

        hidden_states = set()

        # Determine visibility behavior
        stays_visible = transition.stays_visible

        if stays_visible == StaysVisible.FALSE:
            # Hide current states that aren't being exited
            states_to_hide = context.current_states - transition.exit

            for state_id in states_to_hide:
                if self.active_states.hide_state(state_id):
                    hidden_states.add(state_id)

        elif stays_visible == StaysVisible.TRUE:
            # Ensure current states remain visible
            for state_id in context.current_states:
                self.active_states.show_state(state_id)

        return hidden_states

    def _update_state_memory(self, result: TransitionResult) -> None:
        """Update state memory with transition results.

        Args:
            result: The transition result
        """
        if not self.state_memory:
            return

        # Update memory with activated states
        for state_id in result.activated_states:
            self.state_memory.add_active_state(state_id)

        # Update memory with deactivated states
        for state_id in result.deactivated_states:
            self.state_memory.remove_active_state(state_id)

        # Note: Hidden states are tracked in active_states.hidden_states,
        # not in state_memory

    def _get_current_active_states(self) -> set[int]:
        """Get current active states.

        Returns:
            Set of active state IDs
        """
        if self.active_states:
            # EnhancedActiveStateSet returns set[int] directly
            return self.active_states.get_active_states()
        return set()

    def execute_group_transition(
        self, group_name: str, transition: StateTransition
    ) -> TransitionResult:
        """Execute a transition for a state group.

        Args:
            group_name: Name of the state group
            transition: Transition to execute

        Returns:
            TransitionResult with execution details
        """
        # Get states in the group
        group_states = self.joint_table.get_group(group_name)

        if not group_states:
            result = TransitionResult(successful=False)
            result.add_error(f"Unknown state group: {group_name}")
            return result

        logger.info(f"Executing transition for group '{group_name}': {group_states}")

        # Execute transition with group states as current
        return self.execute_transition(transition, group_states)

    def __str__(self) -> str:
        """String representation."""
        return (
            f"TransitionExecutor(execute_incoming={self.execute_incoming}, "
            f"batch_activation={self.batch_activation})"
        )
