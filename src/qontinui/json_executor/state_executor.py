"""State machine executor for Qontinui automation."""

import logging
from typing import Any

from ..action_executors import DelegatingActionExecutor
from ..wrappers import TimeWrapper
from .config_parser import (
    IncomingTransition,
    OutgoingTransition,
    QontinuiConfig,
    Transition,
    Workflow,
)

logger = logging.getLogger(__name__)


class StateExecutor:
    """Executes state machine-based automation workflows.

    StateExecutor manages the complete lifecycle of state-based automation including
    state activation/deactivation, state verification through image matching, transition
    discovery and execution, and workflow execution within transitions.

    The executor implements a state machine pattern:
        1. Initialize to initial state (marked with is_initial=True)
        2. Verify current state is active (by checking identifying images)
        3. Find applicable outgoing transitions from current state
        4. Execute transition's workflow (sequence of actions)
        5. Verify incoming transitions to target state (if any)
        6. Activate target state(s) and deactivate origin state
        7. Repeat until final state reached or no transitions available

    State Visibility Rules:
        - Origin state deactivated by default after transition
        - Set stays_visible=True to keep origin state active
        - Multiple states can be active simultaneously
        - activate_states and deactivate_states control parallel states

    Attributes:
        config: Automation configuration with states, transitions, workflows, images.
        active_states: Set of currently active state IDs.
        current_state: ID of the primary current state.
        state_history: Chronological list of visited state IDs.
        action_executor: Executor for running individual actions.

    Example:
        >>> config = ConfigParser().parse_file("automation.json")
        >>> executor = StateExecutor(config)
        >>> executor.initialize()
        >>> success = executor.execute()

    Note:
        States are verified by checking if their identifying images are visible
        on screen. This requires actual GUI environment (not headless).

    See Also:
        - :class:`DelegatingActionExecutor`: Executes individual actions within workflows
        - :class:`Workflow`: Sequence of actions in a transition
        - :class:`OutgoingTransition`: Transition from a state
        - :class:`IncomingTransition`: Verification when entering a state
    """

    def __init__(self, config: QontinuiConfig) -> None:
        self.config = config
        self.active_states: set[str] = set()
        self.current_state: str | None = None
        self.state_history: list[str] = []
        # Pass self to action executor so it can change states
        self.action_executor = DelegatingActionExecutor(config, state_executor=self)

    def initialize(self):
        """Initialize the state machine."""
        # Find all initial states
        initial_states = []
        for state in self.config.states:
            if state.is_initial:
                if not self.current_state:
                    self.current_state = state.id  # Set first as current
                self.active_states.add(state.id)
                initial_states.append(state.name)

        if initial_states:
            if len(initial_states) == 1:
                logger.info(f"Initial state: {initial_states[0]}")
            else:
                logger.info(f"Initial states: {', '.join(initial_states)}")
        elif self.config.states:
            # Use first state as initial if none marked
            self.current_state = self.config.states[0].id
            self.active_states.add(self.current_state)
            logger.info(f"Using first state as initial: {self.config.states[0].name}")

    def execute(self):
        """Execute the state machine automation workflow.

        Runs the complete state machine from initial state to final state (or until
        no transitions are available). The execution loop continuously verifies the
        current state, finds applicable transitions, executes them, and updates the
        active states.

        The execution stops when:
            - No applicable transitions are found
            - Maximum iterations reached (prevents infinite loops)
            - Failure strategy indicates stop (execution_settings.failure_strategy)

        Returns:
            bool: True if execution completed normally, False if errors occurred.

        Example:
            >>> executor = StateExecutor(config)
            >>> success = executor.execute()
            >>> if success:
            ...     print(f"Final states: {executor.get_active_states()}")
            ...     print(f"History: {executor.get_state_history()}")

        Note:
            - Max iterations is set to 1000 to prevent infinite loops
            - States are verified by checking if identifying images are visible
            - Waits 1 second between iterations if no transitions found

        See Also:
            - :meth:`initialize`: Sets up initial state
            - :meth:`_execute_transitions`: Finds and executes transitions
        """
        self.initialize()

        if not self.current_state:
            logger.error("No initial state found")
            return False

        max_iterations = 1000  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Verify current state is active
            if not self._verify_state(self.current_state):
                logger.warning(f"State {self.current_state} is not active")
                # Try to find active state
                if not self._find_active_state():
                    logger.error("No active state found")
                    break

            # Find and execute applicable transitions
            transition_executed = self._execute_transitions()

            if not transition_executed:
                logger.info("No applicable transitions found")
                # Check if we should wait or exit
                if self._should_continue():
                    TimeWrapper.wait(1)
                else:
                    break

        return True

    def _verify_state(self, state_id: str) -> bool:
        """Verify if a state is currently active using async parallel FindAction.

        Uses async image finding to check all required images in parallel
        for improved performance.

        Args:
            state_id: State identifier

        Returns:
            True if all required state images are found
        """
        import asyncio

        from ..actions.find import FindAction
        from ..model.element import Pattern

        state = self.config.state_map.get(state_id)
        if not state:
            return False

        if not state.identifying_images:
            # State has no identifying images, consider it active
            return True

        # Collect all required patterns
        patterns = []
        for state_image in state.identifying_images:
            if state_image.required:
                image = self.config.image_map.get(state_image.id)
                if image and image.file_path:
                    # Create Pattern from file with similarity from state_image
                    pattern = Pattern.from_file(
                        img_path=str(image.file_path),
                        name=image.name or state_image.id,
                    )
                    # Set similarity threshold - will be used by cascade
                    if state_image.threshold:
                        pattern = pattern.with_similarity(state_image.threshold)
                    patterns.append(pattern)

        if not patterns:
            return True

        # Check all patterns in parallel using async finding with cascade
        action = FindAction()

        async def verify_async():
            from ..actions.find.find_options_builder import CascadeContext, build_find_options

            # Build options with proper cascade
            # Patterns already have similarity set via with_similarity(), cascade will use that
            try:
                from ..config.settings import QontinuiSettings

                project_config = QontinuiSettings()
            except Exception:
                project_config = None

            ctx = CascadeContext(
                search_options=None,
                pattern=None,  # Not pattern-specific
                state_image=None,
                project_config=project_config,
            )
            options = build_find_options(ctx)

            # Check all patterns concurrently
            find_results = await action.find_async(patterns, options)

            # All required patterns must be found
            return all(result.found for result in find_results)

        # Execute async verification
        return asyncio.run(verify_async())

    def _find_active_state(self) -> bool:
        """Find which state is currently active."""
        for state_id in self.active_states:
            if self._verify_state(state_id):
                self.current_state = state_id
                logger.info(f"Found active state: {self.config.state_map[state_id].name}")
                return True

        # Check all states if none of the active ones match
        for state in self.config.states:
            if self._verify_state(state.id):
                self.current_state = state.id
                self.active_states = {state.id}
                logger.info(f"Found state: {state.name}")
                return True

        return False

    def _execute_transitions(self) -> bool:
        """Execute applicable transitions from current state."""
        if not self.current_state:
            return False

        # Find OutgoingTransitions from current state
        outgoing_transitions = self._find_outgoing_transitions(self.current_state)

        for transition in outgoing_transitions:
            if self._execute_transition(transition):
                return True

        return False

    def _find_outgoing_transitions(self, state_id: str) -> list[OutgoingTransition]:
        """Find all OutgoingTransitions from a given state."""
        state = self.config.state_map.get(state_id)
        if state:
            return state.outgoing_transitions
        return []

    def _find_incoming_transitions(self, state_id: str) -> list[IncomingTransition]:
        """Find all IncomingTransitions to a given state."""
        state = self.config.state_map.get(state_id)
        if state:
            return state.incoming_transitions
        return []

    def _execute_transition(self, transition: Transition) -> bool:
        """Execute a single transition.

        Executes all workflows defined in the transition's workflows list.
        If any workflow fails, the entire transition fails.
        """
        logger.info(f"Executing transition: {transition.id}")

        # Execute all workflows in the transition
        for workflow_id in transition.workflows:
            workflow = self.config.workflow_map.get(workflow_id)
            if workflow:
                workflow_result = self._execute_workflow(workflow)
                logger.debug(f"Workflow '{workflow.name}' execution result: {workflow_result}")
                if not workflow_result:
                    logger.error(f"Workflow {workflow.name} failed")
                    return False
            else:
                # Workflow not found - fail fast with clear error
                logger.error(f"Workflow {workflow_id} not found in workflow_map")
                logger.error(f"Available workflows: {list(self.config.workflow_map.keys())}")
                logger.error("This workflow was not loaded during configuration parsing.")
                logger.error(
                    "If this is an inline workflow, it may have invalid format - please re-export your configuration."
                )
                return False

        # Handle state changes for OutgoingTransition
        if isinstance(transition, OutgoingTransition):
            # Collect all states to activate (to_state + activate_states)
            states_to_activate = set(transition.activate_states)
            if transition.to_state:
                states_to_activate.add(transition.to_state)

            # Deactivate states in deactivate_states
            for state_id in transition.deactivate_states:
                self.active_states.discard(state_id)
                state_obj: Any = self.config.state_map.get(state_id, {})
                if isinstance(state_obj, dict):
                    logger.info(f"Deactivated state: {state_id}")
                else:
                    logger.info(f"Deactivated state: {state_obj.name}")

            # Activate target states with IncomingTransition verification
            for state_id in states_to_activate:
                # Check if state has IncomingTransitions
                incoming_transitions = self._find_incoming_transitions(state_id)

                activation_allowed = True
                if incoming_transitions:
                    # Execute IncomingTransitions - if any fail, don't activate this state
                    for incoming_trans in incoming_transitions:
                        if not self._execute_transition(incoming_trans):
                            logger.warning(
                                f"IncomingTransition failed for state {state_id}, not activating"
                            )
                            activation_allowed = False
                            break

                # Only activate if IncomingTransitions succeeded (or none exist)
                if activation_allowed:
                    self.active_states.add(state_id)
                    state_obj = self.config.state_map.get(state_id, {})
                    if isinstance(state_obj, dict):
                        logger.info(f"Activated state: {state_id}")
                    else:
                        logger.info(f"Activated state: {state_obj.name}")

                    # Update current_state if this is the to_state
                    if state_id == transition.to_state:
                        self.current_state = transition.to_state
                        self.state_history.append(transition.to_state)
                        target_state = self.config.state_map.get(transition.to_state)
                        if target_state:
                            logger.info(f"Transitioned to state: {target_state.name}")

            # Handle origin state: deactivate by DEFAULT unless stays_visible=True
            if transition.from_state and not transition.stays_visible:
                self.active_states.discard(transition.from_state)
                from_state = self.config.state_map.get(transition.from_state)
                if from_state:
                    logger.info(f"Deactivated origin state: {from_state.name}")

        return True

    def _execute_workflow(self, workflow: Workflow) -> bool:
        """Execute a workflow (sequence of actions).

        Args:
            workflow: Workflow instance containing actions to execute.

        Returns:
            bool: True if workflow completed successfully, False otherwise
        """
        logger.info(f"Executing workflow: {workflow.name}")

        # All workflows now use graph format, execute actions sequentially
        for i, action in enumerate(workflow.actions):
            action_result = self.action_executor.execute_action(action)
            logger.debug(
                f"Action {i+1}/{len(workflow.actions)} ({action.type}) result: {action_result}"
            )
            if not action_result:
                if action.continue_on_error:
                    logger.debug(f"Action {i+1} failed but continue_on_error=True, continuing...")
                    continue
                logger.error(f"Workflow '{workflow.name}' FAILED at action {i+1}")
                return False

        logger.debug(f"Workflow '{workflow.name}' COMPLETED successfully")
        return True

    def _should_continue(self) -> bool:
        """Determine if the state machine should continue executing."""
        # Check failure strategy
        if self.config.execution_settings.failure_strategy == "stop":
            return False
        elif self.config.execution_settings.failure_strategy == "retry":
            return len(self.state_history) < 100  # Prevent infinite loops
        else:
            return True

    def get_active_states(self) -> list[str]:
        """Get list of currently active state names."""
        return [
            self.config.state_map[sid].name
            for sid in self.active_states
            if sid in self.config.state_map
        ]

    def get_state_history(self) -> list[str]:
        """Get history of visited states."""
        return [
            self.config.state_map[sid].name
            for sid in self.state_history
            if sid in self.config.state_map
        ]
