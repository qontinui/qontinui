"""State machine executor for Qontinui automation."""

from typing import Any

from ..wrappers import Time
from .action_executor import ActionExecutor
from .config_parser import (
    IncomingTransition,
    OutgoingTransition,
    Process,
    QontinuiConfig,
    Transition,
)


class StateExecutor:
    """Executes state machine based automation."""

    def __init__(self, config: QontinuiConfig):
        self.config = config
        self.active_states: set[str] = set()
        self.current_state: str | None = None
        self.state_history: list[str] = []
        # Pass self to action executor so it can change states
        self.action_executor = ActionExecutor(config, state_executor=self)

    def initialize(self):
        """Initialize the state machine."""
        # Find initial state
        for state in self.config.states:
            if state.is_initial:
                self.current_state = state.id
                self.active_states.add(state.id)
                print(f"Initial state: {state.name}")
                break

        if not self.current_state and self.config.states:
            # Use first state as initial if none marked
            self.current_state = self.config.states[0].id
            self.active_states.add(self.current_state)
            print(f"Using first state as initial: {self.config.states[0].name}")

    def execute(self):
        """Execute the state machine."""
        self.initialize()

        if not self.current_state:
            print("No initial state found")
            return False

        max_iterations = 1000  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Check if we're in a final state
            current = self.config.state_map.get(self.current_state)
            if current and current.is_final:
                print(f"Reached final state: {current.name}")
                break

            # Verify current state is active
            if not self._verify_state(self.current_state):
                print(f"State {self.current_state} is not active")
                # Try to find active state
                if not self._find_active_state():
                    print("No active state found")
                    break

            # Find and execute applicable transitions
            transition_executed = self._execute_transitions()

            if not transition_executed:
                print("No applicable transitions found")
                # Check if we should wait or exit
                if self._should_continue():
                    Time.wait(1)
                else:
                    break

        return True

    def _verify_state(self, state_id: str) -> bool:
        """Verify if a state is currently active by checking its identifying images."""
        state = self.config.state_map.get(state_id)
        if not state:
            return False

        if not state.identifying_images:
            # State has no identifying images, consider it active
            return True

        # Check all required images
        for state_image in state.identifying_images:
            if state_image.required:
                image = self.config.image_map.get(state_image.image_id)
                if image and image.file_path:
                    location = self.action_executor._find_image_on_screen(
                        image.file_path, state_image.threshold
                    )
                    if not location:
                        return False

        return True

    def _find_active_state(self) -> bool:
        """Find which state is currently active."""
        for state_id in self.active_states:
            if self._verify_state(state_id):
                self.current_state = state_id
                print(f"Found active state: {self.config.state_map[state_id].name}")
                return True

        # Check all states if none of the active ones match
        for state in self.config.states:
            if self._verify_state(state.id):
                self.current_state = state.id
                self.active_states = {state.id}
                print(f"Found state: {state.name}")
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
        transitions = []
        for trans in self.config.transitions:
            if isinstance(trans, OutgoingTransition) and trans.from_state == state_id:
                transitions.append(trans)
        return transitions

    def _find_incoming_transitions(self, state_id: str) -> list[IncomingTransition]:
        """Find all IncomingTransitions to a given state."""
        transitions = []
        for trans in self.config.transitions:
            if isinstance(trans, IncomingTransition) and trans.to_state == state_id:
                transitions.append(trans)
        return transitions

    def _execute_transition(self, transition: Transition) -> bool:
        """Execute a single transition."""
        print(f"\nExecuting transition: {transition.id}")

        # Execute process in the transition
        if transition.process:
            process = self.config.process_map.get(transition.process)
            if process:
                process_result = self._execute_process(process)
                print(f"[DEBUG] Process '{process.name}' execution result: {process_result}")
                if not process_result:
                    print(f"Process {process.name} failed")
                    return False
            else:
                print(f"Process {transition.process} not found")
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
                    print(f"Deactivated state: {state_id}")
                else:
                    print(f"Deactivated state: {state_obj.name}")

            # Activate target states with IncomingTransition verification
            for state_id in states_to_activate:
                # Check if state has IncomingTransitions
                incoming_transitions = self._find_incoming_transitions(state_id)

                activation_allowed = True
                if incoming_transitions:
                    # Execute IncomingTransitions - if any fail, don't activate this state
                    for incoming_trans in incoming_transitions:
                        if not self._execute_transition(incoming_trans):
                            print(f"IncomingTransition failed for state {state_id}, not activating")
                            activation_allowed = False
                            break

                # Only activate if IncomingTransitions succeeded (or none exist)
                if activation_allowed:
                    self.active_states.add(state_id)
                    state_obj = self.config.state_map.get(state_id, {})
                    if isinstance(state_obj, dict):
                        print(f"Activated state: {state_id}")
                    else:
                        print(f"Activated state: {state_obj.name}")

                    # Update current_state if this is the to_state
                    if state_id == transition.to_state:
                        self.current_state = transition.to_state
                        self.state_history.append(transition.to_state)
                        target_state = self.config.state_map.get(transition.to_state)
                        if target_state:
                            print(f"Transitioned to state: {target_state.name}")

            # Handle origin state: deactivate by DEFAULT unless stays_visible=True
            if transition.from_state and not transition.stays_visible:
                self.active_states.discard(transition.from_state)
                from_state = self.config.state_map.get(transition.from_state)
                if from_state:
                    print(f"Deactivated origin state: {from_state.name}")

        return True

    def _execute_process(self, process: Process) -> bool:
        """Execute a process (sequence of actions)."""
        print(f"Executing process: {process.name}")

        if process.type == "sequence":
            # Execute actions in sequence
            for i, action in enumerate(process.actions):
                action_result = self.action_executor.execute_action(action)
                print(f"[DEBUG] Action {i+1}/{len(process.actions)} ({action.type}) result: {action_result}")
                if not action_result:
                    if action.continue_on_error:
                        print(f"[DEBUG] Action {i+1} failed but continue_on_error=True, continuing...")
                        continue
                    print(f"[DEBUG] Process '{process.name}' FAILED at action {i+1}")
                    return False
        elif process.type == "parallel":
            # For now, execute sequentially (parallel execution would need threading)
            for action in process.actions:
                self.action_executor.execute_action(action)

        print(f"[DEBUG] Process '{process.name}' COMPLETED successfully")
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
