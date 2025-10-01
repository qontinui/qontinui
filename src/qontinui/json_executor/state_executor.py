"""State machine executor for Qontinui automation."""

import time
from typing import Any

from .action_executor import ActionExecutor
from .config_parser import FromTransition, Process, QontinuiConfig, ToTransition, Transition


class StateExecutor:
    """Executes state machine based automation."""

    def __init__(self, config: QontinuiConfig):
        self.config = config
        self.action_executor = ActionExecutor(config)
        self.active_states: set[str] = set()
        self.current_state: str | None = None
        self.state_history: list[str] = []

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
                    time.sleep(1)
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

        # Find FromTransitions from current state
        from_transitions = self._find_from_transitions(self.current_state)

        for transition in from_transitions:
            if self._execute_transition(transition):
                return True

        return False

    def _find_from_transitions(self, state_id: str) -> list[FromTransition]:
        """Find all FromTransitions from a given state."""
        transitions = []
        for trans in self.config.transitions:
            if isinstance(trans, FromTransition) and trans.from_state == state_id:
                transitions.append(trans)
        return transitions

    def _find_to_transitions(self, state_id: str) -> list[ToTransition]:
        """Find all ToTransitions to a given state."""
        transitions = []
        for trans in self.config.transitions:
            if isinstance(trans, ToTransition) and trans.to_state == state_id:
                transitions.append(trans)
        return transitions

    def _execute_transition(self, transition: Transition) -> bool:
        """Execute a single transition."""
        print(f"\nExecuting transition: {transition.id}")

        # Execute processes in the transition
        for process_id in transition.processes:
            process = self.config.process_map.get(process_id)
            if process:
                if not self._execute_process(process):
                    print(f"Process {process.name} failed")
                    return False
            else:
                print(f"Process {process_id} not found")

        # Handle state changes for FromTransition
        if isinstance(transition, FromTransition):
            # Deactivate states
            for state_id in transition.deactivate_states:
                self.active_states.discard(state_id)
                state_obj: Any = self.config.state_map.get(state_id, {})
                if isinstance(state_obj, dict):
                    print(f"Deactivated state: {state_id}")
                else:
                    print(f"Deactivated state: {state_obj.name}")

            # Activate states
            for state_id in transition.activate_states:
                self.active_states.add(state_id)
                state_obj = self.config.state_map.get(state_id, {})
                if isinstance(state_obj, dict):
                    print(f"Activated state: {state_id}")
                else:
                    print(f"Activated state: {state_obj.name}")

            # Move to target state
            if transition.to_state:
                self.current_state = transition.to_state
                self.active_states.add(transition.to_state)
                self.state_history.append(transition.to_state)

                target_state = self.config.state_map.get(transition.to_state)
                if target_state:
                    print(f"Transitioned to state: {target_state.name}")

                # Execute ToTransitions for the new state
                to_transitions = self._find_to_transitions(transition.to_state)
                for to_trans in to_transitions:
                    self._execute_transition(to_trans)

            # Handle stays_visible
            if not transition.stays_visible and transition.from_state:
                self.active_states.discard(transition.from_state)

        return True

    def _execute_process(self, process: Process) -> bool:
        """Execute a process (sequence of actions)."""
        print(f"Executing process: {process.name}")

        if process.type == "sequence":
            # Execute actions in sequence
            for action in process.actions:
                if not self.action_executor.execute_action(action):
                    return False
        elif process.type == "parallel":
            # For now, execute sequentially (parallel execution would need threading)
            for action in process.actions:
                self.action_executor.execute_action(action)

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
