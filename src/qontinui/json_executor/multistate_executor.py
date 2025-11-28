"""Multistate-based executor for Qontinui JSON configurations.

Replaces state_executor.py with proper multistate integration.
"""

import logging
import time
from typing import Any

from ..model.state.state import State
from ..model.transition.enhanced_state_transition import (
    StaysVisible,
    TaskSequenceStateTransition,
)
from ..multistate_integration.multistate_adapter import MultiStateAdapter
from .action_executor import ActionExecutor
from .config_parser import QontinuiConfig

logger = logging.getLogger(__name__)


class MultiStateExecutor:
    """Executes state machine using MultiState framework."""

    def __init__(self, config: QontinuiConfig):
        self.config = config
        self.adapter = MultiStateAdapter()
        # Track current and active states by name (string)
        self.current_state_name: str | None = None
        self.active_state_names: set[str] = set()
        self.state_history: list[str] = []

        # Create ActionExecutor with reference to this executor
        self.action_executor = ActionExecutor(config, state_executor=self)

        # Maps to track multistate IDs
        # Key: state name (string), Value: multistate ID
        self.qontinui_to_multistate: dict[str, str] = {}
        self.multistate_to_qontinui: dict[str, str] = {}

        # Map state names to their int hash IDs (used by transitions)
        self.state_name_to_hash: dict[str, int] = {}
        self.state_hash_to_name: dict[int, str] = {}

        # Track transitions by from_state for quick lookup
        self.transitions_from_state: dict[str, list[TaskSequenceStateTransition]] = {}
        # Track all transitions
        self.all_transitions: list[TaskSequenceStateTransition] = []

        self._register_states_and_transitions()

    def _register_states_and_transitions(self):
        """Register all states and transitions with multistate."""
        print("Initializing state machine with MultiState integration...")

        # Register all states with MultiState
        for state in self.config.states:
            # Determine if state is blocking (modal dialogs, etc.)
            blocking = self._is_blocking_state(state)

            # Create MultiState representation with state verification callback
            multistate_id = f"state_{state.name.replace(' ', '_').lower()}"
            self.adapter.manager.add_state(
                id=multistate_id,
                name=state.name,
                blocking=blocking,
                group=self._get_state_group(state),
            )

            # Store mapping between Qontinui state name and MultiState ID
            self.qontinui_to_multistate[state.name] = multistate_id
            self.multistate_to_qontinui[multistate_id] = state.name

            # Store hash mapping (transitions use hash(name) as int ID)
            state_hash = hash(state.name)
            self.state_name_to_hash[state.name] = state_hash
            self.state_hash_to_name[state_hash] = state.name

            print(
                f"Registered state: {state.name} (hash: {state_hash}) -> MultiState: {multistate_id}"
            )

        # Register all transitions with MultiState
        for trans in self.config.transitions:
            # Store all transitions
            self.all_transitions.append(trans)

            # Build transition maps by from_state for quick lookup
            # from_states are int hashes, need to convert to names
            for from_state_hash in trans.from_states:
                from_state_name = self.state_hash_to_name.get(from_state_hash)
                if from_state_name:
                    if from_state_name not in self.transitions_from_state:
                        self.transitions_from_state[from_state_name] = []
                    self.transitions_from_state[from_state_name].append(trans)

            # Register with MultiState
            self._register_multistate_transition(trans)

        print(
            f"Registered {len(self.config.states)} states and {len(self.config.transitions)} transitions"
        )

    def _register_multistate_transition(self, transition: TaskSequenceStateTransition):
        """Register a TaskSequenceStateTransition with MultiState."""
        # Convert int hash IDs to state names, then to MultiState IDs
        from_states = []
        for state_hash in transition.from_states:
            state_name = self.state_hash_to_name.get(state_hash)
            if state_name:
                ms_id = self.qontinui_to_multistate.get(state_name)
                if ms_id:
                    from_state_obj = self.adapter.manager.get_state(ms_id)
                    if from_state_obj:
                        from_states.append(from_state_obj)

        activate_states = []
        for state_hash in transition.activate:
            state_name = self.state_hash_to_name.get(state_hash)
            if state_name:
                ms_id = self.qontinui_to_multistate.get(state_name)
                if ms_id:
                    activate_state_obj = self.adapter.manager.get_state(ms_id)
                    if activate_state_obj:
                        activate_states.append(activate_state_obj)

        exit_states = []
        # Handle stays_visible semantics
        if transition.stays_visible == StaysVisible.FALSE:
            # Add from_states to exit_states
            for state_hash in transition.from_states:
                state_name = self.state_hash_to_name.get(state_hash)
                if state_name:
                    ms_id = self.qontinui_to_multistate.get(state_name)
                    if ms_id:
                        exit_state_obj = self.adapter.manager.get_state(ms_id)
                        if exit_state_obj:
                            exit_states.append(exit_state_obj)

        # Add explicit exit states
        for state_hash in transition.exit:
            state_name = self.state_hash_to_name.get(state_hash)
            if state_name:
                ms_id = self.qontinui_to_multistate.get(state_name)
                if ms_id:
                    exit_state_obj = self.adapter.manager.get_state(ms_id)
                    if exit_state_obj:
                        exit_states.append(exit_state_obj)

        # Register transition with MultiState
        if from_states or activate_states:  # Only register if we have valid states
            self.adapter.manager.add_transition(
                id=f"trans_{transition.id}",
                name=transition.name or f"Transition {transition.id}",
                from_states=from_states,
                activate_states=activate_states,
                exit_states=exit_states,
                path_cost=1.0,  # Default cost, could be customized
            )
            print(f"  Registered transition: {transition.name or transition.id}")

    def _is_blocking_state(self, state: State) -> bool:
        """Determine if a state is blocking (modal)."""
        modal_indicators = ["modal", "dialog", "popup", "alert", "confirm"]
        state_name_lower = state.name.lower()

        for indicator in modal_indicators:
            if indicator in state_name_lower:
                return True

        return False

    def _get_state_group(self, state: State) -> str | None:
        """Get the group for a state."""
        if "toolbar" in state.name.lower():
            return "workspace_ui"
        elif "sidebar" in state.name.lower():
            return "workspace_ui"
        elif "menu" in state.name.lower() and "main" not in state.name.lower():
            return "menus"
        elif "panel" in state.name.lower():
            return "panels"

        return None

    def initialize(self, start_state: str | None = None):
        """Initialize the state machine.

        Args:
            start_state: Name of the state to start from (optional)
        """
        if start_state:
            # Use provided start state (by name)
            self.current_state_name = start_state
            self.active_state_names.add(start_state)
            state = self.config.state_map.get(start_state)
            if state:
                print(f"Starting from state: {state.name}")
            else:
                print(f"Warning: Start state {start_state} not found")
        else:
            # Use first state as initial (Brobot doesn't have is_initial flag)
            if self.config.states:
                first_state = self.config.states[0]
                self.current_state_name = first_state.name
                self.active_state_names.add(first_state.name)
                print(f"Using first state as initial: {first_state.name}")

    def execute(self) -> bool:
        """Execute the state machine."""
        if not self.current_state_name:
            print("No initial state found")
            return False

        max_iterations = 1000  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Verify current state is active
            if not self._verify_state(self.current_state_name):
                print(f"State {self.current_state_name} is not active")
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

    def _verify_state(self, state_name: str) -> bool:
        """Verify if a state is currently active by checking its identifying images.

        Args:
            state_name: Name of the state to verify

        Returns:
            True if state's identifying images are found on screen
        """
        state = self.config.state_map.get(state_name)
        if not state:
            return False

        # Brobot StateImage objects have Image objects, not image IDs
        if not state.state_images:
            # State has no identifying images, consider it active
            return True

        # Check all state images (Brobot uses state_images list)
        for state_image in state.state_images:
            # StateImage has an Image object with a file path
            if state_image.image and hasattr(state_image.image, "path"):
                similarity = state_image.get_similarity()
                location = self.action_executor._find_image_on_screen(
                    state_image.image.path, similarity
                )
                if not location:
                    return False

        return True

    def _find_active_state(self) -> bool:
        """Find which state is currently active."""
        for state_name in self.active_state_names:
            if self._verify_state(state_name):
                self.current_state_name = state_name
                print(f"Found active state: {state_name}")
                return True

        # Check all states if none of the active ones match
        for state in self.config.states:
            if self._verify_state(state.name):
                self.current_state_name = state.name
                self.active_state_names = {state.name}
                print(f"Found state: {state.name}")
                return True

        return False

    def _execute_transitions(self) -> bool:
        """Execute applicable transitions from current state."""
        if not self.current_state_name:
            return False

        # Find transitions from current state
        transitions = self.transitions_from_state.get(self.current_state_name, [])

        for transition in transitions:
            if self._execute_transition(transition):
                return True

        return False

    def _execute_transition(self, transition: TaskSequenceStateTransition) -> bool:
        """Execute a transition with proper multistate semantics.

        Args:
            transition: TaskSequenceStateTransition to execute

        Returns:
            True if transition executed successfully
        """
        print(f"\nExecuting transition: {transition.name or transition.id}")

        # Execute task sequence (if present)
        if transition.task_sequence:
            # TODO: Execute TaskSequence when Process → TaskSequence conversion is implemented
            print("Task sequence execution not yet implemented")

        # Apply state changes using multistate semantics

        # 1. Handle exit states
        for state_hash in transition.exit:
            state_name = self.state_hash_to_name.get(state_hash)
            if state_name and state_name in self.active_state_names:
                self.active_state_names.discard(state_name)
                print(f"Exited state: {state_name}")

        # 2. Handle stays_visible - if FALSE, exit from_states
        if transition.stays_visible == StaysVisible.FALSE:
            for state_hash in transition.from_states:
                state_name = self.state_hash_to_name.get(state_hash)
                if state_name and state_name in self.active_state_names:
                    self.active_state_names.discard(state_name)
                    print(f"Exited origin state: {state_name}")

        # Sync with MultiState after exits
        self._sync_multistate_active_states()

        # 3. Activate states (with verification)
        for state_hash in transition.activate:
            state_name = self.state_hash_to_name.get(state_hash)
            if state_name:
                # Verify the state is actually visible before activating
                if self._verify_state(state_name):
                    self.active_state_names.add(state_name)
                    self.current_state_name = state_name  # Update current state
                    self.state_history.append(state_name)
                    print(f"Activated state: {state_name}")
                else:
                    print(f"Cannot activate state '{state_name}': identifying images not found")
                    return False

        # Sync with MultiState after activations
        self._sync_multistate_active_states()

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
        return list(self.active_state_names)

    def get_state_history(self) -> list[str]:
        """Get history of visited states."""
        return self.state_history.copy()

    def _find_outgoing_transitions(self, state_name: str) -> list[TaskSequenceStateTransition]:
        """Find all transitions from a given state.

        This method is called by action_executor for GO_TO_STATE action.

        Args:
            state_name: Name of the state

        Returns:
            List of transitions that can execute from this state
        """
        return self.transitions_from_state.get(state_name, [])

    def find_path_to_states(
        self, target_state_names: list[str], from_state_names: set[str] | None = None
    ) -> list[TaskSequenceStateTransition] | None:
        """Find optimal path to reach target states using MultiState pathfinding.

        Args:
            target_state_names: List of target state names to reach
            from_state_names: Starting state names (uses current active states if None)

        Returns:
            Sequence of TaskSequenceStateTransitions to execute, or None if no path exists
        """
        # Use current active states if no starting states provided
        if from_state_names is None:
            from_state_names = self.active_state_names

        if not from_state_names:
            logger.warning("No current states for pathfinding")
            return None

        # Convert Qontinui state names to MultiState IDs
        from_multistate_ids = set()
        for state_name in from_state_names:
            ms_id = self.qontinui_to_multistate.get(state_name)
            if ms_id:
                from_multistate_ids.add(ms_id)

        target_multistate_ids = []
        for state_name in target_state_names:
            ms_id = self.qontinui_to_multistate.get(state_name)
            if ms_id:
                target_multistate_ids.append(ms_id)

        if not target_multistate_ids:
            logger.warning(f"No valid target states for pathfinding: {target_state_names}")
            return None

        # Find path using MultiState
        path = self.adapter.manager.find_path_to(
            target_multistate_ids, from_states=from_multistate_ids
        )

        if not path:
            logger.info(f"No path found to states: {target_state_names}")
            return None

        # Convert MultiState transition sequence back to Qontinui TaskSequenceStateTransitions
        qontinui_transitions = []
        for multi_trans in path.transitions_sequence:
            # Extract transition ID (format: "trans_{hash_id}")
            if multi_trans.id.startswith("trans_"):
                trans_hash = int(multi_trans.id[6:])  # Remove "trans_" prefix and convert to int

                # Find the corresponding Qontinui transition by hash ID
                for trans in self.all_transitions:
                    if trans.id == trans_hash:
                        qontinui_transitions.append(trans)
                        break

        if len(qontinui_transitions) != len(path.transitions_sequence):
            logger.warning(
                f"Could not convert all MultiState transitions to Qontinui transitions "
                f"({len(qontinui_transitions)}/{len(path.transitions_sequence)})"
            )

        logger.info(f"Found path with {len(qontinui_transitions)} transitions")
        return qontinui_transitions if qontinui_transitions else None

    def is_state_active(self, state_id: str) -> bool:
        """Check if a state is currently active (callback for MultiState).

        This verifies the state by checking its identifying images on screen.

        Args:
            state_id: Qontinui state ID (or MultiState ID)

        Returns:
            True if state is active
        """
        # Handle both Qontinui and MultiState IDs
        if state_id.startswith("state_"):
            # This is a MultiState ID, convert to Qontinui ID
            qontinui_id = self.multistate_to_qontinui.get(state_id)
            if not qontinui_id:
                return False
            state_id = qontinui_id

        return self._verify_state(state_id)

    def _sync_multistate_active_states(self):
        """Sync Qontinui's active states with MultiState framework.

        This updates MultiState to reflect the current active states after
        a transition has been executed.
        """
        # Convert Qontinui state names to MultiState IDs
        multistate_active = set()
        for state_name in self.active_state_names:
            ms_id = self.qontinui_to_multistate.get(state_name)
            if ms_id:
                multistate_active.add(ms_id)

        # Update MultiState manager's active states
        self.adapter.manager.activate_states(multistate_active)
        logger.debug(f"Synced {len(multistate_active)} active states with MultiState")

    def get_multistate_statistics(self) -> dict[str, Any]:
        """Get statistics about the MultiState integration.

        Returns:
            Dictionary with system statistics including:
            - Total states registered
            - Total transitions registered
            - Active states
            - Reachable states from current position
            - State groups
        """
        complexity = self.adapter.manager.analyze_complexity()

        return {
            "total_states": len(self.config.states),
            "total_transitions": len(self.config.transitions),
            "multistate_states": complexity["num_states"],
            "multistate_transitions": complexity["num_transitions"],
            "active_state_count": len(self.active_state_names),
            "active_states": list(self.active_state_names),
            "reachable_states": complexity["reachable_states"],
            "groups": complexity["num_groups"],
            "state_history_length": len(self.state_history),
        }
