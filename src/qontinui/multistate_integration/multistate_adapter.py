"""Adapter to integrate MultiState framework with Qontinui.

This adapter bridges MultiState's theoretical framework with Qontinui's practical
GUI automation implementation.
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, cast

# Add multistate to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../multistate/src"))

from multistate.core.state import State as MultiState  # noqa: E402
from multistate.dynamics.hidden_states import HiddenStateManager  # noqa: E402
from multistate.manager import StateManager, StateManagerConfig  # noqa: E402
from multistate.pathfinding.multi_target import SearchStrategy  # noqa: E402
from multistate.transitions.transition import Transition as MultiTransition  # noqa: E402

from qontinui.model.state.state import State as QontinuiState  # noqa: E402
from qontinui.model.transition.enhanced_state_transition import TaskSequenceStateTransition
from qontinui.model.transition.enhanced_state_transition import (
    TaskSequenceStateTransition as StateTransition,
)
from qontinui.state_management.state_memory import StateMemory  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class StateMapping:
    """Maps between Qontinui and MultiState representations."""

    qontinui_id: int
    multistate_id: str
    qontinui_state: QontinuiState
    multistate_state: MultiState


class MultiStateAdapter:
    """Adapts MultiState framework for use in Qontinui.

    This adapter provides:
    1. State mapping between Qontinui and MultiState
    2. Transition conversion and enhancement
    3. Dynamic transition generation
    4. Occlusion detection for GUI states
    5. Multi-target pathfinding
    """

    def __init__(self, state_memory: StateMemory | None = None) -> None:
        """Initialize the MultiState adapter.

        Args:
            state_memory: Qontinui's state memory to integrate with
        """
        # Initialize MultiState components
        config = StateManagerConfig(
            default_search_strategy=SearchStrategy.DIJKSTRA, log_transitions=True
        )
        self.manager = StateManager(config)
        self.hidden_manager = HiddenStateManager()

        # State mapping
        self.state_mappings: dict[int, StateMapping] = {}
        self.reverse_mappings: dict[str, StateMapping] = {}

        # Transition mapping: MultiState transition ID -> Qontinui transition
        self.transition_mappings: dict[str, StateTransition] = {}

        # Qontinui integration
        self.state_memory = state_memory

        # Track dynamic transitions
        self.dynamic_transitions: dict[str, MultiTransition] = {}

    def register_qontinui_state(self, qontinui_state: QontinuiState) -> MultiState:
        """Register a Qontinui state in MultiState framework.

        Args:
            qontinui_state: Qontinui state to register

        Returns:
            Corresponding MultiState object
        """
        # Generate MultiState ID from Qontinui state
        multistate_id = f"state_{qontinui_state.name.replace(' ', '_').lower()}"

        # Check if already registered
        if qontinui_state.id is not None and qontinui_state.id in self.state_mappings:
            return self.state_mappings[qontinui_state.id].multistate_state

        # Determine if state is blocking (modal dialogs, etc.)
        blocking = self._is_blocking_state(qontinui_state)

        # Create MultiState representation
        multistate = self.manager.add_state(
            id=multistate_id,
            name=qontinui_state.name,
            blocking=blocking,
            group=self._get_state_group(qontinui_state),
        )

        # Create mapping (only if qontinui_state has a valid ID)
        if qontinui_state.id is None:
            raise ValueError(f"Cannot register state '{qontinui_state.name}' without an ID")

        mapping = StateMapping(
            qontinui_id=qontinui_state.id,
            multistate_id=multistate_id,
            qontinui_state=qontinui_state,
            multistate_state=multistate,
        )

        self.state_mappings[qontinui_state.id] = mapping
        self.reverse_mappings[multistate_id] = mapping

        logger.info(
            f"Registered Qontinui state '{qontinui_state.name}' as MultiState '{multistate_id}'"
        )

        return multistate

    def register_qontinui_transition(self, transition: StateTransition) -> MultiTransition:
        """Register a Qontinui transition in MultiState framework.

        Args:
            transition: Qontinui transition to register

        Returns:
            Corresponding MultiTransition object
        """
        # Ensure all states are registered
        from_states = set()
        activate_states = set()
        exit_states = set()

        # Convert from states
        if transition.from_states:
            for state_id in transition.from_states:
                if state_id in self.state_mappings:
                    from_states.add(self.state_mappings[state_id].multistate_id)
                else:
                    logger.error(
                        f"Transition {transition.id}: from_state ID {state_id} not found in state_mappings. "
                        f"Available mappings: {list(self.state_mappings.keys())}"
                    )

        # Convert activate states (ALL will be activated together)
        for state_id in transition.activate:
            if state_id in self.state_mappings:
                activate_states.add(self.state_mappings[state_id].multistate_id)
            else:
                logger.error(
                    f"Transition {transition.id}: activate state ID {state_id} not found in state_mappings. "
                    f"Available mappings: {list(self.state_mappings.keys())}"
                )

        # Convert exit states
        for state_id in transition.exit:
            if state_id in self.state_mappings:
                exit_states.add(self.state_mappings[state_id].multistate_id)

        # Create MultiState transition
        multi_transition_id = f"trans_{transition.id}"
        multi_transition = self.manager.add_transition(
            id=multi_transition_id,
            name=transition.name or f"Transition {transition.id}",
            from_states=list(from_states),
            activate_states=list(activate_states),
            exit_states=list(exit_states),
            path_cost=transition.score,  # Use Brobot's score as path cost
        )

        # Store mapping for reverse lookup during pathfinding
        # Use the actual transition ID from the returned MultiState transition object
        # in case the manager modified it
        actual_multi_id = multi_transition.id
        self.transition_mappings[actual_multi_id] = transition

        logger.info(
            f"Registered transition mapping: Qontinui '{transition.id}' -> MultiState '{actual_multi_id}'"
        )

        return multi_transition

    def detect_occlusions(self, active_state_ids: set[int]) -> list[tuple[int, int]]:
        """Detect which Qontinui states are occluded by others.

        Args:
            active_state_ids: Currently active Qontinui state IDs

        Returns:
            List of (occluding_id, occluded_id) tuples
        """
        # Convert to MultiState representations
        active_multistates = set()
        for state_id in active_state_ids:
            if state_id in self.state_mappings:
                active_multistates.add(self.state_mappings[state_id].multistate_state)

        # Detect occlusions
        occlusions = self.hidden_manager.detect_occlusion(active_multistates)

        # Convert back to Qontinui IDs
        qontinui_occlusions = []
        for occ_relation in occlusions:
            covering_mapping = self.reverse_mappings.get(occ_relation.covering_state.id)
            hidden_mapping = self.reverse_mappings.get(occ_relation.hidden_state.id)

            if covering_mapping and hidden_mapping:
                qontinui_occlusions.append(
                    (covering_mapping.qontinui_id, hidden_mapping.qontinui_id)
                )

        return qontinui_occlusions

    def find_path_to_states(
        self, target_state_ids: list[int], current_state_ids: set[int] | None = None
    ) -> list[StateTransition] | None:
        """Find optimal path to reach ALL target states.

        This uses MultiState's multi-target pathfinding to find the optimal
        sequence of transitions to reach all specified states.

        Args:
            target_state_ids: ALL states that must be reached
            current_state_ids: Starting states (uses state_memory if not provided)

        Returns:
            Sequence of Qontinui transitions to execute, or None if impossible
        """
        # Get current states
        if current_state_ids is None and self.state_memory:
            current_state_ids = self.state_memory.active_states

        if not current_state_ids:
            logger.warning("No current states for pathfinding")
            return None

        # Convert to MultiState
        current_multi = set()
        for state_id in current_state_ids:
            if state_id in self.state_mappings:
                current_multi.add(self.state_mappings[state_id].multistate_state)

        target_multi = set()
        for state_id in target_state_ids:
            if state_id in self.state_mappings:
                target_multi.add(self.state_mappings[state_id].multistate_state)

        if not target_multi:
            logger.warning("No valid target states for pathfinding")
            return None

        # Find path using MultiState
        target_ids = [ms.id for ms in target_multi]
        from_ids = {ms.id for ms in current_multi}

        path = self.manager.find_path_to(target_ids, from_states=from_ids)

        if not path:
            logger.info(f"No path found to states: {target_state_ids}")
            return None

        # Convert path to Qontinui transitions
        # MultiState returns transition objects - we need to extract IDs and look up Qontinui transitions
        logger.info(f"Found path with {len(path.transitions_sequence)} transitions")

        qontinui_transitions = []
        for multi_transition in path.transitions_sequence:
            # Extract the ID from the MultiState Transition object
            multi_transition_id = (
                multi_transition.id if hasattr(multi_transition, "id") else str(multi_transition)
            )

            if multi_transition_id in self.transition_mappings:
                qontinui_transitions.append(self.transition_mappings[multi_transition_id])
            else:
                logger.warning(
                    f"Transition '{multi_transition_id}' not found in mappings. Available: {list(self.transition_mappings.keys())}"
                )

        logger.info(f"Converted to {len(qontinui_transitions)} Qontinui transitions")
        return qontinui_transitions

    def generate_reveal_transition(
        self, covering_state_id: int, hidden_state_ids: set[int]
    ) -> TaskSequenceStateTransition | None:
        """Generate a reveal transition for hidden states.

        When a covering state (like a modal dialog) closes, this generates
        a transition to reveal the previously hidden states.

        Args:
            covering_state_id: State that is covering others
            hidden_state_ids: States that are hidden

        Returns:
            Dynamic StateTransition to reveal hidden states
        """
        if covering_state_id not in self.state_mappings:
            return None

        covering_multi = self.state_mappings[covering_state_id].multistate_state
        hidden_multi = set()

        for state_id in hidden_state_ids:
            if state_id in self.state_mappings:
                hidden_multi.add(self.state_mappings[state_id].multistate_state)

        if not hidden_multi:
            return None

        # Generate reveal transition in MultiState
        reveal_trans = self.hidden_manager.generate_reveal_transition(
            covering_state=covering_multi, hidden_states=hidden_multi
        )

        # Convert to Qontinui TaskSequenceStateTransition
        qontinui_trans = TaskSequenceStateTransition(
            name=f"Reveal hidden states under {covering_multi.name}",
            activate=hidden_state_ids,
            exit={covering_state_id},
            path_cost=1,  # Reveal transitions are low-cost
        )

        # Track dynamic transition
        self.dynamic_transitions[reveal_trans.id] = reveal_trans

        logger.info(f"Generated reveal transition: {qontinui_trans.name}")
        return qontinui_trans

    def _is_blocking_state(self, state: QontinuiState) -> bool:
        """Determine if a Qontinui state is blocking (modal).

        Args:
            state: Qontinui state to check

        Returns:
            True if state blocks others (like modal dialogs)
        """
        # Check for modal dialog indicators
        modal_indicators = ["modal", "dialog", "popup", "alert", "confirm"]
        state_name_lower = state.name.lower()

        for indicator in modal_indicators:
            if indicator in state_name_lower:
                return True

        # Check if state has blocking property set
        if hasattr(state, "blocking"):
            return cast(bool, state.blocking)

        return False

    def _get_state_group(self, state: QontinuiState) -> str | None:
        """Get the group for a Qontinui state.

        States in the same group activate/deactivate together.

        Args:
            state: Qontinui state

        Returns:
            Group name or None
        """
        # Check for UI component groups
        if "toolbar" in state.name.lower():
            return "workspace_ui"
        elif "sidebar" in state.name.lower():
            return "workspace_ui"
        elif "menu" in state.name.lower() and "main" not in state.name.lower():
            return "menus"
        elif "panel" in state.name.lower():
            return "panels"

        return None

    def sync_with_state_memory(self):
        """Sync MultiState's active states with Qontinui's StateMemory."""
        if not self.state_memory:
            return

        # Get Qontinui's active states
        qontinui_active = self.state_memory.active_states

        # Convert to MultiState IDs
        multistate_active = set()
        for state_id in qontinui_active:
            if state_id in self.state_mappings:
                multistate_active.add(self.state_mappings[state_id].multistate_id)

        # Update MultiState
        self.manager.activate_states(multistate_active)

        logger.debug(f"Synced {len(multistate_active)} active states with MultiState")

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the integrated system.

        Returns:
            Dictionary with system statistics
        """
        complexity = self.manager.analyze_complexity()

        return {
            "total_states": len(self.state_mappings),
            "multistate_states": complexity["num_states"],
            "multistate_transitions": complexity["num_transitions"],
            "dynamic_transitions": len(self.dynamic_transitions),
            "active_states": complexity["active_states"],
            "reachable_states": complexity["reachable_states"],
            "groups": complexity["num_groups"],
        }
