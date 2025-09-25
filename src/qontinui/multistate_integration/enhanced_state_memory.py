"""Enhanced StateMemory with MultiState features.

Extends Qontinui's StateMemory with:
- Dynamic transition tracking
- Occlusion detection
- Multi-target path planning
- Group state management
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from qontinui.model.transition.enhanced_state_transition import StateTransition
from qontinui.state_management.state_memory import StateMemory

from .multistate_adapter import MultiStateAdapter

if TYPE_CHECKING:
    from qontinui.model.state.state_service import StateService

logger = logging.getLogger(__name__)


@dataclass
class OcclusionInfo:
    """Information about state occlusion."""

    covering_state_id: int
    hidden_state_ids: set[int]
    occlusion_type: str  # 'modal', 'spatial', 'logical'
    confidence: float = 1.0


class EnhancedStateMemory(StateMemory):
    """StateMemory enhanced with MultiState framework features.

    This enhanced version adds:
    1. Automatic occlusion detection
    2. Dynamic transition generation
    3. Multi-target navigation planning
    4. State group management
    5. Temporal transition tracking
    """

    def __init__(self, state_service: Optional["StateService"] = None):
        """Initialize enhanced state memory.

        Args:
            state_service: Service for accessing state definitions
        """
        super().__init__(state_service)

        # Initialize MultiState adapter
        self.multistate_adapter = MultiStateAdapter(state_memory=self)

        # Track occlusions
        self.occlusions: dict[int, OcclusionInfo] = {}

        # Track dynamic transitions
        self.dynamic_transitions: list[StateTransition] = []

        # Track state groups
        self.state_groups: dict[str, set[int]] = {}

        # Multi-target navigation cache
        self.navigation_cache: dict[tuple[int, ...], list[StateTransition]] = {}

    def add_state(self, state_id: int, activate: bool = True) -> None:
        """Add a state to memory with MultiState registration.

        Args:
            state_id: ID of state to add
            activate: Whether to activate the state
        """
        # Get state from service
        if not self.state_service:
            logger.warning("No state service available")
            return

        state = self.state_service.get_state(state_id)
        if not state:
            logger.warning(f"State {state_id} not found")
            return

        # Register with MultiState
        self.multistate_adapter.register_qontinui_state(state)

        # Add to active states if requested
        if activate:
            self.active_states.add(state_id)
            self._update_occlusions()
            self._check_dynamic_transitions()

    def remove_state(self, state_id: int) -> None:
        """Remove a state and handle reveal transitions.

        Args:
            state_id: ID of state to remove
        """
        if state_id not in self.active_states:
            return

        # Check if this state was occluding others
        if state_id in self.occlusions:
            occlusion_info = self.occlusions[state_id]

            # Generate reveal transition
            reveal_transition = self.multistate_adapter.generate_reveal_transition(
                covering_state_id=state_id, hidden_state_ids=occlusion_info.hidden_state_ids
            )

            if reveal_transition:
                self.dynamic_transitions.append(reveal_transition)
                logger.info(
                    f"Generated reveal transition for {len(occlusion_info.hidden_state_ids)} hidden states"
                )

            # Remove occlusion tracking
            del self.occlusions[state_id]

        # Remove from active states
        self.active_states.discard(state_id)
        self._update_occlusions()

    def find_path_to_states(
        self, target_state_ids: list[int], use_cache: bool = True
    ) -> list[StateTransition] | None:
        """Find optimal path to reach ALL target states.

        Uses MultiState's multi-target pathfinding algorithm.

        Args:
            target_state_ids: ALL states that must be reached
            use_cache: Whether to use cached paths

        Returns:
            Sequence of transitions to execute, or None if impossible
        """
        # Check cache
        cache_key = tuple(sorted(target_state_ids))
        if use_cache and cache_key in self.navigation_cache:
            logger.debug(f"Using cached path to {target_state_ids}")
            return self.navigation_cache[cache_key]

        # Find path using MultiState
        path = self.multistate_adapter.find_path_to_states(
            target_state_ids=target_state_ids, current_state_ids=self.active_states
        )

        # Cache result
        if path and use_cache:
            self.navigation_cache[cache_key] = path

        return path

    def activate_group(self, group_name: str) -> None:
        """Activate all states in a group atomically.

        Args:
            group_name: Name of group to activate
        """
        if group_name not in self.state_groups:
            logger.warning(f"Unknown group: {group_name}")
            return

        group_states = self.state_groups[group_name]
        logger.info(f"Activating group '{group_name}' with {len(group_states)} states")

        # Activate all states together
        self.active_states.update(group_states)

        # Update MultiState
        self.multistate_adapter.sync_with_state_memory()

        # Check for occlusions and dynamic transitions
        self._update_occlusions()
        self._check_dynamic_transitions()

    def deactivate_group(self, group_name: str) -> None:
        """Deactivate all states in a group atomically.

        Args:
            group_name: Name of group to deactivate
        """
        if group_name not in self.state_groups:
            logger.warning(f"Unknown group: {group_name}")
            return

        group_states = self.state_groups[group_name]
        logger.info(f"Deactivating group '{group_name}' with {len(group_states)} states")

        # Deactivate all states together
        self.active_states.difference_update(group_states)

        # Update MultiState
        self.multistate_adapter.sync_with_state_memory()

        # Check for reveal transitions
        for state_id in group_states:
            if state_id in self.occlusions:
                self.remove_state(state_id)  # Handles reveal transitions

    def register_state_group(self, group_name: str, state_ids: set[int]) -> None:
        """Register a group of states that activate/deactivate together.

        Args:
            group_name: Name for the group
            state_ids: States in the group
        """
        self.state_groups[group_name] = state_ids
        logger.info(f"Registered group '{group_name}' with {len(state_ids)} states")

    def get_occluded_states(self) -> set[int]:
        """Get all currently occluded state IDs.

        Returns:
            Set of state IDs that are hidden by other states
        """
        occluded = set()
        for occlusion_info in self.occlusions.values():
            occluded.update(occlusion_info.hidden_state_ids)
        return occluded

    def get_visible_states(self) -> set[int]:
        """Get currently visible (non-occluded) state IDs.

        Returns:
            Set of state IDs that are visible
        """
        occluded = self.get_occluded_states()
        return self.active_states - occluded

    def get_dynamic_transitions(self) -> list[StateTransition]:
        """Get currently available dynamic transitions.

        Returns:
            List of dynamically generated transitions
        """
        # Filter out expired transitions
        valid_transitions = []
        for trans in self.dynamic_transitions:
            # Check if transition is still valid
            # (Would check temporal validity here)
            valid_transitions.append(trans)

        return valid_transitions

    def _update_occlusions(self) -> None:
        """Update occlusion tracking based on current active states."""
        # Detect occlusions using MultiState
        occlusion_pairs = self.multistate_adapter.detect_occlusions(self.active_states)

        # Update occlusion tracking
        new_occlusions = {}
        for covering_id, occluded_id in occlusion_pairs:
            if covering_id not in new_occlusions:
                new_occlusions[covering_id] = OcclusionInfo(
                    covering_state_id=covering_id,
                    hidden_state_ids=set(),
                    occlusion_type="modal",  # Simplified - would detect type
                )
            new_occlusions[covering_id].hidden_state_ids.add(occluded_id)

        self.occlusions = new_occlusions

        if new_occlusions:
            logger.debug(f"Updated occlusions: {len(new_occlusions)} covering states")

    def _check_dynamic_transitions(self) -> None:
        """Check for and generate dynamic transitions based on current state."""
        # This would check for:
        # 1. Reveal transitions for occluded states
        # 2. Self-transitions for validation
        # 3. Discovered service transitions
        # 4. Temporal transitions that have become valid
        pass

    def get_statistics(self) -> dict:
        """Get enhanced statistics about state memory.

        Returns:
            Dictionary with memory statistics
        """
        base_stats = {
            "active_states": len(self.active_states),
            "visible_states": len(self.get_visible_states()),
            "occluded_states": len(self.get_occluded_states()),
            "dynamic_transitions": len(self.dynamic_transitions),
            "state_groups": len(self.state_groups),
            "navigation_cache_size": len(self.navigation_cache),
        }

        # Add MultiState statistics
        multistate_stats = self.multistate_adapter.get_statistics()
        base_stats.update(multistate_stats)

        return base_stats
