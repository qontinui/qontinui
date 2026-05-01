"""Enhanced StateTransitionsJointTable with multi-state activation support.

Central registry for managing state transitions with support for:
- Multi-state activation groups
- Bidirectional transition mapping
- State group management
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .enhanced_state_transition import StateTransition


@dataclass
class StateTransitionsJointTable:
    """Central registry with multi-state support.

    This enhanced version combines:
    - Brobot's bidirectional transition mapping
    - Multi-state activation tracking
    - State group management for logical organization
    - Efficient transition lookup
    """

    # Bidirectional mapping (from Brobot)
    from_transitions: dict[int, list[StateTransition]] = field(
        default_factory=lambda: defaultdict(list)
    )
    to_transitions: dict[int, list[StateTransition]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Multi-state activation groups
    activation_groups: dict[str, set[int]] = field(default_factory=dict)

    # Reverse mapping for group lookup
    state_to_groups: dict[int, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )

    # Initial states by profile
    initial_states: dict[str, set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )

    # All registered states
    all_states: set[int] = field(default_factory=set)

    def add_transition(self, transition: StateTransition, from_state: int) -> None:
        """Add a transition to the joint table.

        Args:
            transition: The transition to add
            from_state: The source state ID
        """
        # Add to from_transitions
        self.from_transitions[from_state].append(transition)

        # Add to to_transitions for all activated states
        for state_id in transition.activate:
            self.to_transitions[state_id].append(transition)
            self.all_states.add(state_id)

        # Track the from state as well
        self.all_states.add(from_state)

        # Track exited states
        for state_id in transition.exit:
            self.all_states.add(state_id)

    def get_transitions_from(self, state_id: int) -> list[StateTransition]:
        """Get all transitions from a specific state.

        Args:
            state_id: The source state ID

        Returns:
            List of transitions from the state
        """
        return self.from_transitions.get(state_id, [])

    def get_transitions_to(self, state_id: int) -> list[StateTransition]:
        """Get all transitions to a specific state.

        Args:
            state_id: The target state ID

        Returns:
            List of transitions to the state
        """
        return self.to_transitions.get(state_id, [])

    def get_transitions_to_activate(
        self, target_states: set[int]
    ) -> list[StateTransition]:
        """Find transitions that activate ALL target states.

        This is crucial for multi-state activation - we need transitions
        that can activate all required states together.

        Args:
            target_states: Set of state IDs that must all be activated

        Returns:
            List of transitions that activate all target states
        """
        if not target_states:
            return []

        # Find transitions that activate all target states
        valid_transitions = []

        # Check all transitions
        for transitions in self.from_transitions.values():
            for transition in transitions:
                if transition.activates_all(target_states):
                    valid_transitions.append(transition)

        return valid_transitions

    def get_transitions_to_any(self, target_states: set[int]) -> list[StateTransition]:
        """Find transitions that activate ANY of the target states.

        Args:
            target_states: Set of potential target state IDs

        Returns:
            List of transitions that activate any target state
        """
        transitions = []
        seen = set()  # Avoid duplicates

        for state_id in target_states:
            for transition in self.to_transitions.get(state_id, []):
                trans_id = id(transition)
                if trans_id not in seen:
                    seen.add(trans_id)
                    transitions.append(transition)

        return transitions

    def get_incoming_transitions(
        self, states: set[int]
    ) -> dict[int, list[StateTransition]]:
        """Get all incoming transitions for a set of states.

        This is used after activation to execute incoming transitions
        for all newly activated states.

        Args:
            states: Set of state IDs

        Returns:
            Dictionary mapping state ID to its incoming transitions
        """
        incoming = {}
        for state_id in states:
            transitions = self.to_transitions.get(state_id, [])
            if transitions:
                incoming[state_id] = transitions
        return incoming

    def define_group(self, group_name: str, state_ids: set[int]) -> None:
        """Define a named group of states.

        Groups allow logical organization of related states that
        often activate together (e.g., toolbar + sidebar + content).

        Args:
            group_name: Name of the group
            state_ids: Set of state IDs in the group
        """
        self.activation_groups[group_name] = state_ids.copy()

        # Update reverse mapping
        for state_id in state_ids:
            self.state_to_groups[state_id].add(group_name)
            self.all_states.add(state_id)

    def get_group(self, group_name: str) -> set[int]:
        """Get states in a named group.

        Args:
            group_name: Name of the group

        Returns:
            Set of state IDs in the group
        """
        return self.activation_groups.get(group_name, set()).copy()

    def get_groups_for_state(self, state_id: int) -> set[str]:
        """Get all groups that contain a state.

        Args:
            state_id: State ID to look up

        Returns:
            Set of group names containing the state
        """
        return self.state_to_groups.get(state_id, set()).copy()

    def add_initial_state(self, state_id: int, profile: str = "default") -> None:
        """Add an initial state for a specific profile.

        Args:
            state_id: State ID to mark as initial
            profile: Profile name (default: "default")
        """
        self.initial_states[profile].add(state_id)
        self.all_states.add(state_id)

    def get_initial_states(self, profile: str = "default") -> set[int]:
        """Get initial states for a profile.

        Args:
            profile: Profile name (default: "default")

        Returns:
            Set of initial state IDs
        """
        return self.initial_states.get(profile, set()).copy()

    def find_transition_between(
        self, from_state: int, to_states: set[int]
    ) -> StateTransition | None:
        """Find a transition from a state that activates target states.

        Args:
            from_state: Source state ID
            to_states: Target state IDs to activate

        Returns:
            First matching transition or None
        """
        transitions = self.from_transitions.get(from_state, [])

        for transition in transitions:
            if transition.activates_all(to_states):
                return transition

        return None

    def get_states_with_transitions_to(self, target_state: int) -> set[int]:
        """Get all states that have transitions to a target state.

        This is used in pathfinding to find parent states.

        Args:
            target_state: Target state ID

        Returns:
            Set of state IDs with transitions to target
        """
        parent_states = set()

        for from_state, transitions in self.from_transitions.items():
            for transition in transitions:
                if target_state in transition.activate:
                    parent_states.add(from_state)
                    break

        return parent_states

    def get_all_transitions(self) -> list[StateTransition]:
        """Get all registered transitions.

        Returns:
            List of all transitions
        """
        all_transitions = []
        seen = set()

        for transitions in self.from_transitions.values():
            for transition in transitions:
                trans_id = id(transition)
                if trans_id not in seen:
                    seen.add(trans_id)
                    all_transitions.append(transition)

        return all_transitions

    def clear(self) -> None:
        """Clear all transition data."""
        self.from_transitions.clear()
        self.to_transitions.clear()
        self.activation_groups.clear()
        self.state_to_groups.clear()
        self.initial_states.clear()
        self.all_states.clear()

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the joint table.

        Returns:
            Dictionary with statistics
        """
        total_transitions = len(self.get_all_transitions())

        return {
            "total_states": len(self.all_states),
            "total_transitions": total_transitions,
            "total_groups": len(self.activation_groups),
            "states_with_outgoing": len(self.from_transitions),
            "states_with_incoming": len(self.to_transitions),
            "profiles_with_initial": len(self.initial_states),
        }

    def __str__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"StateTransitionsJointTable("
            f"states={stats['total_states']}, "
            f"transitions={stats['total_transitions']}, "
            f"groups={stats['total_groups']})"
        )
