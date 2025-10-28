"""State Transitions Joint Table - Central registry for all state transitions.

This module is part of the Brobot-to-Qontinui migration and provides
a joint table that manages all state transitions in the automation framework.
It serves as the single source of truth for state machine configuration.
"""

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from .state_transition import StateTransition
from .state_transitions import StateTransitions

logger = logging.getLogger(__name__)


@dataclass
class TransitionEntry:
    """Entry in the joint table representing transitions for a state."""

    state_name: str
    transitions: list[StateTransition] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class StateTransitionsJointTable:
    """Central joint table for managing all state transitions.

    This class serves as the single source of truth for:
    - All registered states
    - All transitions between states
    - State machine configuration
    - Transition priorities and scoring

    Based on Brobot's StateTransitionsJointTable pattern, this provides
    thread-safe access to the state machine configuration.
    """

    def __init__(self) -> None:
        """Initialize the joint table."""
        # Main table: state_name -> TransitionEntry
        self._table: dict[str, TransitionEntry] = {}

        # Quick lookup: state_name -> set of reachable states
        self._reachability_map: dict[str, set[str]] = {}

        # Reverse lookup: state_name -> states that can reach it
        self._reverse_map: dict[str, set[str]] = {}

        # Thread safety
        self._lock = RLock()

        # Statistics
        self._total_transitions = 0
        self._last_update_time = None

        logger.info("StateTransitionsJointTable initialized")

    def add_to_joint_table(self, state_transitions: StateTransitions) -> None:
        """Add state transitions to the joint table.

        This is the primary method for registering transitions.
        It updates all internal maps and indices.

        Args:
            state_transitions: Container of transitions for a specific state
        """
        with self._lock:
            state_name = state_transitions.get_state_name()

            if not state_name:
                logger.warning("Cannot add transitions without state name")
                return

            # Get or create entry
            if state_name not in self._table:
                self._table[state_name] = TransitionEntry(state_name=state_name)
                self._reachability_map[state_name] = set()

            entry = self._table[state_name]
            transitions = state_transitions.get_transitions()

            # Add all transitions
            for transition in transitions:
                self._add_single_transition(state_name, transition, entry)

            # Update statistics
            self._total_transitions = sum(len(e.transitions) for e in self._table.values())

            import time

            self._last_update_time = time.time()

            logger.debug(f"Added {len(transitions)} transitions for state: {state_name}")
            logger.debug(f"Total transitions in joint table: {self._total_transitions}")

    def _add_single_transition(
        self, from_state: str, transition: StateTransition, entry: TransitionEntry
    ) -> None:
        """Add a single transition and update maps.

        Args:
            from_state: Source state name
            transition: The transition to add
            entry: The table entry to update
        """
        # Add to entry
        entry.transitions.append(transition)

        # Determine target states
        target_states = set()

        # Check various transition types for target states
        if hasattr(transition, "to_state") and transition.to_state:
            target_states.add(transition.to_state)

        if hasattr(transition, "activate_names"):
            target_states.update(transition.activate_names)

        if hasattr(transition, "target_state") and transition.target_state:
            target_states.add(transition.target_state)

        # Update reachability map
        self._reachability_map[from_state].update(target_states)

        # Update reverse map
        for target in target_states:
            if target not in self._reverse_map:
                self._reverse_map[target] = set()
            self._reverse_map[target].add(from_state)

    def get_transitions(self, state_name: str) -> StateTransitions | None:
        """Get all transitions for a specific state.

        Args:
            state_name: Name of the state

        Returns:
            StateTransitions container or None if state not found
        """
        with self._lock:
            if state_name not in self._table:
                return None

            entry = self._table[state_name]

            # Build StateTransitions container
            builder = StateTransitions.builder()
            builder.with_state_name(state_name)
            for transition in entry.transitions:
                builder.add_transition(transition)

            return builder.build()

    def get_all_transitions(self) -> dict[str, StateTransitions]:
        """Get all transitions in the joint table.

        Returns:
            Dictionary mapping state names to their transitions
        """
        with self._lock:
            result = {}

            for state_name in self._table:
                transitions = self.get_transitions(state_name)
                if transitions:
                    result[state_name] = transitions

            return result

    def has_state(self, state_name: str) -> bool:
        """Check if a state is registered in the joint table.

        Args:
            state_name: Name of the state

        Returns:
            True if state is registered
        """
        with self._lock:
            return state_name in self._table

    def get_registered_states(self) -> set[str]:
        """Get all registered state names.

        Returns:
            Set of all state names
        """
        with self._lock:
            # Include states that are sources or targets
            all_states = set(self._table.keys())
            all_states.update(self._reverse_map.keys())
            return all_states

    def get_reachable_states(self, from_state: str) -> set[str]:
        """Get all states reachable from a given state.

        Args:
            from_state: Source state name

        Returns:
            Set of reachable state names
        """
        with self._lock:
            return self._reachability_map.get(from_state, set()).copy()

    def get_states_that_reach(self, to_state: str) -> set[str]:
        """Get all states that can reach a given state.

        Args:
            to_state: Target state name

        Returns:
            Set of state names that can reach the target
        """
        with self._lock:
            return self._reverse_map.get(to_state, set()).copy()

    def find_transition(self, from_state: str, to_state: str) -> StateTransition | None:
        """Find a specific transition between two states.

        Args:
            from_state: Source state
            to_state: Target state

        Returns:
            The transition if found, None otherwise
        """
        with self._lock:
            if from_state not in self._table:
                return None

            entry = self._table[from_state]

            for transition in entry.transitions:
                # Check if this transition leads to the target
                if hasattr(transition, "to_state") and transition.to_state == to_state:
                    return transition

                if hasattr(transition, "activate_names") and to_state in transition.activate_names:
                    return transition

                if hasattr(transition, "target_state") and transition.target_state == to_state:
                    return transition

            return None

    def remove_state(self, state_name: str) -> bool:
        """Remove a state and all its transitions from the joint table.

        Args:
            state_name: Name of the state to remove

        Returns:
            True if state was removed, False if not found
        """
        with self._lock:
            if state_name not in self._table:
                return False

            # Remove from main table
            del self._table[state_name]

            # Remove from reachability map
            if state_name in self._reachability_map:
                # Update reverse map for states this one could reach
                for target in self._reachability_map[state_name]:
                    if target in self._reverse_map:
                        self._reverse_map[target].discard(state_name)
                        if not self._reverse_map[target]:
                            del self._reverse_map[target]

                del self._reachability_map[state_name]

            # Remove as a target from other states
            for source_state in list(self._reachability_map.keys()):
                self._reachability_map[source_state].discard(state_name)

            # Update statistics
            self._total_transitions = sum(len(e.transitions) for e in self._table.values())

            logger.info(f"Removed state from joint table: {state_name}")
            return True

    def clear(self) -> None:
        """Clear all entries from the joint table."""
        with self._lock:
            self._table.clear()
            self._reachability_map.clear()
            self._reverse_map.clear()
            self._total_transitions = 0
            self._last_update_time = None
            logger.info("Joint table cleared")

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the joint table.

        Returns:
            Dictionary with statistics
        """
        with self._lock:
            total_states = len(self.get_registered_states())
            total_source_states = len(self._table)
            total_target_states = len(self._reverse_map)

            # Calculate average transitions per state
            avg_transitions = 0
            if total_source_states > 0:
                avg_transitions = self._total_transitions / total_source_states

            # Find most connected states
            most_outgoing = None
            max_outgoing = 0
            for state, targets in self._reachability_map.items():
                if len(targets) > max_outgoing:
                    max_outgoing = len(targets)
                    most_outgoing = state

            most_incoming = None
            max_incoming = 0
            for state, sources in self._reverse_map.items():
                if len(sources) > max_incoming:
                    max_incoming = len(sources)
                    most_incoming = state

            return {
                "total_states": total_states,
                "total_source_states": total_source_states,
                "total_target_states": total_target_states,
                "total_transitions": self._total_transitions,
                "average_transitions_per_state": avg_transitions,
                "most_outgoing_connections": {"state": most_outgoing, "count": max_outgoing},
                "most_incoming_connections": {"state": most_incoming, "count": max_incoming},
                "last_update_time": self._last_update_time,
            }

    def validate(self) -> list[str]:
        """Validate the joint table for consistency.

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        with self._lock:
            # Check that all targets in reachability map exist as states
            all_targets = set()
            for targets in self._reachability_map.values():
                all_targets.update(targets)

            registered = self.get_registered_states()
            orphan_targets = all_targets - registered

            if orphan_targets:
                issues.append(f"Transitions to non-existent states: {orphan_targets}")

            # Check reverse map consistency
            for target, sources in self._reverse_map.items():
                for source in sources:
                    if source not in self._reachability_map:
                        issues.append(
                            f"Reverse map references non-existent source: {source} -> {target}"
                        )
                    elif target not in self._reachability_map.get(source, set()):
                        issues.append(f"Reverse map inconsistency: {source} -> {target}")

            # Check for isolated states (no incoming or outgoing)
            for state in registered:
                has_outgoing = state in self._reachability_map and self._reachability_map[state]
                has_incoming = state in self._reverse_map and self._reverse_map[state]

                if not has_outgoing and not has_incoming and state in self._table:
                    issues.append(f"Isolated state with no connections: {state}")

        return issues

    def __repr__(self) -> str:
        """String representation of the joint table."""
        with self._lock:
            stats = self.get_statistics()
            return (
                f"StateTransitionsJointTable("
                f"states={stats['total_states']}, "
                f"transitions={stats['total_transitions']})"
            )
