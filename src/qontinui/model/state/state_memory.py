"""StateMemory - ported from Qontinui framework.

Memory system for tracking state and transition history.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..transitions.state_transition import StateTransition
    from .state import State


@dataclass
class StateMemory:
    """Memory for state and transition history.

    Port of StateMemory from Qontinui framework class.
    Tracks the history of states visited and transitions executed.
    """

    # History storage
    _state_history: deque[StateRecord] = field(default_factory=lambda: deque(maxlen=1000))
    _transition_history: deque[TransitionRecord] = field(default_factory=lambda: deque(maxlen=1000))

    # Memory limits
    _max_state_history: int = 1000
    _max_transition_history: int = 1000

    # Statistics
    _total_states_recorded: int = 0
    _total_transitions_recorded: int = 0

    def record_state(self, state: State) -> StateMemory:
        """Record a state visit (fluent).

        Args:
            state: State that was visited

        Returns:
            Self for chaining
        """
        record = StateRecord(state=state, timestamp=time.time(), state_name=state.name)
        self._state_history.append(record)
        self._total_states_recorded += 1
        return self

    def record_transition(self, transition: StateTransition) -> StateMemory:
        """Record a transition execution (fluent).

        Args:
            transition: Transition that was executed

        Returns:
            Self for chaining
        """
        record = TransitionRecord(
            transition=transition,
            timestamp=time.time(),
            from_state=transition.from_state.name if transition.from_state else None,
            to_state=transition.to_state.name if transition.to_state else None,
            transition_type=transition.transition_type.value,
        )
        self._transition_history.append(record)
        self._total_transitions_recorded += 1
        return self

    def get_state_history(self, limit: int | None = None) -> list[State]:
        """Get state visit history.

        Args:
            limit: Maximum number of states to return

        Returns:
            List of states in visit order
        """
        records = list(self._state_history)
        if limit:
            records = records[-limit:]
        return [r.state for r in records if r.state]

    def get_transition_history(self, limit: int | None = None) -> list[StateTransition]:
        """Get transition execution history.

        Args:
            limit: Maximum number of transitions to return

        Returns:
            List of transitions in execution order
        """
        records = list(self._transition_history)
        if limit:
            records = records[-limit:]
        return [r.transition for r in records if r.transition]

    def get_last_state(self) -> State | None:
        """Get the last visited state.

        Returns:
            Last state or None
        """
        if self._state_history:
            return self._state_history[-1].state
        return None

    def get_last_transition(self) -> StateTransition | None:
        """Get the last executed transition.

        Returns:
            Last transition or None
        """
        if self._transition_history:
            return self._transition_history[-1].transition
        return None

    def get_state_count(self, state_name: str) -> int:
        """Get number of times a state was visited.

        Args:
            state_name: Name of state

        Returns:
            Visit count
        """
        return sum(1 for r in self._state_history if r.state_name == state_name)

    def get_transition_count(self, from_state: str, to_state: str) -> int:
        """Get number of times a transition was executed.

        Args:
            from_state: Source state name
            to_state: Target state name

        Returns:
            Execution count
        """
        return sum(
            1
            for r in self._transition_history
            if r.from_state == from_state and r.to_state == to_state
        )

    def get_recent_path(self, length: int = 5) -> list[State]:
        """Get recent path through states.

        Args:
            length: Number of states in path

        Returns:
            List of recent states
        """
        return self.get_state_history(limit=length)

    def find_loops(self) -> list[list[State]]:
        """Find loops in state history.

        Returns:
            List of state loops found
        """
        loops = []
        states = self.get_state_history()

        # Look for repeated patterns
        for i in range(len(states)):
            for j in range(i + 2, len(states)):
                if states[i] == states[j]:
                    # Found potential loop
                    loop = states[i : j + 1]
                    if len(loop) > 1:
                        loops.append(loop)

        return loops

    def get_statistics(self) -> dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary of statistics
        """
        unique_states = len({r.state_name for r in self._state_history})
        unique_transitions = len({(r.from_state, r.to_state) for r in self._transition_history})

        return {
            "total_states_recorded": self._total_states_recorded,
            "total_transitions_recorded": self._total_transitions_recorded,
            "states_in_memory": len(self._state_history),
            "transitions_in_memory": len(self._transition_history),
            "unique_states_visited": unique_states,
            "unique_transitions_executed": unique_transitions,
            "loops_detected": len(self.find_loops()),
        }

    def clear(self) -> StateMemory:
        """Clear all history (fluent).

        Returns:
            Self for chaining
        """
        self._state_history.clear()
        self._transition_history.clear()
        return self

    def set_max_history(self, states: int, transitions: int) -> StateMemory:
        """Set maximum history sizes (fluent).

        Args:
            states: Maximum state history size
            transitions: Maximum transition history size

        Returns:
            Self for chaining
        """
        self._max_state_history = states
        self._max_transition_history = transitions

        # Recreate deques with new limits
        self._state_history = deque(self._state_history, maxlen=states)
        self._transition_history = deque(self._transition_history, maxlen=transitions)

        return self

    def __str__(self) -> str:
        """String representation."""
        return (
            f"StateMemory(states={len(self._state_history)}, "
            f"transitions={len(self._transition_history)})"
        )


@dataclass
class StateRecord:
    """Record of a state visit.

    Used internally by StateMemory.
    """

    state: State | None
    timestamp: float
    state_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TransitionRecord:
    """Record of a transition execution.

    Used internally by StateMemory.
    """

    transition: StateTransition | None
    timestamp: float
    from_state: str | None
    to_state: str | None
    transition_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
