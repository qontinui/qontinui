"""StateTransitions - ported from Qontinui framework.

Container for multiple state transitions.
"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..state.state import State

from ..transition.state_transition import StateTransition
from ..transition.transition_function import TransitionType


@dataclass
class StateTransitions:
    """Container for multiple state transitions.

    Port of StateTransitions from Qontinui framework class.
    Groups and manages related transitions.
    """

    transitions: list[StateTransition] = field(default_factory=list)
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    state_name: str | None = None

    def get_state_name(self) -> str | None:
        """Get the state name this transition set is associated with.

        Returns:
            State name or None
        """
        return self.state_name

    def get_transitions(self) -> list[StateTransition]:
        """Get all transitions in this set.

        Returns:
            List of transitions
        """
        return self.transitions

    @classmethod
    def builder(cls) -> StateTransitionsBuilder:
        """Create a builder for StateTransitions.

        Returns:
            StateTransitionsBuilder instance
        """
        return StateTransitionsBuilder()

    def add(self, transition: StateTransition) -> StateTransitions:
        """Add a transition (fluent).

        Args:
            transition: Transition to add

        Returns:
            Self for chaining
        """
        self.transitions.append(transition)
        return self

    def add_all(self, transitions: list[StateTransition]) -> StateTransitions:
        """Add multiple transitions (fluent).

        Args:
            transitions: List of transitions to add

        Returns:
            Self for chaining
        """
        self.transitions.extend(transitions)
        return self

    def get_from_state(self, state: State) -> list[StateTransition]:
        """Get all transitions from a specific state.

        Args:
            state: Source state

        Returns:
            List of transitions from state
        """
        # Filter transitions where from_state matches
        return [t for t in self.transitions if t.from_state == state.name]

    def get_to_state(self, state: State) -> list[StateTransition]:
        """Get all transitions to a specific state.

        Args:
            state: Target state

        Returns:
            List of transitions to state
        """
        # Filter transitions where to_state matches
        return [t for t in self.transitions if t.to_state == state.name]

    def get_between_states(self, from_state: State, to_state: State) -> list[StateTransition]:
        """Get transitions between two states.

        Args:
            from_state: Source state
            to_state: Target state

        Returns:
            List of transitions between states
        """
        # Filter transitions matching both from and to states
        return [
            t
            for t in self.transitions
            if t.from_state == from_state.name and t.to_state == to_state.name
        ]

    def get_by_type(self, transition_type: TransitionType) -> list[StateTransition]:
        """Get transitions of a specific type.

        Args:
            transition_type: Type of transition

        Returns:
            List of transitions of that type
        """
        return [t for t in self.transitions if t.transition_type == transition_type]

    def get_best_transition(self, from_state: State, to_state: State) -> StateTransition | None:
        """Get the best transition between states.

        Args:
            from_state: Source state
            to_state: Target state

        Returns:
            Best transition or None
        """
        candidates = self.get_between_states(from_state, to_state)
        if not candidates:
            return None

        # Return transition with highest score * probability
        return max(candidates, key=lambda t: t.score * t.probability)

    def execute_first_valid(self) -> bool:
        """Execute the first valid transition.

        Returns:
            True if a transition was executed successfully
        """
        for transition in self.transitions:
            if transition.check_conditions():
                if transition.execute():
                    return True
        return False

    def execute_all_valid(self) -> int:
        """Execute all valid transitions.

        Returns:
            Number of successful executions
        """
        successful = 0
        for transition in self.transitions:
            if transition.check_conditions():
                if transition.execute():
                    successful += 1
        return successful

    def filter_by_probability(self, min_probability: float) -> StateTransitions:
        """Filter transitions by minimum probability.

        Args:
            min_probability: Minimum probability threshold

        Returns:
            New StateTransitions with filtered transitions
        """
        filtered = [t for t in self.transitions if t.probability >= min_probability]
        return StateTransitions(transitions=filtered, name=f"{self.name}_filtered")

    def sort_by_score(self, reverse: bool = True) -> StateTransitions:
        """Sort transitions by score.

        Args:
            reverse: True for descending order

        Returns:
            Self for chaining
        """
        self.transitions.sort(key=lambda t: t.score, reverse=reverse)
        return self

    def sort_by_probability(self, reverse: bool = True) -> StateTransitions:
        """Sort transitions by probability.

        Args:
            reverse: True for descending order

        Returns:
            Self for chaining
        """
        self.transitions.sort(key=lambda t: t.probability, reverse=reverse)
        return self

    def clear(self) -> StateTransitions:
        """Clear all transitions.

        Returns:
            Self for chaining
        """
        self.transitions.clear()
        return self

    def size(self) -> int:
        """Get number of transitions.

        Returns:
            Number of transitions
        """
        return len(self.transitions)

    def is_empty(self) -> bool:
        """Check if empty.

        Returns:
            True if no transitions
        """
        return len(self.transitions) == 0

    def __iter__(self):
        """Iterator over transitions."""
        return iter(self.transitions)

    def __len__(self) -> int:
        """Number of transitions."""
        return len(self.transitions)

    def __getitem__(self, index: int) -> StateTransition:
        """Get transition by index."""
        return self.transitions[index]

    def __str__(self) -> str:
        """String representation."""
        return f"StateTransitions({len(self.transitions)} transitions)"


class StateTransitionsBuilder:
    """Builder for creating StateTransitions objects."""

    def __init__(self) -> None:
        """Initialize builder."""
        self._transitions: list[StateTransition] = []
        self._name: str | None = None
        self._state_name: str | None = None
        self._metadata: dict[str, Any] = {}

    def with_name(self, name: str) -> StateTransitionsBuilder:
        """Set the name (fluent).

        Args:
            name: Name for the transitions

        Returns:
            Self for chaining
        """
        self._name = name
        return self

    def with_state_name(self, state_name: str) -> StateTransitionsBuilder:
        """Set the associated state name (fluent).

        Args:
            state_name: State name

        Returns:
            Self for chaining
        """
        self._state_name = state_name
        return self

    def add_transition(self, transition: StateTransition) -> StateTransitionsBuilder:
        """Add a transition (fluent).

        Args:
            transition: Transition to add

        Returns:
            Self for chaining
        """
        self._transitions.append(transition)
        return self

    def add_transitions(self, transitions: list[StateTransition]) -> StateTransitionsBuilder:
        """Add multiple transitions (fluent).

        Args:
            transitions: List of transitions to add

        Returns:
            Self for chaining
        """
        self._transitions.extend(transitions)
        return self

    def with_metadata(self, key: str, value: Any) -> StateTransitionsBuilder:
        """Add metadata (fluent).

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            Self for chaining
        """
        self._metadata[key] = value
        return self

    def build(self) -> StateTransitions:
        """Build the StateTransitions object.

        Returns:
            Constructed StateTransitions
        """
        return StateTransitions(
            transitions=self._transitions,
            name=self._name,
            state_name=self._state_name,
            metadata=self._metadata,
        )
