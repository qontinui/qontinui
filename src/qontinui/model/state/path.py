"""Path - ported from Qontinui framework.

Represents a path through states.
"""

from dataclasses import dataclass, field
from typing import Any

from ..transition.state_transition import StateTransition
from .state import State


@dataclass
class Path:
    """Represents a path through states.

    Port of Path from Qontinui framework class.
    Contains a sequence of states and transitions between them.
    """

    states: list[State] = field(default_factory=list)
    transitions: list[StateTransition | None] = field(default_factory=list)

    # Path properties
    _score: float = 0.0  # Total path score
    _probability: float = 1.0  # Total path probability
    _length: int = 0  # Number of states in path

    # Metadata
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize path properties."""
        self._length = len(self.states)
        self._calculate_score()
        self._calculate_probability()

    def add_state(
        self, state: State, transition: StateTransition | None = None
    ) -> "Path":
        """Add a state to the path (fluent).

        Args:
            state: State to add
            transition: Transition to reach this state

        Returns:
            Self for chaining
        """
        self.states.append(state)
        self.transitions.append(transition)
        self._length = len(self.states)
        self._calculate_score()
        self._calculate_probability()
        return self

    def get_state(self, index: int) -> State | None:
        """Get state at index.

        Args:
            index: State index

        Returns:
            State or None
        """
        if 0 <= index < len(self.states):
            return self.states[index]
        return None

    def get_transition(self, index: int) -> StateTransition | None:
        """Get transition at index.

        Args:
            index: Transition index

        Returns:
            Transition or None
        """
        if 0 <= index < len(self.transitions):
            return self.transitions[index]
        return None

    def get_first_state(self) -> State | None:
        """Get first state in path.

        Returns:
            First state or None
        """
        return self.states[0] if self.states else None

    def get_last_state(self) -> State | None:
        """Get last state in path.

        Returns:
            Last state or None
        """
        return self.states[-1] if self.states else None

    def contains_state(self, state: State) -> bool:
        """Check if path contains a state.

        Args:
            state: State to check

        Returns:
            True if state is in path
        """
        return state in self.states

    def contains_loop(self) -> bool:
        """Check if path contains a loop.

        Returns:
            True if path has repeated states
        """
        return len(self.states) != len(set(self.states))

    def get_loops(self) -> list[list[State]]:
        """Get all loops in the path.

        Returns:
            List of state loops
        """
        loops = []
        for i in range(len(self.states)):
            for j in range(i + 1, len(self.states)):
                if self.states[i] == self.states[j]:
                    loop = self.states[i : j + 1]
                    loops.append(loop)
        return loops

    def remove_loops(self) -> "Path":
        """Remove loops from path (fluent).

        Returns:
            Self for chaining
        """
        if not self.contains_loop():
            return self

        # Keep only first occurrence of each state
        seen = set()
        new_states = []
        new_transitions = []

        for i, state in enumerate(self.states):
            if state not in seen:
                seen.add(state)
                new_states.append(state)
                if i < len(self.transitions):
                    new_transitions.append(self.transitions[i])

        self.states = new_states
        self.transitions = new_transitions
        self._length = len(self.states)
        self._calculate_score()
        self._calculate_probability()

        return self

    def reverse(self) -> "Path":
        """Create reversed path.

        Returns:
            New reversed Path
        """
        reversed_path = Path()
        reversed_path.states = list(reversed(self.states))

        # Transitions need special handling
        # The reversed transitions might not exist
        reversed_path.transitions = [None] * len(reversed_path.states)

        return reversed_path

    def subpath(self, start: int, end: int) -> "Path":
        """Get subpath from start to end index.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)

        Returns:
            New Path with subpath
        """
        subpath = Path()
        subpath.states = self.states[start:end]
        subpath.transitions = self.transitions[start:end]
        return subpath

    def merge(self, other: "Path") -> "Path":
        """Merge this path with another.

        Args:
            other: Path to merge with

        Returns:
            New merged Path
        """
        merged = Path()
        merged.states = self.states + other.states
        merged.transitions = self.transitions + other.transitions
        return merged

    def _calculate_score(self):
        """Calculate total path score."""
        self._score = 0.0
        for transition in self.transitions:
            if transition:
                self._score += transition.score

    def _calculate_probability(self):
        """Calculate total path probability."""
        self._probability = 1.0
        for transition in self.transitions:
            if transition:
                self._probability *= transition.probability

    def get_score(self) -> float:
        """Get total path score.

        Returns:
            Path score
        """
        return self._score

    def get_probability(self) -> float:
        """Get total path probability.

        Returns:
            Path probability
        """
        return self._probability

    def get_length(self) -> int:
        """Get path length.

        Returns:
            Number of states
        """
        return self._length

    def is_empty(self) -> bool:
        """Check if path is empty.

        Returns:
            True if no states
        """
        return len(self.states) == 0

    def is_valid(self) -> bool:
        """Check if path is valid.

        Returns:
            True if path has valid transitions
        """
        if len(self.states) < 2:
            return True

        # Check that transitions connect states properly
        for i in range(len(self.states) - 1):
            transition = self.transitions[i] if i < len(self.transitions) else None
            if transition:
                if transition.from_state != self.states[i]:
                    return False
                if transition.to_state != self.states[i + 1]:
                    return False

        return True

    def to_string(self) -> str:
        """Get string representation of path.

        Returns:
            Path as string
        """
        if not self.states:
            return "Empty path"

        path_str = self.states[0].name
        for i in range(1, len(self.states)):
            transition = (
                self.transitions[i - 1] if i - 1 < len(self.transitions) else None
            )
            if transition:
                # transition_type is already a string, not an enum
                transition_type = transition.transition_type
                path_str += f" --[{transition_type}]--> "
            else:
                path_str += " --> "
            path_str += self.states[i].name

        return path_str

    def __str__(self) -> str:
        """String representation."""
        if not self.states:
            return "Path(empty)"

        start = self.states[0].name
        end = self.states[-1].name
        return f"Path({start} -> {end}, length={self._length})"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"Path(states={len(self.states)}, score={self._score:.2f}, probability={self._probability:.2f})"

    def __len__(self) -> int:
        """Path length."""
        return len(self.states)

    def __iter__(self):
        """Iterator over states."""
        return iter(self.states)

    def __getitem__(self, index: int) -> State:
        """Get state by index."""
        return self.states[index]
