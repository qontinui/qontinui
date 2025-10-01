"""ActiveStateSet - ported from Qontinui framework.

Lightweight enum-based state tracking for building state collections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ..model.state.state import State


@dataclass
class ActiveStateSet:
    """Lightweight enum-based state tracking.

    Port of ActiveStateSet from Qontinui framework class.

    Provides a simple way to build and manage collections of states
    using state enums for type-safe state references.
    """

    # Active states collection
    states: set[int | Enum] = field(default_factory=set)

    def add_state(self, state: int | Enum | State) -> ActiveStateSet:
        """Add a state to the set.

        Args:
            state: State ID, enum, or State object

        Returns:
            Self for fluent interface
        """
        if hasattr(state, "id"):
            # State object
            state_id = state.id
            if state_id is not None:
                self.states.add(state_id)
        elif isinstance(state, Enum):
            # State enum
            self.states.add(state)
        elif isinstance(state, int):
            # Direct ID
            self.states.add(state)
        return self

    def add_states(self, *states: int | Enum | State) -> ActiveStateSet:
        """Add multiple states to the set.

        Args:
            *states: Variable number of states to add

        Returns:
            Self for fluent interface
        """
        for state in states:
            self.add_state(state)
        return self

    def remove_state(self, state: int | Enum | State) -> ActiveStateSet:
        """Remove a state from the set.

        Args:
            state: State to remove

        Returns:
            Self for fluent interface
        """
        if hasattr(state, "id"):
            state_id = state.id
            if state_id is not None:
                self.states.discard(state_id)
        elif isinstance(state, Enum):
            self.states.discard(state)
        elif isinstance(state, int):
            self.states.discard(state)
        else:
            self.states.discard(state)
        return self

    def clear(self) -> ActiveStateSet:
        """Remove all states from the set.

        Returns:
            Self for fluent interface
        """
        self.states.clear()
        return self

    def contains(self, state: int | Enum | State) -> bool:
        """Check if state is in the set.

        Args:
            state: State to check

        Returns:
            True if state is in set
        """
        if hasattr(state, "id"):
            return state.id in self.states
        elif isinstance(state, Enum):
            return state in self.states
        else:
            return state in self.states

    def get_active_states(self) -> set[int | Enum]:
        """Get the set of active states.

        Returns:
            Copy of active states set
        """
        return self.states.copy()

    def get_state_ids(self) -> set[int]:
        """Get state IDs only (filtering out enums).

        Returns:
            Set of integer state IDs
        """
        ids = set()
        for state in self.states:
            if isinstance(state, int):
                ids.add(state)
            elif hasattr(state, "value") and isinstance(state.value, int):
                # Enum with integer value
                ids.add(state.value)
        return ids

    def get_state_enums(self) -> set[Enum]:
        """Get state enums only (filtering out IDs).

        Returns:
            Set of state enums
        """
        enums = set()
        for state in self.states:
            if isinstance(state, Enum):
                enums.add(state)
        return enums

    def is_empty(self) -> bool:
        """Check if set is empty.

        Returns:
            True if no states in set
        """
        return len(self.states) == 0

    def size(self) -> int:
        """Get number of states in set.

        Returns:
            Count of states
        """
        return len(self.states)

    def merge(self, other: ActiveStateSet) -> ActiveStateSet:
        """Merge another state set into this one.

        Args:
            other: State set to merge

        Returns:
            Self for fluent interface
        """
        self.states.update(other.states)
        return self

    def intersect(self, other: ActiveStateSet) -> ActiveStateSet:
        """Keep only states present in both sets.

        Args:
            other: State set to intersect with

        Returns:
            Self for fluent interface
        """
        self.states.intersection_update(other.states)
        return self

    def difference(self, other: ActiveStateSet) -> ActiveStateSet:
        """Remove states present in other set.

        Args:
            other: State set to subtract

        Returns:
            Self for fluent interface
        """
        self.states.difference_update(other.states)
        return self

    @staticmethod
    def of(*states: int | Enum | State) -> ActiveStateSet:
        """Create state set from states.

        Args:
            *states: States to include

        Returns:
            New ActiveStateSet
        """
        return ActiveStateSet().add_states(*states)

    @staticmethod
    def empty() -> ActiveStateSet:
        """Create empty state set.

        Returns:
            Empty ActiveStateSet
        """
        return ActiveStateSet()

    def to_list(self) -> list[int | Enum]:
        """Convert to list.

        Returns:
            List of states
        """
        return list(self.states)

    def __contains__(self, state: int | Enum | State) -> bool:
        """Support 'in' operator.

        Args:
            state: State to check

        Returns:
            True if state in set
        """
        return self.contains(state)

    def __len__(self) -> int:
        """Support len() function.

        Returns:
            Number of states
        """
        return self.size()

    def __bool__(self) -> bool:
        """Support boolean evaluation.

        Returns:
            True if set is not empty
        """
        return not self.is_empty()

    def __iter__(self):
        """Support iteration.

        Returns:
            Iterator over states
        """
        return iter(self.states)

    def __repr__(self) -> str:
        """String representation.

        Returns:
            Description of state set
        """
        if self.is_empty():
            return "ActiveStateSet(empty)"

        # Show first few states
        states_list = list(self.states)[:3]
        states_str = ", ".join(str(s) for s in states_list)
        if len(self.states) > 3:
            states_str += f", ... ({len(self.states)} total)"

        return f"ActiveStateSet({states_str})"


class ActiveStateSetBuilder:
    """Builder for ActiveStateSet.

    Provides fluent interface for building state sets.
    """

    def __init__(self):
        """Initialize builder."""
        self._set = ActiveStateSet()

    def add(self, state: int | Enum | State) -> ActiveStateSetBuilder:
        """Add a state.

        Args:
            state: State to add

        Returns:
            Self for fluent interface
        """
        self._set.add_state(state)
        return self

    def add_all(self, *states: int | Enum | State) -> ActiveStateSetBuilder:
        """Add multiple states.

        Args:
            *states: States to add

        Returns:
            Self for fluent interface
        """
        self._set.add_states(*states)
        return self

    def build(self) -> ActiveStateSet:
        """Build the state set.

        Returns:
            Completed ActiveStateSet
        """
        return cast(ActiveStateSet, self._set)
