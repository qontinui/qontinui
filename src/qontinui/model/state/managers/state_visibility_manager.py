"""State visibility manager - handles state hiding and blocking logic.

Manages which states can be hidden by this state and tracks currently
hidden states, supporting the state hierarchy and occlusion model.
"""

from dataclasses import dataclass, field


@dataclass
class StateVisibilityManager:
    """Manages state visibility, hiding, and blocking relationships.

    Responsible for tracking which states can be hidden by this state,
    which states are currently hidden, and blocking behavior.
    """

    blocking: bool = False
    """When true, this State needs to be acted on before accessing other States."""

    can_hide: set[str] = field(default_factory=set)
    """States that this State can hide when it becomes active."""

    can_hide_ids: set[int] = field(default_factory=set)
    """IDs of states that this State can hide."""

    hidden_state_names: set[str] = field(default_factory=set)
    """Names of currently hidden states (used when initializing in code)."""

    hidden_state_ids: set[int] = field(default_factory=set)
    """IDs of currently hidden states (used at runtime)."""

    def set_blocking(self, blocking: bool) -> None:
        """Set the blocking state.

        Args:
            blocking: Whether this state is blocking
        """
        self.blocking = blocking

    def is_blocking(self) -> bool:
        """Check if this state is blocking.

        Returns:
            True if state is blocking, False otherwise
        """
        return self.blocking

    def add_can_hide(self, state_name: str) -> None:
        """Add a state that this state can hide.

        Args:
            state_name: Name of state that can be hidden
        """
        self.can_hide.add(state_name)

    def add_can_hide_id(self, state_id: int) -> None:
        """Add a state ID that this state can hide.

        Args:
            state_id: ID of state that can be hidden
        """
        self.can_hide_ids.add(state_id)

    def can_hide_state(self, state_name: str) -> bool:
        """Check if this state can hide another state.

        Args:
            state_name: Name of state to check

        Returns:
            True if this state can hide the given state
        """
        return state_name in self.can_hide

    def can_hide_state_id(self, state_id: int) -> bool:
        """Check if this state can hide another state by ID.

        Args:
            state_id: ID of state to check

        Returns:
            True if this state can hide the given state
        """
        return state_id in self.can_hide_ids

    def add_hidden_state(self, state_id: int) -> None:
        """Add a currently hidden state ID.

        Args:
            state_id: ID of state to mark as hidden
        """
        self.hidden_state_ids.add(state_id)

    def add_hidden_state_name(self, state_name: str) -> None:
        """Add a currently hidden state name.

        Args:
            state_name: Name of state to mark as hidden
        """
        self.hidden_state_names.add(state_name)

    def remove_hidden_state(self, state_id: int) -> None:
        """Remove a state from the hidden set.

        Args:
            state_id: ID of state to unhide
        """
        self.hidden_state_ids.discard(state_id)

    def remove_hidden_state_name(self, state_name: str) -> None:
        """Remove a state name from the hidden set.

        Args:
            state_name: Name of state to unhide
        """
        self.hidden_state_names.discard(state_name)

    def is_state_hidden(self, state_id: int) -> bool:
        """Check if a state is currently hidden.

        Args:
            state_id: ID of state to check

        Returns:
            True if state is hidden, False otherwise
        """
        return state_id in self.hidden_state_ids

    def is_state_hidden_by_name(self, state_name: str) -> bool:
        """Check if a state is currently hidden by name.

        Args:
            state_name: Name of state to check

        Returns:
            True if state is hidden, False otherwise
        """
        return state_name in self.hidden_state_names

    def reset_hidden(self) -> None:
        """Reset all hidden state names."""
        self.hidden_state_names = set()

    def clear_hidden_ids(self) -> None:
        """Clear all hidden state IDs."""
        self.hidden_state_ids = set()

    def get_hidden_state_count(self) -> int:
        """Get the number of currently hidden states.

        Returns:
            Number of hidden states
        """
        return len(self.hidden_state_ids)

    def __str__(self) -> str:
        """String representation."""
        parts = []
        parts.append(f"Blocking: {self.blocking}")
        parts.append(f"Can hide {len(self.can_hide)} states: {sorted(self.can_hide)}")
        parts.append(f"Currently hiding {len(self.hidden_state_ids)} states")
        return "\n".join(parts)
