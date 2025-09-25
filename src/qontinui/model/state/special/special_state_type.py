"""SpecialStateType enum - ported from Qontinui framework.

Defines types of special states in the system.
"""

from enum import Enum, auto


class SpecialStateType(Enum):
    """Types of special states.

    Port of SpecialStateType from Qontinui framework enum.

    Special states are used for specific system behaviors and edge cases
    that don't fit the normal state model. They provide standardized ways
    to handle common automation scenarios.
    """

    # Null state - represents absence of state
    NULL = auto()

    # Unknown state - state cannot be determined
    UNKNOWN = auto()

    # Error state - system is in error condition
    ERROR = auto()

    # Loading state - waiting for state to load
    LOADING = auto()

    # Transitioning state - between two states
    TRANSITIONING = auto()

    # Initial state - system startup state
    INITIAL = auto()

    # Terminal state - final state, no transitions out
    TERMINAL = auto()

    # Wildcard state - matches any state
    WILDCARD = auto()

    # Hidden state - state exists but is not visible
    HIDDEN = auto()

    # Disabled state - state exists but is disabled
    DISABLED = auto()

    def is_error_type(self) -> bool:
        """Check if this is an error-related state type.

        Returns:
            True if error type
        """
        return self in (SpecialStateType.ERROR, SpecialStateType.UNKNOWN)

    def is_transient(self) -> bool:
        """Check if this is a transient state type.

        Transient states are temporary and expected to change.

        Returns:
            True if transient
        """
        return self in (SpecialStateType.LOADING, SpecialStateType.TRANSITIONING)

    def is_terminal(self) -> bool:
        """Check if this is a terminal state type.

        Terminal states don't transition to other states.

        Returns:
            True if terminal
        """
        return self == SpecialStateType.TERMINAL

    def allows_transitions(self) -> bool:
        """Check if transitions are allowed from this state type.

        Returns:
            True if transitions allowed
        """
        return self not in (SpecialStateType.TERMINAL, SpecialStateType.ERROR)

    def requires_action(self) -> bool:
        """Check if this state type requires user/system action.

        Returns:
            True if action required
        """
        return self in (SpecialStateType.ERROR, SpecialStateType.DISABLED)

    def is_visible(self) -> bool:
        """Check if this state type is visible in UI.

        Returns:
            True if visible
        """
        return self != SpecialStateType.HIDDEN

    def is_actionable(self) -> bool:
        """Check if actions can be performed in this state type.

        Returns:
            True if actionable
        """
        return self not in (
            SpecialStateType.DISABLED,
            SpecialStateType.HIDDEN,
            SpecialStateType.NULL,
            SpecialStateType.LOADING,
        )

    @classmethod
    def from_string(cls, value: str) -> "SpecialStateType":
        """Create SpecialStateType from string.

        Args:
            value: String representation

        Returns:
            SpecialStateType enum value

        Raises:
            ValueError: If string doesn't match any type
        """
        value_upper = value.upper()
        for state_type in cls:
            if state_type.name == value_upper:
                return state_type
        raise ValueError(f"No SpecialStateType for '{value}'")

    def __str__(self) -> str:
        """String representation."""
        return self.name.lower()

    def __repr__(self) -> str:
        """Developer representation."""
        return f"SpecialStateType.{self.name}"
