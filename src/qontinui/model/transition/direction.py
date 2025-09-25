"""Direction enum - ported from Qontinui framework.

Represents directional relationships in state transitions and navigation.
"""

from enum import Enum


class Direction(Enum):
    """Represents directional relationships in state transitions and navigation.

    Port of Direction from Qontinui framework enum.

    Direction provides a simple but essential enumeration for specifying the
    orientation of relationships between states, transitions, and navigation paths.
    This binary distinction between TO and FROM enables clear expression of
    directional semantics throughout the framework, particularly in state graph
    traversal and transition definitions.

    Usage contexts:
    - State Transitions: Specify transition direction in state graphs
    - Path Navigation: Indicate traversal direction along paths
    - Relationship Queries: Find states connected TO or FROM a given state
    - Animation Direction: Control visual feedback direction
    - Data Flow: Express information flow between components

    Common patterns:
    - TO: Moving towards a target state or destination
    - FROM: Coming from a source state or origin
    - Bidirectional: Using both TO and FROM for complete relationships
    - Query filters: Finding all transitions TO or FROM specific states

    Examples in state management:
    - Find all states reachable FROM current state (TO direction)
    - Find all states that can reach current state (FROM direction)
    - Define transition direction: LoginState TO HomeState
    - Reverse navigation: Going FROM destination back TO origin

    Semantic clarity:
    - TO implies forward movement or target-oriented action
    - FROM implies backward reference or source-oriented query
    - Direction-neutral operations don't use this enum
    - Always relative to a reference point (usually current state)
    """

    TO = "TO"
    FROM = "FROM"

    def is_forward(self) -> bool:
        """Check if this is forward direction (TO).

        Returns:
            True if direction is TO
        """
        return self == Direction.TO

    def is_backward(self) -> bool:
        """Check if this is backward direction (FROM).

        Returns:
            True if direction is FROM
        """
        return self == Direction.FROM

    def reverse(self) -> "Direction":
        """Get the opposite direction.

        Returns:
            FROM if current is TO, TO if current is FROM
        """
        return Direction.FROM if self == Direction.TO else Direction.TO

    def __str__(self) -> str:
        """String representation."""
        return self.value

    @classmethod
    def from_string(cls, direction: str) -> "Direction":
        """Create Direction from string.

        Args:
            direction: Direction string (case-insensitive)

        Returns:
            Direction enum value

        Raises:
            ValueError: If direction string is invalid
        """
        try:
            return cls[direction.upper()]
        except KeyError as e:
            raise ValueError(f"Invalid direction: {direction}") from e
