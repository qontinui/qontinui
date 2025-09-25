"""Pair utility class - ported from Qontinui framework.

Generic immutable container for holding two related values as a single unit.
"""

from dataclasses import dataclass
from typing import TypeVar

U = TypeVar("U")
V = TypeVar("V")


@dataclass(frozen=True)
class Pair[U, V]:
    """Generic immutable container for holding two related values.

    Port of Pair from Qontinui framework class.

    Pair provides a type-safe way to group two objects together without creating
    specialized classes. This utility class is essential throughout Qontinui for
    representing dual relationships, coordinates, mappings, and any scenario where
    two values naturally belong together. The immutable design ensures thread safety
    and prevents accidental modifications.

    Key characteristics:
    - Immutable: Values cannot be changed after construction
    - Type-Safe: Generic parameters ensure compile-time type checking
    - Null-Safe: Properly handles None values in equals/hash
    - Value Semantics: Equality based on contained values
    - Map-Compatible: Provides get_key() and get_value() methods

    Common usage patterns in Qontinui:
    - Coordinates: Pair[int, int] for x,y positions
    - Mappings: Pair[State, Transition] for state relationships
    - Results: Pair[bool, str] for success/message returns
    - Ranges: Pair[float, float] for min/max values
    - Associations: Pair[Pattern, Region] for search contexts

    Factory method usage:
        # Create coordinate pair
        point = Pair.of(100, 200)

        # Create state-transition pair
        mapping = Pair.of(state, transition)

        # Create result pair
        result = Pair.of(True, "Success")

    Design benefits:
    - Eliminates need for multiple specialized two-field classes
    - Provides consistent API for paired data
    - Supports use in collections (proper equals/hash)
    - Clear semantics with descriptive field names
    - Interoperable with dict via get_key/get_value

    Thread safety:
    - Immutable fields ensure thread-safe access
    - No synchronization needed for reads
    - Safe to share across threads
    - Values should also be immutable for complete safety

    In the model-based approach, Pair serves as a fundamental building block for
    representing binary relationships throughout the framework. Its simplicity and
    immutability make it ideal for functional programming patterns and concurrent
    operations common in automation scenarios.
    """

    first: U
    """First field of the pair."""

    second: V
    """Second field of the pair."""

    @classmethod
    def of(cls, first: U, second: V) -> "Pair[U, V]":
        """Factory method for creating a typed Pair instance.

        Args:
            first: First value
            second: Second value

        Returns:
            New Pair instance
        """
        return cls(first=first, second=second)

    def get_key(self) -> U:
        """Get the first value (key).

        Map.Entry compatibility method.

        Returns:
            First value
        """
        return self.first

    def get_value(self) -> V:
        """Get the second value.

        Map.Entry compatibility method.

        Returns:
            Second value
        """
        return self.second

    def to_tuple(self) -> tuple[U, V]:
        """Convert to tuple.

        Returns:
            (first, second) tuple
        """
        return (self.first, self.second)

    def __str__(self) -> str:
        """String representation."""
        return f"({self.first}, {self.second})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Pair(first={self.first!r}, second={self.second!r})"
