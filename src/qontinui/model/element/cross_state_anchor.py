"""CrossStateAnchor class - ported from Qontinui framework.

Provides anchoring between states for multi-state operations.
"""

from dataclasses import dataclass, field
from typing import cast

from ..state.state_object import StateObject
from .anchor import Anchor
from .location import Location


@dataclass
class CrossStateAnchor:
    """Anchors state objects across multiple states.

    Port of CrossStateAnchor from Qontinui framework class.

    CrossStateAnchor enables operations that span multiple states by providing
    positional references that persist across state transitions. This is crucial
    for workflows where UI elements maintain relative positions across screens
    or where actions need to reference elements from previous states.

    Use cases:
    - Multi-step wizards where buttons stay in same position
    - Drag operations across state boundaries
    - Reference points that persist through transitions
    - Validation that elements moved correctly between states

    Example:
        # Anchor a button position that persists across wizard steps
        anchor = CrossStateAnchor()
        anchor.add_anchor("wizard_step1", button_anchor)
        anchor.add_anchor("wizard_step2", button_anchor)

        # Use anchor to click button in any wizard state
        location = anchor.get_location_for_state(current_state)
    """

    # Map of state names to anchors
    state_anchors: dict[str, Anchor] = field(default_factory=dict)

    # Primary state object being anchored
    primary_object: StateObject | None = None

    # Default anchor to use when state-specific anchor not found
    default_anchor: Anchor | None = None

    # Whether to allow anchoring to states not in the map
    strict_mode: bool = False

    def add_anchor(self, state_name: str, anchor: Anchor) -> "CrossStateAnchor":
        """Add an anchor for a specific state.

        Args:
            state_name: Name of the state
            anchor: Anchor to use in that state

        Returns:
            Self for fluent interface
        """
        self.state_anchors[state_name] = anchor
        return self

    def add_anchors(self, anchors: dict[str, Anchor]) -> "CrossStateAnchor":
        """Add multiple state anchors at once.

        Args:
            anchors: Map of state names to anchors

        Returns:
            Self for fluent interface
        """
        self.state_anchors.update(anchors)
        return self

    def set_primary_object(self, obj: StateObject) -> "CrossStateAnchor":
        """Set the primary state object being anchored.

        Args:
            obj: Primary state object

        Returns:
            Self for fluent interface
        """
        self.primary_object = obj
        return self

    def set_default_anchor(self, anchor: Anchor) -> "CrossStateAnchor":
        """Set default anchor for states not in map.

        Args:
            anchor: Default anchor

        Returns:
            Self for fluent interface
        """
        self.default_anchor = anchor
        return self

    def set_strict_mode(self, strict: bool) -> "CrossStateAnchor":
        """Set whether to enforce strict state matching.

        Args:
            strict: If True, raises error for unknown states

        Returns:
            Self for fluent interface
        """
        self.strict_mode = strict
        return self

    def get_anchor_for_state(self, state_name: str) -> Anchor | None:
        """Get anchor for a specific state.

        Args:
            state_name: Name of the state

        Returns:
            Anchor for the state, default anchor, or None

        Raises:
            ValueError: If strict mode and state not found
        """
        if state_name in self.state_anchors:
            return self.state_anchors[state_name]

        if self.strict_mode and self.default_anchor is None:
            raise ValueError(f"No anchor found for state '{state_name}' in strict mode")

        return self.default_anchor

    def get_location_for_state(
        self, state_name: str, base_location: Location | None = None
    ) -> Location | None:
        """Get anchored location for a specific state.

        Args:
            state_name: Name of the state
            base_location: Optional base location to anchor from

        Returns:
            Anchored location or None if no anchor found
        """
        anchor = self.get_anchor_for_state(state_name)
        if anchor is None:
            return None

        # If we have a base location, apply anchor offset
        if base_location:
            return Location(base_location.x + anchor.offset_x, base_location.y + anchor.offset_y)

        # Otherwise return anchor's absolute position if available
        pos = anchor.get_position()
        if pos:
            return Location(pos[0], pos[1])

        return None

    def has_state(self, state_name: str) -> bool:
        """Check if anchor exists for state.

        Args:
            state_name: Name of the state

        Returns:
            True if state has anchor
        """
        return state_name in self.state_anchors

    def get_states(self) -> list[str]:
        """Get list of all states with anchors.

        Returns:
            List of state names
        """
        return list(self.state_anchors.keys())

    def remove_state(self, state_name: str) -> bool:
        """Remove anchor for a state.

        Args:
            state_name: Name of the state

        Returns:
            True if state was removed
        """
        if state_name in self.state_anchors:
            del self.state_anchors[state_name]
            return True
        return False

    def clear(self) -> "CrossStateAnchor":
        """Clear all state anchors.

        Returns:
            Self for fluent interface
        """
        self.state_anchors.clear()
        return self

    def is_valid(self) -> bool:
        """Check if cross-state anchor is valid.

        Returns:
            True if has at least one anchor or default
        """
        return bool(self.state_anchors) or self.default_anchor is not None

    def anchor_count(self) -> int:
        """Get number of state anchors.

        Returns:
            Count of state anchors
        """
        return len(self.state_anchors)

    def merge(self, other: "CrossStateAnchor") -> "CrossStateAnchor":
        """Merge another cross-state anchor into this one.

        Args:
            other: CrossStateAnchor to merge

        Returns:
            Self for fluent interface
        """
        self.state_anchors.update(other.state_anchors)
        if other.default_anchor and not self.default_anchor:
            self.default_anchor = other.default_anchor
        if other.primary_object and not self.primary_object:
            self.primary_object = other.primary_object
        return self

    def copy(self) -> "CrossStateAnchor":
        """Create a copy of this cross-state anchor.

        Returns:
            New CrossStateAnchor instance
        """
        return CrossStateAnchor(
            state_anchors=self.state_anchors.copy(),
            primary_object=self.primary_object,
            default_anchor=self.default_anchor,
            strict_mode=self.strict_mode,
        )

    def __str__(self) -> str:
        """String representation."""
        states = len(self.state_anchors)
        default = "with default" if self.default_anchor else "no default"
        strict = "strict" if self.strict_mode else "flexible"
        return f"CrossStateAnchor({states} states, {default}, {strict})"

    def __repr__(self) -> str:
        """Developer representation."""
        return (
            f"CrossStateAnchor(state_anchors={list(self.state_anchors.keys())}, "
            f"has_default={self.default_anchor is not None}, "
            f"strict_mode={self.strict_mode})"
        )


class CrossStateAnchorBuilder:
    """Builder for CrossStateAnchor.

    Provides fluent interface for constructing cross-state anchors.
    """

    def __init__(self):
        """Initialize builder."""
        self._anchor = CrossStateAnchor()

    def with_state_anchor(self, state_name: str, anchor: Anchor) -> "CrossStateAnchorBuilder":
        """Add a state anchor.

        Args:
            state_name: Name of the state
            anchor: Anchor for that state

        Returns:
            Self for chaining
        """
        self._anchor.add_anchor(state_name, anchor)
        return self

    def with_state_anchors(self, anchors: dict[str, Anchor]) -> "CrossStateAnchorBuilder":
        """Add multiple state anchors.

        Args:
            anchors: Map of state names to anchors

        Returns:
            Self for chaining
        """
        self._anchor.add_anchors(anchors)
        return self

    def with_primary_object(self, obj: StateObject) -> "CrossStateAnchorBuilder":
        """Set primary object.

        Args:
            obj: Primary state object

        Returns:
            Self for chaining
        """
        self._anchor.set_primary_object(obj)
        return self

    def with_default_anchor(self, anchor: Anchor) -> "CrossStateAnchorBuilder":
        """Set default anchor.

        Args:
            anchor: Default anchor

        Returns:
            Self for chaining
        """
        self._anchor.set_default_anchor(anchor)
        return self

    def strict(self) -> "CrossStateAnchorBuilder":
        """Enable strict mode.

        Returns:
            Self for chaining
        """
        self._anchor.set_strict_mode(True)
        return self

    def flexible(self) -> "CrossStateAnchorBuilder":
        """Disable strict mode (default).

        Returns:
            Self for chaining
        """
        self._anchor.set_strict_mode(False)
        return self

    def build(self) -> CrossStateAnchor:
        """Build the CrossStateAnchor.

        Returns:
            Configured CrossStateAnchor instance
        """
        return cast(CrossStateAnchor, self._anchor.copy())
