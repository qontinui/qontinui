"""Anchors class - ported from Qontinui framework.

Collection of Anchor objects for defining complex relative positioning constraints.
"""

from dataclasses import dataclass, field

from .anchor import Anchor


@dataclass
class Anchors:
    """Collection of Anchor objects for defining complex relative positioning constraints.

    Port of Anchors from Qontinui framework class.

    Anchors manages multiple Anchor objects, enabling sophisticated spatial relationships
    between visual elements. This collection allows defining regions with multiple constraints,
    creating more precise and flexible positioning rules than single anchors allow.

    Key capabilities:
    - Multiple Constraints: Apply several positioning rules simultaneously
    - Complex Shapes: Define non-rectangular or irregular regions
    - Redundant Positioning: Use multiple anchors for robustness
    - Dynamic Adjustment: Adapt to varying element arrangements

    Common patterns:
    - Two anchors to define a rectangular region between elements
    - Four anchors for precise boundary definition
    - Multiple anchors for average positioning (center of mass)
    - Fallback anchors when primary references may be absent

    Use cases:
    - Defining regions bounded by multiple visual landmarks
    - Creating adaptive search areas in complex layouts
    - Establishing relationships with multiple reference points
    - Building fault-tolerant positioning strategies
    """

    anchor_list: list[Anchor] = field(default_factory=list)

    def add(self, anchor: Anchor) -> None:
        """Add an anchor to the collection.

        Args:
            anchor: Anchor to add
        """
        self.anchor_list.append(anchor)

    def size(self) -> int:
        """Get number of anchors.

        Returns:
            Number of anchors in collection
        """
        return len(self.anchor_list)

    def is_empty(self) -> bool:
        """Check if collection is empty.

        Returns:
            True if no anchors
        """
        return self.size() == 0

    def get(self, index: int) -> Anchor | None:
        """Get anchor at specific index.

        Args:
            index: Index of anchor

        Returns:
            Anchor at index or None if out of bounds
        """
        if 0 <= index < len(self.anchor_list):
            return self.anchor_list[index]
        return None

    def get_all(self) -> list[Anchor]:
        """Get all anchors.

        Returns:
            List of all anchors
        """
        return self.anchor_list.copy()

    def clear(self) -> None:
        """Remove all anchors."""
        self.anchor_list.clear()

    def remove(self, anchor: Anchor) -> bool:
        """Remove specific anchor.

        Args:
            anchor: Anchor to remove

        Returns:
            True if anchor was removed
        """
        try:
            self.anchor_list.remove(anchor)
            return True
        except ValueError:
            return False

    def contains(self, anchor: Anchor) -> bool:
        """Check if anchor is in collection.

        Args:
            anchor: Anchor to check

        Returns:
            True if anchor is in collection
        """
        return anchor in self.anchor_list

    def __str__(self) -> str:
        """String representation."""
        anchors_str = " ".join(str(anchor) for anchor in self.anchor_list)
        return f"Anchors: {anchors_str}" if anchors_str else "Anchors: (empty)"

    def __repr__(self) -> str:
        """Developer representation."""
        return f"Anchors(anchor_list={self.anchor_list})"

    def __bool__(self) -> bool:
        """Boolean evaluation - True if has anchors."""
        return not self.is_empty()

    def __iter__(self):
        """Make Anchors iterable."""
        return iter(self.anchor_list)

    def __len__(self):
        """Support len() function."""
        return self.size()

    @classmethod
    def from_list(cls, anchors: list[Anchor]) -> "Anchors":
        """Create Anchors from list.

        Args:
            anchors: List of Anchor objects

        Returns:
            New Anchors instance
        """
        return cls(anchor_list=anchors.copy())

    @classmethod
    def from_anchor(cls, anchor: Anchor) -> "Anchors":
        """Create Anchors with single anchor.

        Args:
            anchor: Initial anchor

        Returns:
            New Anchors instance
        """
        instance = cls()
        instance.add(anchor)
        return instance
