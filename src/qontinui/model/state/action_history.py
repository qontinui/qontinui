"""ActionHistory - Collection of ActionSnapshots for state objects.

Maintains a history of all actions performed on or with a state object,
enabling replay, testing, and analysis of automation behavior.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .action_snapshot import ActionSnapshot, ActionType


@dataclass
class ActionHistory:
    """Collection of ActionSnapshots associated with a state object.

    Stores all recorded actions for a StateImage, StateLocation, or StateRegion,
    providing methods to query, filter, and select snapshots for replay.

    This implements Qontinui's get_random_snapshot logic for selecting
    appropriate actions based on current state and context.
    """

    snapshots: list[ActionSnapshot] = field(default_factory=list)
    last_updated: datetime | None = None

    def add_snapshot(self, snapshot: ActionSnapshot) -> None:
        """Add a new snapshot to the history."""
        self.snapshots.append(snapshot)
        self.last_updated = datetime.now()

    def remove_snapshot(self, snapshot_id: str) -> bool:
        """Remove a snapshot by ID.

        Returns:
            True if removed, False if not found
        """
        initial_length = len(self.snapshots)
        self.snapshots = [s for s in self.snapshots if s.id != snapshot_id]
        if len(self.snapshots) < initial_length:
            self.last_updated = datetime.now()
            return True
        return False

    def get_snapshot(self, snapshot_id: str) -> ActionSnapshot | None:
        """Get a specific snapshot by ID."""
        for snapshot in self.snapshots:
            if snapshot.id == snapshot_id:
                return snapshot
        return None

    def get_snapshots_by_action_type(self, action_type: ActionType) -> list[ActionSnapshot]:
        """Get all snapshots of a specific action type."""
        return [s for s in self.snapshots if s.action_type == action_type]

    def get_snapshots_by_state(self, state_id: str) -> list[ActionSnapshot]:
        """Get all snapshots taken in a specific state."""
        return [s for s in self.snapshots if s.matches_state(state_id)]

    def get_successful_snapshots(self) -> list[ActionSnapshot]:
        """Get all successful snapshots."""
        return [s for s in self.snapshots if s.is_successful()]

    def get_failed_snapshots(self) -> list[ActionSnapshot]:
        """Get all failed snapshots."""
        return [s for s in self.snapshots if not s.is_successful()]

    def get_random_snapshot(
        self, action_type: ActionType, active_states: list[str], prefer_success: bool = True
    ) -> ActionSnapshot | None:
        """Get a random snapshot matching criteria.

        This implements Qontinui's snapshot selection logic for integration testing.

        Args:
            action_type: Type of action to match
            active_states: Currently active state IDs
            prefer_success: Whether to prefer successful snapshots

        Returns:
            A matching snapshot or None if no match found
        """
        # Filter by action type
        candidates = self.get_snapshots_by_action_type(action_type)

        if not candidates:
            return None

        # Filter by success if requested
        if prefer_success:
            successful = [s for s in candidates if s.is_successful()]
            if successful:
                candidates = successful

        # Try exact state matches first
        exact_matches = [s for s in candidates if s.state_id in active_states]

        if exact_matches:
            return random.choice(exact_matches)

        # Try overlapping active states
        overlap_matches = [
            s for s in candidates if any(state in active_states for state in s.active_states)
        ]

        if overlap_matches:
            return random.choice(overlap_matches)

        # Fall back to any matching action type
        return random.choice(candidates)

    def get_transitions_from_screenshot(self, screenshot_id: str) -> list[ActionSnapshot]:
        """Get all snapshots that transition from a specific screenshot."""
        return [
            s for s in self.snapshots if s.screenshot_id == screenshot_id and s.next_screenshot_id
        ]

    def get_transitions_to_screenshot(self, screenshot_id: str) -> list[ActionSnapshot]:
        """Get all snapshots that transition to a specific screenshot."""
        return [s for s in self.snapshots if s.next_screenshot_id == screenshot_id]

    def clear(self) -> None:
        """Clear all snapshots."""
        self.snapshots.clear()
        self.last_updated = datetime.now()

    def size(self) -> int:
        """Get the number of snapshots."""
        return len(self.snapshots)

    def is_empty(self) -> bool:
        """Check if history is empty."""
        return len(self.snapshots) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "snapshots": [s.to_dict() for s in self.snapshots],
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionHistory":
        """Create from dictionary."""
        snapshots = [ActionSnapshot.from_dict(s) for s in data.get("snapshots", [])]
        last_updated = None
        if data.get("last_updated"):
            last_updated = datetime.fromisoformat(data["last_updated"])

        return cls(snapshots=snapshots, last_updated=last_updated)

    def merge(self, other: "ActionHistory") -> None:
        """Merge another history into this one."""
        self.snapshots.extend(other.snapshots)
        self.last_updated = datetime.now()

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the action history."""
        if self.is_empty():
            return {
                "total_snapshots": 0,
                "successful": 0,
                "failed": 0,
                "action_types": {},
                "average_duration": 0,
            }

        successful = len(self.get_successful_snapshots())
        failed = len(self.get_failed_snapshots())

        # Count by action type
        action_types = {}
        for action_type in ActionType:
            count = len(self.get_snapshots_by_action_type(action_type))
            if count > 0:
                action_types[action_type.value] = count

        # Calculate average duration
        durations = [s.duration for s in self.snapshots if s.duration > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "total_snapshots": self.size(),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / self.size() if self.size() > 0 else 0,
            "action_types": action_types,
            "average_duration": avg_duration,
            "unique_states": len({s.state_id for s in self.snapshots}),
            "has_transitions": any(s.next_screenshot_id for s in self.snapshots),
        }

    def __str__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"ActionHistory({stats['total_snapshots']} snapshots, "
            f"{stats['successful']} successful, "
            f"{stats['success_rate']*100:.1f}% success rate)"
        )
