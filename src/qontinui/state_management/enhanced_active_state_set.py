"""Enhanced ActiveStateSet with state group support.

Manages currently active states with support for:
- State groups that activate/deactivate together
- Multi-state activation
- State visibility tracking
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class StateActivation:
    """Record of a state activation."""

    state_id: int
    timestamp: datetime = field(default_factory=datetime.now)
    group: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnhancedActiveStateSet:
    """Manages active states with group support.

    This enhanced version combines:
    - Brobot's active state tracking
    - Multi-state group management
    - Visibility and blocking state handling
    - Activation history for debugging
    """

    # Currently active states
    active_states: set[int] = field(default_factory=set)

    # Hidden states (present but not visible)
    hidden_states: set[int] = field(default_factory=set)

    # Blocking states (must be resolved)
    blocking_states: set[int] = field(default_factory=set)

    # State groups for collective management
    active_groups: dict[str, set[int]] = field(default_factory=dict)

    # Activation history
    activation_history: list[StateActivation] = field(default_factory=list)
    max_history: int = 100

    # Configuration
    allow_multiple_blocking: bool = False  # Allow multiple blocking states
    track_activation_time: bool = True

    def add_state(self, state_id: int, group: str | None = None, blocking: bool = False) -> bool:
        """Add a state to the active set.

        Args:
            state_id: State to activate
            group: Optional group name
            blocking: Whether this is a blocking state

        Returns:
            True if state was added (wasn't already active)
        """
        if state_id in self.active_states:
            logger.debug(f"State {state_id} already active")
            return False

        # Check blocking states
        if blocking and not self.allow_multiple_blocking and self.blocking_states:
            logger.warning(
                f"Cannot add blocking state {state_id}: "
                f"blocking states already exist: {self.blocking_states}"
            )
            return False

        # Add to active states
        self.active_states.add(state_id)

        # Add to blocking if needed
        if blocking:
            self.blocking_states.add(state_id)

        # Remove from hidden if present
        self.hidden_states.discard(state_id)

        # Track activation
        if self.track_activation_time:
            activation = StateActivation(state_id=state_id, group=group)
            self._add_to_history(activation)

        logger.info(f"Activated state {state_id}" + (f" in group {group}" if group else ""))
        return True

    def add_states(self, state_ids: set[int], group: str | None = None) -> set[int]:
        """Add multiple states at once.

        Args:
            state_ids: States to activate
            group: Optional group name for all states

        Returns:
            Set of states that were actually added
        """
        added = set()

        for state_id in state_ids:
            if self.add_state(state_id, group):
                added.add(state_id)

        if group and added:
            self.active_groups[group] = added.copy()

        return added

    def remove_state(self, state_id: int) -> bool:
        """Remove a state from the active set.

        Args:
            state_id: State to deactivate

        Returns:
            True if state was removed
        """
        if state_id not in self.active_states:
            logger.debug(f"State {state_id} not active")
            return False

        # Remove from all sets
        self.active_states.discard(state_id)
        self.hidden_states.discard(state_id)
        self.blocking_states.discard(state_id)

        # Remove from any groups
        for group_name, group_states in list(self.active_groups.items()):
            group_states.discard(state_id)
            if not group_states:
                del self.active_groups[group_name]

        logger.info(f"Deactivated state {state_id}")
        return True

    def remove_states(self, state_ids: set[int]) -> set[int]:
        """Remove multiple states at once.

        Args:
            state_ids: States to deactivate

        Returns:
            Set of states that were actually removed
        """
        removed = set()

        for state_id in state_ids:
            if self.remove_state(state_id):
                removed.add(state_id)

        return removed

    def hide_state(self, state_id: int) -> bool:
        """Hide an active state (remains active but not visible).

        Args:
            state_id: State to hide

        Returns:
            True if state was hidden
        """
        if state_id not in self.active_states:
            logger.warning(f"Cannot hide inactive state {state_id}")
            return False

        if state_id in self.hidden_states:
            logger.debug(f"State {state_id} already hidden")
            return False

        self.hidden_states.add(state_id)
        logger.debug(f"Hidden state {state_id}")
        return True

    def show_state(self, state_id: int) -> bool:
        """Show a hidden state.

        Args:
            state_id: State to show

        Returns:
            True if state was shown
        """
        if state_id not in self.hidden_states:
            logger.debug(f"State {state_id} not hidden")
            return False

        self.hidden_states.discard(state_id)
        logger.debug(f"Shown state {state_id}")
        return True

    def activate_group(self, group_name: str, state_ids: set[int]) -> bool:
        """Activate a group of states together.

        Args:
            group_name: Name of the group
            state_ids: States in the group

        Returns:
            True if group was activated
        """
        logger.info(f"Activating group '{group_name}' with states: {state_ids}")

        # Add all states in the group
        added = self.add_states(state_ids, group_name)

        if added:
            self.active_groups[group_name] = added
            return True

        return False

    def deactivate_group(self, group_name: str) -> bool:
        """Deactivate all states in a group.

        Args:
            group_name: Name of the group

        Returns:
            True if group was deactivated
        """
        if group_name not in self.active_groups:
            logger.warning(f"Unknown group: {group_name}")
            return False

        state_ids = self.active_groups[group_name].copy()
        logger.info(f"Deactivating group '{group_name}' with states: {state_ids}")

        removed = self.remove_states(state_ids)

        if group_name in self.active_groups:
            del self.active_groups[group_name]

        return len(removed) > 0

    def is_active(self, state_id: int) -> bool:
        """Check if a state is active.

        Args:
            state_id: State to check

        Returns:
            True if state is active
        """
        return state_id in self.active_states

    def is_visible(self, state_id: int) -> bool:
        """Check if a state is visible (active and not hidden).

        Args:
            state_id: State to check

        Returns:
            True if state is visible
        """
        return state_id in self.active_states and state_id not in self.hidden_states

    def is_blocking(self, state_id: int) -> bool:
        """Check if a state is blocking.

        Args:
            state_id: State to check

        Returns:
            True if state is blocking
        """
        return state_id in self.blocking_states

    def get_active_states(self) -> set[int]:
        """Get all active states.

        Returns:
            Set of active state IDs
        """
        return self.active_states.copy()

    def get_visible_states(self) -> set[int]:
        """Get all visible states (active and not hidden).

        Returns:
            Set of visible state IDs
        """
        return self.active_states - self.hidden_states

    def get_hidden_states(self) -> set[int]:
        """Get all hidden states.

        Returns:
            Set of hidden state IDs
        """
        return self.hidden_states.copy()

    def get_blocking_states(self) -> set[int]:
        """Get all blocking states.

        Returns:
            Set of blocking state IDs
        """
        return self.blocking_states.copy()

    def get_group_states(self, group_name: str) -> set[int]:
        """Get states in a group.

        Args:
            group_name: Name of the group

        Returns:
            Set of state IDs in the group
        """
        return self.active_groups.get(group_name, set()).copy()

    def get_state_groups(self, state_id: int) -> set[str]:
        """Get all groups containing a state.

        Args:
            state_id: State to look up

        Returns:
            Set of group names containing the state
        """
        groups = set()

        for group_name, group_states in self.active_groups.items():
            if state_id in group_states:
                groups.add(group_name)

        return groups

    def clear(self) -> None:
        """Clear all active states."""
        self.active_states.clear()
        self.hidden_states.clear()
        self.blocking_states.clear()
        self.active_groups.clear()
        logger.info("Cleared all active states")

    def clear_blocking(self) -> None:
        """Clear blocking states (resolve blocks)."""
        self.blocking_states.clear()
        logger.info("Cleared blocking states")

    def _add_to_history(self, activation: StateActivation) -> None:
        """Add activation to history.

        Args:
            activation: Activation record
        """
        self.activation_history.append(activation)

        # Trim history if needed
        if len(self.activation_history) > self.max_history:
            self.activation_history = self.activation_history[-self.max_history :]

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about active states.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_active": len(self.active_states),
            "total_visible": len(self.get_visible_states()),
            "total_hidden": len(self.hidden_states),
            "total_blocking": len(self.blocking_states),
            "total_groups": len(self.active_groups),
            "history_size": len(self.activation_history),
        }

    def __str__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"ActiveStateSet("
            f"active={stats['total_active']}, "
            f"visible={stats['total_visible']}, "
            f"blocking={stats['total_blocking']}, "
            f"groups={stats['total_groups']})"
        )

    def __contains__(self, state_id: int) -> bool:
        """Check if state is active using 'in' operator."""
        return self.is_active(state_id)
