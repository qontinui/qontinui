"""State tracker for the scheduling system.

Tracks current active states and state transitions.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from qontinui_schemas.common import utc_now

logger = logging.getLogger(__name__)


@dataclass
class StateTransitionEvent:
    """Record of a state transition."""

    from_states: set[str]
    to_states: set[str]
    timestamp: datetime
    success: bool
    duration_ms: float | None = None


class StateTracker:
    """Tracks application state for the scheduler.

    Following Brobot principles:
    - Monitors active states
    - Records state transitions
    - Provides state history
    - Notifies scheduler of state changes
    """

    def __init__(self) -> None:
        """Initialize the state tracker."""
        self._active_states: set[str] = set()
        self._state_history: list[StateTransitionEvent] = []
        self._state_listeners: list[Callable[[set[str]], None]] = []

        # State statistics
        self._state_entry_count: dict[str, int] = {}
        self._state_duration: dict[str, float] = {}
        self._state_entry_time: dict[str, datetime] = {}

        logger.info("StateTracker initialized")

    def update_states(self, active_states: set[str]) -> bool:
        """Update the current active states.

        Args:
            active_states: New set of active states

        Returns:
            True if states changed
        """
        if active_states == self._active_states:
            return False

        # Record transition
        old_states = self._active_states.copy()
        self._active_states = active_states.copy()

        # Update statistics
        self._update_statistics(old_states, active_states)

        # Record event
        event = StateTransitionEvent(
            from_states=old_states,
            to_states=active_states,
            timestamp=utc_now(),
            success=True,
        )
        self._state_history.append(event)

        # Notify listeners
        self._notify_listeners(active_states)

        logger.info(f"States updated: {old_states} -> {active_states}")
        return True

    def activate_state(self, state_name: str) -> bool:
        """Activate a single state.

        Args:
            state_name: State to activate

        Returns:
            True if state was activated
        """
        if state_name in self._active_states:
            return False

        new_states = self._active_states.copy()
        new_states.add(state_name)
        return self.update_states(new_states)

    def deactivate_state(self, state_name: str) -> bool:
        """Deactivate a single state.

        Args:
            state_name: State to deactivate

        Returns:
            True if state was deactivated
        """
        if state_name not in self._active_states:
            return False

        new_states = self._active_states.copy()
        new_states.remove(state_name)
        return self.update_states(new_states)

    def get_active_states(self) -> set[str]:
        """Get current active states.

        Returns:
            Copy of active states set
        """
        return self._active_states.copy()

    def is_state_active(self, state_name: str) -> bool:
        """Check if a state is active.

        Args:
            state_name: State to check

        Returns:
            True if state is active
        """
        return state_name in self._active_states

    def add_listener(self, listener: Callable[[set[str]], None]):
        """Add a state change listener.

        Args:
            listener: Function to call when states change
        """
        self._state_listeners.append(listener)

    def remove_listener(self, listener: Callable[[set[str]], None]):
        """Remove a state change listener.

        Args:
            listener: Listener to remove
        """
        if listener in self._state_listeners:
            self._state_listeners.remove(listener)

    def _notify_listeners(self, active_states: set[str]):
        """Notify all listeners of state change.

        Args:
            active_states: New active states
        """
        for listener in self._state_listeners:
            try:
                listener(active_states)
            except Exception as e:
                logger.error(f"Error notifying state listener: {e}")

    def _update_statistics(self, old_states: set[str], new_states: set[str]):
        """Update state statistics.

        Args:
            old_states: Previous active states
            new_states: New active states
        """
        now = utc_now()

        # Update duration for exited states
        exited_states = old_states - new_states
        for state in exited_states:
            if state in self._state_entry_time:
                duration = (now - self._state_entry_time[state]).total_seconds()
                self._state_duration[state] = self._state_duration.get(state, 0) + duration
                del self._state_entry_time[state]

        # Record entry for new states
        entered_states = new_states - old_states
        for state in entered_states:
            self._state_entry_count[state] = self._state_entry_count.get(state, 0) + 1
            self._state_entry_time[state] = now

    def get_state_statistics(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all states.

        Returns:
            Dictionary of state statistics
        """
        stats = {}
        now = utc_now()

        for state in set(self._state_entry_count.keys()) | set(self._state_duration.keys()):
            state_stats = {
                "entry_count": self._state_entry_count.get(state, 0),
                "total_duration": self._state_duration.get(state, 0),
                "currently_active": state in self._active_states,
            }

            # Add current session duration if active
            if state in self._state_entry_time:
                current_duration = (now - self._state_entry_time[state]).total_seconds()
                state_stats["current_duration"] = current_duration
                state_stats["total_duration"] += current_duration

            stats[state] = state_stats

        return stats

    def get_history(self, limit: int | None = None) -> list[StateTransitionEvent]:
        """Get state transition history.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of transition events
        """
        if limit:
            return self._state_history[-limit:]
        return self._state_history.copy()

    def clear_history(self):
        """Clear state transition history."""
        self._state_history.clear()
        logger.info("State history cleared")
