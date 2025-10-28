"""StateLifecycleManager - Manages state activation and lifecycle.

Tracks active states, current state, and state status.
"""

import logging
import threading
from enum import Enum, auto

logger = logging.getLogger(__name__)


class StateStatus(Enum):
    """Status of a state in the store."""

    ACTIVE = auto()  # Currently active/loaded
    INACTIVE = auto()  # Not currently active
    TRANSITIONING = auto()  # In process of transition
    ERROR = auto()  # State has errors
    UNKNOWN = auto()  # Status unknown


class StateLifecycleManager:
    """Manages state activation, deactivation, and status tracking.

    Single responsibility: Track which states are active and manage their lifecycle.
    """

    def __init__(self) -> None:
        """Initialize the lifecycle manager."""
        self._active_states: set[str] = set()
        self._current_state: str | None = None
        self._status: dict[str, StateStatus] = {}
        self._lock = threading.RLock()

    def register_state(self, name: str) -> None:
        """Register a new state with initial inactive status.

        Args:
            name: State name to register
        """
        with self._lock:
            self._status[name] = StateStatus.INACTIVE

    def unregister_state(self, name: str) -> None:
        """Unregister a state and clean up its tracking.

        Args:
            name: State name to unregister
        """
        with self._lock:
            # Remove from active states
            self._active_states.discard(name)

            # Clear current state if it's this one
            if self._current_state == name:
                self._current_state = None

            # Remove status
            if name in self._status:
                del self._status[name]

    def activate(self, name: str) -> bool:
        """Activate a state.

        Args:
            name: Name of state to activate

        Returns:
            True if activated successfully
        """
        with self._lock:
            if name not in self._status:
                logger.error(f"Cannot activate unknown state '{name}'")
                return False

            self._active_states.add(name)
            self._status[name] = StateStatus.ACTIVE
            logger.debug(f"Activated state '{name}'")
            return True

    def deactivate(self, name: str) -> bool:
        """Deactivate a state.

        Args:
            name: Name of state to deactivate

        Returns:
            True if deactivated successfully
        """
        with self._lock:
            if name not in self._status:
                return False

            self._active_states.discard(name)

            # Don't change status if it's the current state
            if self._current_state != name:
                self._status[name] = StateStatus.INACTIVE

            logger.debug(f"Deactivated state '{name}'")
            return True

    def set_current_state(self, name: str) -> bool:
        """Set the current state.

        Args:
            name: Name of state to make current

        Returns:
            True if set successfully
        """
        with self._lock:
            if name not in self._status:
                logger.error(f"Cannot set current state: '{name}' not registered")
                return False

            # Deactivate previous current state
            if self._current_state:
                self._status[self._current_state] = StateStatus.INACTIVE
                self._active_states.discard(self._current_state)

            # Activate new current state
            self._current_state = name
            self._active_states.add(name)
            self._status[name] = StateStatus.ACTIVE

            logger.info(f"Current state set to '{name}'")
            return True

    def get_current_state(self) -> str | None:
        """Get the current state name.

        Returns:
            Current state name or None
        """
        with self._lock:
            return self._current_state

    def get_active_states(self) -> set[str]:
        """Get all active state names.

        Returns:
            Set of active state names
        """
        with self._lock:
            return self._active_states.copy()

    def get_status(self, name: str) -> StateStatus:
        """Get status of a state.

        Args:
            name: State name

        Returns:
            State status
        """
        with self._lock:
            return self._status.get(name, StateStatus.UNKNOWN)

    def set_status(self, name: str, status: StateStatus) -> bool:
        """Set status of a state.

        Args:
            name: State name
            status: New status

        Returns:
            True if set successfully
        """
        with self._lock:
            if name not in self._status:
                return False
            self._status[name] = status
            return True

    def is_active(self, name: str) -> bool:
        """Check if a state is active.

        Args:
            name: State name

        Returns:
            True if state is active
        """
        with self._lock:
            return name in self._active_states

    def active_count(self) -> int:
        """Get the number of active states.

        Returns:
            Number of active states
        """
        with self._lock:
            return len(self._active_states)

    def clear(self) -> None:
        """Clear all lifecycle tracking."""
        with self._lock:
            self._active_states.clear()
            self._current_state = None
            self._status.clear()
            logger.debug("Lifecycle manager cleared")
