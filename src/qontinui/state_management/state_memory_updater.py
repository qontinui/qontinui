"""State Memory Updater - Automatically updates state memory based on visual matches.

This module bridges the gap between action execution and state management
by automatically updating the StateMemory when images belonging to states are found.
This ensures that the framework maintains an accurate understanding of which states
are currently active based on visual evidence.

Ported from Brobot's StateMemoryUpdater component.
"""

import logging
from typing import Any, cast

from ..actions.action_result import ActionResult
from ..find.match import Match
from .manager import QontinuiStateManager as StateManager
from .state_memory import StateMemory

logger = logging.getLogger(__name__)


class StateMemoryUpdater:
    """Updates StateMemory based on matches found during action execution.

    This component automatically activates states when their associated images
    are found during action execution, maintaining accurate state tracking
    without manual intervention.

    Attributes:
        state_memory: The StateMemory instance to update
        state_manager: The StateManager for accessing state information
        auto_activation_enabled: Whether automatic state activation is enabled
        activation_confidence_threshold: Minimum confidence for state activation
    """

    def __init__(
        self,
        state_memory: StateMemory,
        state_manager: StateManager | None = None,
        auto_activation_enabled: bool = True,
        activation_confidence_threshold: float = 0.9,
    ) -> None:
        """Initialize StateMemoryUpdater.

        Args:
            state_memory: The StateMemory instance to update
            state_manager: The StateManager for accessing states (optional)
            auto_activation_enabled: Whether to enable automatic state activation
            activation_confidence_threshold: Minimum match confidence for activation
        """
        self.state_memory = state_memory
        # Create a default StateManager if none provided
        self.state_manager = state_manager if state_manager is not None else StateManager()
        self.auto_activation_enabled = auto_activation_enabled
        self.activation_confidence_threshold = activation_confidence_threshold

        # Track activation history for debugging
        self._activation_history: list[dict[str, Any]] = []

        logger.info(
            "StateMemoryUpdater initialized",
            extra={
                "auto_activation": auto_activation_enabled,
                "confidence_threshold": activation_confidence_threshold,
            },
        )

    def update_from_action_result(self, action_result: ActionResult) -> set[str]:
        """Update StateMemory based on matches found in an ActionResult.

        When an image is found, the state it belongs to is set as active.

        Args:
            action_result: The result containing matches to process

        Returns:
            Set of state names that were activated
        """
        if not self.auto_activation_enabled:
            return set()

        if action_result is None or not action_result.matches:
            return set()

        activated_states = set()

        for match in action_result.matches:
            state_name = self._activate_from_match(match)
            if state_name:
                activated_states.add(state_name)

        if activated_states:
            logger.info(
                f"Updated active states based on {len(action_result.matches)} matches",
                extra={
                    "activated_states": list(activated_states),
                    "total_matches": len(action_result.matches),
                    "active_states": self.state_memory.get_active_state_names(),
                },
            )

        return activated_states

    def update_from_match(self, match: Match) -> str | None:
        """Update StateMemory from a single match.

        Args:
            match: The match to process

        Returns:
            Name of the activated state, or None if no state was activated
        """
        if not self.auto_activation_enabled:
            return None

        return self._activate_from_match(match)

    def _activate_from_match(self, match: Match) -> str | None:
        """Activate state based on a match.

        Args:
            match: The match to process

        Returns:
            Name of the activated state, or None if no state was activated
        """
        if match is None:
            return None

        # Check confidence threshold
        if match.confidence < self.activation_confidence_threshold:
            logger.debug(
                f"Match confidence {match.confidence:.2f} below threshold "
                f"{self.activation_confidence_threshold:.2f}, skipping activation"
            )
            return None

        # Get state name from match
        state_name = self._get_state_from_match(match)
        if not state_name:
            return None

        # Check if state exists
        state = self.state_manager.get_state(state_name)
        if not state:
            logger.warning(f"State '{state_name}' not found in StateManager")
            return None

        # Activate state if not already active
        if not self.state_memory.is_state_active(state_name):
            logger.debug(
                f"Activating state '{state_name}' based on match",
                extra={
                    "confidence": match.confidence,
                    "location": (match.x, match.y) if match else None,
                },
            )

            # Get state ID for activation
            if hasattr(state, "id") and state.id is not None:
                self.state_memory.add_active_state(state.id)
            else:
                logger.warning(f"State '{state_name}' has no valid ID")

            # Record activation in history
            self._record_activation(state_name, match)

            return state_name

        return None

    def _get_state_from_match(self, match: Match) -> str | None:
        """Extract state name from a match.

        Args:
            match: The match to extract state from

        Returns:
            State name if found, None otherwise
        """
        # Check if match has state object data
        if hasattr(match, "state_object_data") and match.state_object_data:
            state_obj_data = match.state_object_data
            if hasattr(state_obj_data, "owner_state_name"):
                owner_name = getattr(state_obj_data, "owner_state_name", None)
                return cast(str | None, owner_name)

        # Check if match has state_name directly
        if hasattr(match, "state_name"):
            return cast(str | None, match.state_name)

        # Check if match has metadata with state info
        if hasattr(match, "metadata") and isinstance(match.metadata, dict):
            return match.metadata.get("state_name") or match.metadata.get("owner_state")

        return None

    def _record_activation(self, state_name: str, match: Match) -> None:
        """Record state activation in history.

        Args:
            state_name: Name of activated state
            match: The match that triggered activation
        """
        import time

        activation_record = {
            "timestamp": time.time(),
            "state_name": state_name,
            "match_confidence": match.confidence if match else None,
            "match_location": ((match.x, match.y) if match and hasattr(match, "x") else None),
        }

        self._activation_history.append(activation_record)

        # Limit history size
        max_history = 100
        if len(self._activation_history) > max_history:
            self._activation_history = self._activation_history[-max_history:]

    def set_auto_activation(self, enabled: bool) -> None:
        """Enable or disable automatic state activation.

        Args:
            enabled: Whether to enable auto-activation
        """
        self.auto_activation_enabled = enabled
        logger.info(f"Auto-activation {'enabled' if enabled else 'disabled'}")

    def set_confidence_threshold(self, threshold: float) -> None:
        """Set minimum confidence threshold for state activation.

        Args:
            threshold: Confidence threshold (0.0 to 1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")

        self.activation_confidence_threshold = threshold
        logger.info(f"Activation confidence threshold set to {threshold:.2f}")

    def get_activation_history(self) -> list[dict[str, Any]]:
        """Get the history of state activations.

        Returns:
            List of activation records
        """
        return self._activation_history.copy()

    def clear_activation_history(self) -> None:
        """Clear the activation history."""
        self._activation_history.clear()
        logger.debug("Activation history cleared")

    def deactivate_states_without_matches(
        self, action_result: ActionResult, states_to_check: set[str] | None = None
    ) -> set[str]:
        """Deactivate states that don't have matches in the action result.

        This is useful for maintaining accurate state tracking when certain
        states should only be active if their visual elements are present.

        Args:
            action_result: The action result to check
            states_to_check: Specific states to check (None = check all active)

        Returns:
            Set of state names that were deactivated
        """
        if not self.auto_activation_enabled:
            return set()

        # Get states to check
        if states_to_check is None:
            states_to_check = set(self.state_memory.get_active_state_names())

        # Get states that have matches
        states_with_matches = set()
        if action_result and action_result.matches:
            for match in action_result.matches:
                state_name = self._get_state_from_match(match)
                if state_name:
                    states_with_matches.add(state_name)

        # Deactivate states without matches
        deactivated_states = set()
        for state_name in states_to_check:
            if state_name not in states_with_matches and self.state_memory.is_state_active(
                state_name
            ):
                logger.debug(f"Deactivating state '{state_name}' - no matches found")
                # Get state ID for deactivation
                state = self.state_manager.get_state(state_name)
                if state and hasattr(state, "id") and state.id is not None:
                    self.state_memory.remove_active_state(state.id)
                    deactivated_states.add(state_name)
                else:
                    logger.warning(f"Could not deactivate state '{state_name}' - no valid ID")

        if deactivated_states:
            logger.info(
                f"Deactivated {len(deactivated_states)} states without matches",
                extra={
                    "deactivated_states": list(deactivated_states),
                    "remaining_active": self.state_memory.get_active_state_names(),
                },
            )

        return deactivated_states


# Global instance for convenience
_state_memory_updater: StateMemoryUpdater | None = None


def get_state_memory_updater() -> StateMemoryUpdater:
    """Get or create global StateMemoryUpdater instance.

    Returns:
        StateMemoryUpdater instance
    """
    global _state_memory_updater
    if _state_memory_updater is None:
        from .manager import QontinuiStateManager as StateManager
        from .state_memory import get_state_memory

        # Create a default StateManager instance
        state_manager_instance = StateManager()
        _state_memory_updater = StateMemoryUpdater(
            state_memory=get_state_memory(), state_manager=state_manager_instance
        )

    return _state_memory_updater
