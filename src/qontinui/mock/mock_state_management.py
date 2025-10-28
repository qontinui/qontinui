"""MockStateManagement - Manages state probabilities for mock mode.

Based on Brobot's MockStateManagement, controls which states are "found" during mock execution.
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StateConfig:
    """Configuration for a mocked state."""

    name: str
    probability: int = 100  # 0-100 probability of being found
    active: bool = False
    elements: dict[str, int] = field(default_factory=dict)  # Element probabilities


class MockStateManagement:
    """Manages state probabilities and visibility in mock mode.

    This class controls which states and their elements are "found" during
    mock execution by managing probability settings.
    """

    def __init__(self) -> None:
        """Initialize MockStateManagement."""
        self._state_configs: dict[str, StateConfig] = {}
        self._active_states: set[str] = set()
        self._default_probability: int = 100

        logger.debug("MockStateManagement initialized")

    def set_state_probabilities(self, probability: int, *state_names: str) -> None:
        """Set the probability for one or more states to be found.

        Args:
            probability: Probability (0-100) of states being found
            *state_names: Names of states to configure
        """
        probability = max(0, min(100, probability))  # Clamp to 0-100

        for state_name in state_names:
            if state_name not in self._state_configs:
                self._state_configs[state_name] = StateConfig(name=state_name)

            self._state_configs[state_name].probability = probability

            logger.debug(f"Set probability for state '{state_name}' to {probability}%")

            # Update active states based on probability
            if probability > 0:
                self._active_states.add(state_name)
            else:
                self._active_states.discard(state_name)

    def get_state_probability(self, state_name: str) -> int:
        """Get the probability for a state.

        Args:
            state_name: Name of the state

        Returns:
            Probability (0-100) of the state being found
        """
        if state_name in self._state_configs:
            return self._state_configs[state_name].probability
        return self._default_probability

    def set_element_probability(self, state_name: str, element_name: str, probability: int) -> None:
        """Set the probability for a specific element within a state.

        Args:
            state_name: Name of the state
            element_name: Name of the element
            probability: Probability (0-100) of element being found
        """
        probability = max(0, min(100, probability))

        if state_name not in self._state_configs:
            self._state_configs[state_name] = StateConfig(name=state_name)

        self._state_configs[state_name].elements[element_name] = probability

        logger.debug(
            f"Set probability for element '{element_name}' in state '{state_name}' to {probability}%"
        )

    def get_element_probability(self, state_name: str, element_name: str) -> int:
        """Get the probability for an element within a state.

        Args:
            state_name: Name of the state
            element_name: Name of the element

        Returns:
            Probability (0-100) of the element being found
        """
        if state_name in self._state_configs:
            config = self._state_configs[state_name]
            if element_name in config.elements:
                return config.elements[element_name]
            # Fall back to state probability
            return config.probability
        return self._default_probability

    def activate_state(self, state_name: str) -> None:
        """Mark a state as active (visible).

        Args:
            state_name: Name of the state to activate
        """
        self._active_states.add(state_name)

        if state_name not in self._state_configs:
            self._state_configs[state_name] = StateConfig(name=state_name)

        self._state_configs[state_name].active = True

        logger.info(f"Activated state: {state_name}")

    def deactivate_state(self, state_name: str) -> None:
        """Mark a state as inactive (not visible).

        Args:
            state_name: Name of the state to deactivate
        """
        self._active_states.discard(state_name)

        if state_name in self._state_configs:
            self._state_configs[state_name].active = False

        logger.info(f"Deactivated state: {state_name}")

    def is_state_active(self, state_name: str) -> bool:
        """Check if a state is currently active.

        Args:
            state_name: Name of the state

        Returns:
            True if the state is active
        """
        return state_name in self._active_states

    def get_active_states(self) -> list[str]:
        """Get list of currently active states.

        Returns:
            List of active state names
        """
        return list(self._active_states)

    def simulate_transition(self, from_state: str, to_state: str) -> None:
        """Simulate a state transition.

        Args:
            from_state: State transitioning from
            to_state: State transitioning to
        """
        # Deactivate source state
        self.deactivate_state(from_state)
        self.set_state_probabilities(0, from_state)

        # Activate target state
        self.activate_state(to_state)
        self.set_state_probabilities(100, to_state)

        logger.info(f"Simulated transition: {from_state} -> {to_state}")

    def reset(self) -> None:
        """Reset all state configurations."""
        self._state_configs.clear()
        self._active_states.clear()
        logger.debug("MockStateManagement reset")

    def set_default_probability(self, probability: int) -> None:
        """Set the default probability for unconfigured states.

        Args:
            probability: Default probability (0-100)
        """
        self._default_probability = max(0, min(100, probability))
        logger.debug(f"Set default probability to {self._default_probability}%")

    def configure_initial_states(self, initial_states: dict[str, int]) -> None:
        """Configure initial state probabilities.

        Args:
            initial_states: Dict mapping state names to probabilities
        """
        for state_name, probability in initial_states.items():
            self.set_state_probabilities(probability, state_name)
            if probability > 0:
                self.activate_state(state_name)

        logger.info(f"Configured {len(initial_states)} initial states")

    def log_state(self) -> None:
        """Log current state configuration for debugging."""
        logger.info("=" * 50)
        logger.info("Mock State Configuration:")
        logger.info(f"  Active states: {self.get_active_states()}")
        logger.info(f"  Default probability: {self._default_probability}%")

        for state_name, config in self._state_configs.items():
            logger.info(f"  {state_name}:")
            logger.info(f"    Probability: {config.probability}%")
            logger.info(f"    Active: {config.active}")
            if config.elements:
                logger.info(f"    Elements: {config.elements}")
        logger.info("=" * 50)


# Global instance for convenience
_mock_state_management: MockStateManagement | None = None


def get_mock_state_management() -> MockStateManagement:
    """Get or create global MockStateManagement instance.

    Returns:
        MockStateManagement instance
    """
    global _mock_state_management
    if _mock_state_management is None:
        _mock_state_management = MockStateManagement()
    return _mock_state_management
