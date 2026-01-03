"""Initial states management - ported from Qontinui framework.

Manages probabilistic initial state discovery for automation startup and recovery.
"""

import logging
import random
from typing import Optional

from ..config.framework_settings import FrameworkSettings
from ..model.state.state import State
from .state_detector import StateDetector
from .state_memory import StateMemory

logger = logging.getLogger(__name__)


class InitialStates:
    """Manages probabilistic initial state discovery.

    Port of InitialStates from Qontinui framework class.

    InitialStates implements a sophisticated probabilistic approach to establishing the
    starting position in the GUI state space. Rather than assuming a fixed starting point,
    it maintains a weighted set of possible initial state configurations and uses intelligent
    search strategies to determine which states are actually active when automation begins.

    Key features:
    - Probabilistic State Sets: Associates probability weights with potential state combinations
    - Intelligent Search: Searches only for likely states, avoiding costly full scans
    - Mock Support: Simulates initial state selection for testing without GUI interaction
    - Fallback Strategy: Searches all states if no predefined sets are found
    - Recovery Capability: Can re-establish position when automation gets lost

    State selection process:
    1. Define potential initial state sets with probability weights
    2. In mock mode: Randomly select based on probability distribution
    3. In normal mode: Search for states in order of likelihood
    4. If no states found: Fall back to searching all known states
    5. Update StateMemory with discovered active states

    Probability management:
    - Each state set has an integer probability weight (not percentage)
    - Weights are cumulative for random selection algorithm
    - Higher weights indicate more likely initial configurations
    - Sum can exceed 100 as these are relative weights, not percentages

    Use cases:
    - Starting automation from unknown GUI position
    - Handling multiple possible application entry points
    - Recovering from navigation failures or unexpected states
    - Testing automation logic with simulated state configurations
    - Supporting applications with variable startup sequences

    Example usage:
        # Define possible initial states with probabilities
        initial_states.add_state_set(70, "LoginPage")  # 70% chance
        initial_states.add_state_set(20, "Dashboard")  # 20% chance
        initial_states.add_state_set(10, "MainMenu")   # 10% chance

        # Find which states are actually active
        await initial_states.find_initial_states()

    In the model-based approach, InitialStates embodies the framework's adaptability to
    uncertain starting conditions. Unlike rigid scripts that assume fixed entry points,
    this component enables automation that can begin from various GUI positions and
    intelligently determine its location before proceeding with tasks.

    This probabilistic approach is essential for:
    - Web applications with session-based navigation
    - Desktop applications with persistent state
    - Mobile apps with deep linking or notifications
    - Any GUI where the starting position varies
    """

    def __init__(
        self,
        state_finder: StateDetector | None = None,
        state_memory: StateMemory | None = None,
        all_states_in_project_service: Optional["StateService"] = None,
    ) -> None:
        """Initialize InitialStates.

        Args:
            state_finder: Detector for finding states
            state_memory: Current state memory
            all_states_in_project_service: Service for accessing state definitions
        """
        self.state_finder = state_finder
        self.state_memory = state_memory
        self.all_states_in_project_service = all_states_in_project_service

        self.sum_of_probabilities = 0
        """Running total of all probability weights for normalization."""

        self.potential_active_states: dict[frozenset[int], int] = {}
        """Maps potential state sets to their cumulative probability thresholds.

        Each entry maps a set of state IDs to an integer representing the upper
        bound of its probability range. When selecting randomly, a value between
        1 and sum_of_probabilities is chosen, and the first entry with a threshold
        greater than or equal to this value is selected.

        Example: If three sets have weights 50, 30, 20:
        - Set 1: threshold = 50 (range 1-50)
        - Set 2: threshold = 80 (range 51-80)
        - Set 3: threshold = 100 (range 81-100)
        """

    def add_state_set_from_states(self, probability: int, *states: State) -> None:
        """Add a potential initial state set with its probability weight.

        Registers a combination of states that might be active when automation
        begins, along with a weight indicating how likely this combination is.
        Higher weights make the state set more likely to be selected in mock
        mode or searched first in normal mode.

        Args:
            probability: Weight for this state set (must be positive)
            states: Variable number of State objects forming the set
        """
        if probability <= 0:
            return

        self.sum_of_probabilities += probability
        state_ids = frozenset(state.id for state in states if state.id is not None)
        self.potential_active_states[state_ids] = self.sum_of_probabilities

    def add_state_set(self, probability: int, *state_names: str) -> None:
        """Add a potential initial state set using state names.

        Convenience method that accepts state names instead of State objects.
        Names are resolved to states through StateService. Invalid names are
        silently ignored, allowing flexible configuration.

        Args:
            probability: Weight for this state set (must be positive)
            state_names: Variable number of state names forming the set
        """
        if probability <= 0:
            return

        if not self.all_states_in_project_service:
            return

        self.sum_of_probabilities += probability
        state_ids = set()

        for name in state_names:
            state = self.all_states_in_project_service.get_state_by_name(name)
            if state:
                state_ids.add(state.id)

        if state_ids:
            # Filter out None values before creating frozenset
            valid_ids = {id for id in state_ids if id is not None}
            if valid_ids:
                self.potential_active_states[frozenset(valid_ids)] = self.sum_of_probabilities

    async def find_initial_states(self) -> None:
        """Discover and activate the initial states for automation.

        Main entry point that determines which states are currently active.
        Behavior depends on FrameworkSettings.mock:
        - Mock mode: Randomly selects from defined state sets
        - Normal mode: Searches screen for actual states

        Side effects:
        - Updates StateMemory with discovered active states
        - Resets state probabilities to base values
        - Prints results to Report for debugging
        """
        logger.info("Finding initial states")

        if FrameworkSettings.get_instance().mock:
            self._mock_initial_states()
        else:
            await self._search_for_initial_states()

    def _mock_initial_states(self) -> None:
        """Simulate initial state selection for testing without GUI interaction.

        Uses weighted random selection to choose a state set from the defined
        potential states. Selected states are activated in StateMemory without
        actual screen verification, enabling unit testing and development.

        Selection algorithm:
        1. Generate random number between 1 and sum_of_probabilities
        2. Find first state set with threshold >= random value
        3. Activate all states in the selected set
        4. Reset their probabilities to base values
        """
        if not self.potential_active_states:
            logger.warning("No potential active states defined")
            return

        if not self.state_memory or not self.all_states_in_project_service:
            return

        # Generate a random number between 1 and sum_of_probabilities
        random_value = random.randint(1, self.sum_of_probabilities)
        logger.debug(f"Randomly selected value: {random_value} out of {self.sum_of_probabilities}")

        # Find the state set whose probability range contains the random value
        for state_ids, threshold in self.potential_active_states.items():
            if random_value <= threshold:
                # Activate the selected states
                logger.info(f"Selected {len(state_ids)} initial states")

                for state_id in state_ids:
                    self.state_memory.add_active_state(state_id)
                    state = self.all_states_in_project_service.get_state(state_id)
                    if state:
                        state.set_probability_to_base_probability()
                        logger.debug(f"Activated state: {state.name} (ID: {state_id})")

                active_names = self.state_memory.get_active_state_names()
                logger.info(f"Initial states are: {', '.join(active_names)}")
                return

        # This should never happen if potential_active_states is properly populated
        logger.error("Failed to select any initial states")

    async def _search_for_initial_states(self) -> None:
        """Search for initial states on the actual screen.

        Attempts to find active states by searching for predefined potential
        state sets. If no states are found, falls back to searching for all
        known states individually. This two-phase approach balances efficiency
        with completeness.
        """
        await self._search_for_states(self.potential_active_states)

        if self.state_memory and not self.state_memory.active_states:
            # Fall back to searching all states
            if self.all_states_in_project_service:
                potential = {}
                for state_id in self.all_states_in_project_service.get_all_state_ids():
                    potential[frozenset([state_id])] = 1
                await self._search_for_states(potential)

    async def _search_for_states(
        self, potential_active_states_and_probabilities: dict[frozenset[int], int]
    ) -> None:
        """Execute state searches for all states in the provided sets.

        Collects all unique states from the potential state sets and uses
        StateFinder to search for each one on the screen. Found states are
        automatically added to StateMemory by StateFinder.

        Args:
            potential_active_states_and_probabilities: Map of state sets to search
        """
        if not self.state_finder:
            return

        all_potential_states: set[int] = set()
        for state_ids in potential_active_states_and_probabilities.keys():
            all_potential_states.update(state_ids)

        for state_id in all_potential_states:
            await self.state_finder.find_state(state_id)


class StateService:
    """Placeholder for StateService.

    Will be implemented when migrating the navigation package.
    """

    def get_state(self, state_id: int) -> State | None:
        """Get state by ID."""
        return None

    def get_state_by_name(self, name: str) -> State | None:
        """Get state by name."""
        return None

    def get_all_state_ids(self) -> list[int]:
        """Get all state IDs."""
        return []
