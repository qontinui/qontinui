"""State detector - ported from Qontinui framework.

Discovers active states through visual pattern matching in the framework.
"""

import logging
from typing import TYPE_CHECKING, Optional

from ..actions.action import Action
from ..actions.object_collection import ObjectCollection
from ..model.state.special.special_state_type import SpecialStateType
from .state_memory import StateMemory

if TYPE_CHECKING:
    from ..model.state import State

logger = logging.getLogger(__name__)


class StateDetector:
    """Discovers active states through visual pattern matching.

    Port of StateDetector from Qontinui framework class.

    StateDetector provides mechanisms to identify which states are currently active in the GUI
    by searching for their associated visual patterns. This is essential for recovering from
    lost context, initializing automation, or maintaining awareness of the current application
    state during long-running automation sessions.

    Key operations:
    - Check Active States: Verify if currently tracked states are still active
    - Rebuild Active States: Full discovery when context is lost
    - Search All States: Comprehensive scan of all defined states
    - Find Specific State: Check if a particular state is active
    - Refresh States: Complete reset and rediscovery

    Performance considerations:
    - Full state search is computationally expensive (O(n) with n = total images)
    - Checking existing active states is more efficient than full search
    - Future optimization could use machine learning for instant state recognition
    - Static images in states make ML training data generation feasible

    State discovery strategy:
    1. First checks if known active states are still visible
    2. If no active states remain, performs comprehensive search
    3. Falls back to UNKNOWN state if no states are found
    4. Updates StateMemory with discovered active states

    Common use cases:
    - Initializing automation when starting state is unknown
    - Recovering after application crashes or unexpected navigation
    - Periodic state validation in long-running automation
    - Debugging state detection issues

    Future enhancements:
    - Neural network-based instant state recognition from screenshots
    - Probabilistic state detection based on partial matches
    - Hierarchical search starting with likely states
    - Caching and optimization for frequently checked states

    In the model-based approach, StateDetector serves as the sensory system that connects
    the abstract state model to the concrete visual reality of the GUI. It enables the
    framework to maintain situational awareness and recover gracefully from unexpected
    situations, which is crucial for robust automation.
    """

    def __init__(
        self,
        all_states_in_project_service: Optional["StateService"] = None,
        state_memory: StateMemory | None = None,
        action: Action | None = None,
    ):
        """Initialize StateDetector.

        Args:
            all_states_in_project_service: Service for accessing state definitions
            state_memory: Current state memory
            action: Action executor for performing find operations
        """
        self.all_states_in_project_service = all_states_in_project_service
        self.state_memory = state_memory
        self.action = action

    def check_for_active_states(self) -> None:
        """Verify that currently active states are still visible on screen.

        Iterates through all states marked as active in StateMemory and checks
        if they can still be found on the screen. States that are no longer
        visible are removed from the active state list. Uses a copy
        to avoid concurrent modification while removing states.

        Side effects:
        - Removes states from StateMemory that are no longer visible
        - May result in empty active state list if no states remain
        """
        if not self.state_memory:
            return

        current_active_states = set(self.state_memory.active_states)
        logger.debug(f"Checking {len(current_active_states)} active states for visibility")

        states_no_longer_visible = set()
        for state_id in current_active_states:
            if not self.find_state(state_id):
                states_no_longer_visible.add(state_id)

        for state_id in states_no_longer_visible:
            self.state_memory.remove_active_state(state_id)

        if states_no_longer_visible and self.all_states_in_project_service:
            state_names = []
            for state_id in states_no_longer_visible:
                name = self.all_states_in_project_service.get_state_name(state_id)
                if name:
                    state_names.append(name)

            logger.info(
                f"Removed {len(states_no_longer_visible)} states that are no longer visible: "
                f"{', '.join(state_names)}"
            )

    def rebuild_active_states(self) -> None:
        """Rebuild the active state list when context is lost or uncertain.

        First attempts to verify existing active states. If no states remain
        active after verification, performs a comprehensive search of all
        defined states. If still no states are found, falls back to the
        UNKNOWN state to prevent an empty active state list.

        This method implements a three-tier recovery strategy:
        1. Verify known active states (fast)
        2. Search all states if needed (slow)
        3. Default to UNKNOWN if nothing found
        """
        logger.info("Rebuilding active states")
        self.check_for_active_states()

        if self.state_memory and self.state_memory.active_states:
            logger.info(
                f"Active states still present after verification: "
                f"{self.state_memory.get_active_state_names()}"
            )
            return

        logger.info("No active states found, performing comprehensive search")
        self.search_all_images_for_current_states()

        if self.state_memory:
            if not self.state_memory.active_states:
                logger.warning(
                    "No states found after comprehensive search, defaulting to UNKNOWN state"
                )
                self.state_memory.add_active_state(SpecialStateType.UNKNOWN.get_id())
            else:
                logger.info(f"Rebuilt active states: {self.state_memory.get_active_state_names()}")

    def search_all_images_for_current_states(self) -> None:
        """Perform comprehensive search for all defined states on the current screen.

        Searches for every state in the project (except UNKNOWN) to build a
        complete picture of which states are currently active. This is a
        computationally expensive operation that should be used sparingly.

        Side effects:
        - Updates StateMemory with all found states
        - Prints progress dots to Report for monitoring
        - May take significant time with many states/images

        Performance note: O(n*m) where n = number of states, m = images per state
        """
        if not self.all_states_in_project_service:
            return

        logger.info("Starting comprehensive state search")
        all_state_names = self.all_states_in_project_service.get_all_state_names()

        # Remove UNKNOWN state from search
        unknown_name = str(SpecialStateType.UNKNOWN)
        if unknown_name in all_state_names:
            all_state_names.remove(unknown_name)

        total_states = len(all_state_names)
        found = 0

        for state_name in all_state_names:
            if self.find_state_by_name(state_name):
                found += 1

        logger.info(f"Comprehensive search complete: found {found} of {total_states} states")

    def find_state_by_name(self, state_name: str) -> bool:
        """Search for a specific state by name on the current screen.

        Attempts to find visual patterns associated with the named state.
        If found, the state is automatically added to StateMemory's active
        list by the Action framework. Progress is reported for debugging.

        Args:
            state_name: Name of the state to search for

        Returns:
            True if the state was found on screen, False otherwise
        """
        if not self.all_states_in_project_service or not self.action:
            return False

        logger.debug(f"Searching for state: {state_name}")
        state = self.all_states_in_project_service.get_state_by_name(state_name)

        if not state:
            logger.warning(f"State '{state_name}' not found in service")
            return False

        # The state will be automatically activated in StateMemory by ActionExecution
        # when patterns are found
        result = self.action.find(ObjectCollection.Builder().with_non_shared_images(state).build())

        found = result.is_success()

        if found:
            logger.debug(f"State '{state_name}' found and activated")

        return found

    def find_state(self, state_id: int) -> bool:
        """Search for a specific state by ID on the current screen.

        Attempts to find visual patterns associated with the state ID.
        If found, the state is automatically added to StateMemory's active
        list by the Action framework. State name is printed for debugging.

        Args:
            state_id: ID of the state to search for

        Returns:
            True if the state was found on screen, False otherwise
        """
        if not self.all_states_in_project_service or not self.action:
            return False

        state_name = self.all_states_in_project_service.get_state_name(state_id)
        logger.debug(f"Searching for state by ID: {state_id} ({state_name})")

        state = self.all_states_in_project_service.get_state(state_id)

        if not state:
            logger.warning(f"State with ID {state_id} not found in service")
            return False

        # The state will be automatically activated in StateMemory by ActionExecution
        # when patterns are found
        result = self.action.find(ObjectCollection.Builder().with_non_shared_images(state).build())

        found = result.is_success()

        if found:
            logger.debug(f"State '{state_name}' (ID: {state_id}) found and activated")

        return found

    def refresh_active_states(self) -> set[int]:
        """Completely reset and rediscover all active states.

        Clears all existing active state information and performs a fresh
        comprehensive search. This provides a clean slate when the automation
        needs to re-establish its position from scratch.

        Side effects:
        - Clears all states from StateMemory
        - Performs full state search (expensive)
        - Rebuilds active state list from scratch

        Returns:
            Set of state IDs that were found to be active
        """
        logger.info("Refreshing all active states - clearing and rediscovering")

        if self.state_memory:
            self.state_memory.clear_active_states()

        self.search_all_images_for_current_states()

        if self.state_memory:
            active_states = self.state_memory.active_states
            if self.all_states_in_project_service:
                state_names = []
                for state_id in active_states:
                    name = self.all_states_in_project_service.get_state_name(state_id)
                    if name:
                        state_names.append(name)
                logger.info(f"Refresh complete. Active states: {', '.join(state_names)}")

            return active_states

        return set()


class StateService:
    """Placeholder for StateService.

    Will be implemented when migrating the navigation package.
    """

    def get_state(self, state_id: int) -> Optional["State"]:
        """Get state by ID."""
        return None

    def get_state_by_name(self, name: str) -> Optional["State"]:
        """Get state by name."""
        return None

    def get_state_name(self, state_id: int) -> str | None:
        """Get state name by ID."""
        return None

    def get_all_state_names(self) -> set[str]:
        """Get all state names."""
        return set()
