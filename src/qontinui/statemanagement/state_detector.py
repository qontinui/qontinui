"""StateDetector - ported from Qontinui framework.

Visual pattern-based state discovery system.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, cast

if TYPE_CHECKING:
    from ..actions.basic.find.find import Find
    from ..model.match.match import Match

from ..actions.action_result import ActionResult
from ..actions.object_collection import ObjectCollection
from ..model.state.state import State
from ..model.state.state_service import StateService
from ..model.state.state_store import StateStore
from .state_memory import StateMemory

logger = logging.getLogger(__name__)


@dataclass
class StateDetector:
    """Visual pattern-based state discovery.

    Port of StateDetector from Qontinui framework class.

    StateDetector is responsible for discovering which states are currently
    active by searching for their visual patterns on screen. It maintains
    the StateMemory's active state set based on what is visually present.
    """

    # Dependencies
    state_memory: StateMemory
    state_service: StateService
    state_store: StateStore
    find_action: Optional["Find"] = None  # Find action for visual search

    # Configuration
    max_search_time: float = 30.0  # Maximum time to search for states
    search_all_on_failure: bool = True  # Search all states if expected not found
    min_similarity_for_state: float = 0.7  # Minimum similarity to confirm state

    # Statistics
    total_searches: int = 0
    successful_searches: int = 0

    def check_for_active_states(self, state_ids: set[int] | None = None) -> bool:
        """Verify current active states are still visible.

        Checks if the states marked as active in StateMemory are still
        visible on screen. If not provided, checks all active states.

        Args:
            state_ids: Specific states to check, or None for all active

        Returns:
            True if all checked states are still visible
        """
        if state_ids is None:
            state_ids = self.state_memory.active_states.copy()

        if not state_ids:
            logger.debug("No states to check")
            return True

        logger.debug(f"Checking visibility of states: {state_ids}")
        self.total_searches += 1

        all_visible = True
        states_to_remove = set()

        for state_id in state_ids:
            state = self.state_store.get(state_id)
            if not state:
                logger.warning(f"State {state_id} not found in store")
                states_to_remove.add(state_id)
                continue

            # Search for state's visual patterns
            if self._is_state_visible(state):
                logger.debug(f"State {state.name} ({state_id}) confirmed visible")
            else:
                logger.debug(f"State {state.name} ({state_id}) no longer visible")
                states_to_remove.add(state_id)
                all_visible = False

        # Remove states that are no longer visible
        for state_id in states_to_remove:
            self.state_memory.remove_inactive_state(state_id)

        if all_visible:
            self.successful_searches += 1

        return all_visible

    def rebuild_active_states(self) -> set[int]:
        """Complete discovery when context is lost.

        Searches for all known states to rebuild the active state set
        from scratch. Used when the automation has lost track of where
        it is in the application.

        Returns:
            Set of discovered active state IDs
        """
        logger.info("Rebuilding active states from scratch")
        self.state_memory.clear_active_states()

        # Get all states from store
        all_states = self.state_store.get_all()
        found_states = set()

        for state in all_states:
            if self._is_state_visible(state) and state.id is not None:
                found_states.add(state.id)
                self.state_memory.add_active_state(state.id, state)
                logger.info(f"Found active state: {state.name} ({state.id})")

        logger.info(f"Rebuild complete. Found {len(found_states)} active states")
        return found_states

    def search_all_images_for_current_states(self) -> dict[int, list["Match"]]:
        """Comprehensive state search across all images.

        Searches for all state images of all states and returns
        matches organized by state ID.

        Returns:
            Dictionary mapping state IDs to their matches
        """
        logger.debug("Performing comprehensive state search")
        results = {}

        all_states = self.state_store.get_all()

        for state in all_states:
            if not state.state_images or state.id is None:
                continue

            matches = self._search_state_images(state)
            if matches:
                results[state.id] = matches
                # Update state memory if matches found
                if state.id not in self.state_memory.active_states:
                    self.state_memory.add_active_state(state.id, state)

        return results

    def find_state(self, state_identifier: Any) -> State | None:
        """Search for a specific state by name or ID.

        Args:
            state_identifier: State name (str) or ID (int)

        Returns:
            State if found and visible, None otherwise
        """
        # Resolve state
        if isinstance(state_identifier, str):
            state = self.state_store.get(state_identifier)
        elif isinstance(state_identifier, int):
            state = self.state_store.get(state_identifier)
        else:
            logger.error(f"Invalid state identifier type: {type(state_identifier)}")
            return None

        if not state:
            logger.warning(f"State {state_identifier} not found in store")
            return None

        # Check if visible
        if self._is_state_visible(state):
            if state.id is not None:
                self.state_memory.add_active_state(state.id, state)
            return cast(State | None, state)

        return None

    def refresh_active_states(self) -> set[int]:
        """Complete reset and rediscovery.

        Clears all active states and performs fresh discovery.

        Returns:
            Set of newly discovered state IDs
        """
        logger.info("Refreshing all active states")
        self.state_memory.remove_all_states()
        return self.rebuild_active_states()

    def find_expected_states(
        self, expected_state_ids: set[int], timeout: float | None = None
    ) -> bool:
        """Wait for expected states to appear.

        Args:
            expected_state_ids: States expected to appear
            timeout: Maximum wait time (uses default if None)

        Returns:
            True if all expected states found
        """
        if not expected_state_ids:
            return True

        timeout = timeout or self.max_search_time
        logger.info(f"Waiting for states {expected_state_ids} (timeout: {timeout}s)")

        # Use TimeWrapper for mock/real agnostic waiting
        from ..wrappers import get_controller

        controller = get_controller()

        start_time = controller.time.timestamp()

        while controller.time.timestamp() - start_time < timeout:
            found_all = True

            for state_id in expected_state_ids:
                if state_id not in self.state_memory.active_states:
                    state = self.state_store.get(state_id)
                    if state and self._is_state_visible(state):
                        self.state_memory.add_active_state(state_id, state)
                    else:
                        found_all = False

            if found_all:
                logger.info("All expected states found")
                return True

            controller.time.wait(0.5)  # Check twice per second (instant in mock mode)

        logger.warning(f"Timeout waiting for states {expected_state_ids}")
        return False

    def _is_state_visible(self, state: State) -> bool:
        """Check if a state is currently visible.

        Args:
            state: State to check

        Returns:
            True if state is visible
        """
        if not state.state_images:
            # State has no visual patterns, check text
            if state.state_text:
                return self._check_state_text(state)
            # No way to verify, assume not visible
            return False

        # Search for state images
        matches = self._search_state_images(state)
        return len(matches) > 0

    def _search_state_images(self, state: State) -> list["Match"]:
        """Search for a state's visual patterns.

        Args:
            state: State whose images to search for

        Returns:
            List of matches found
        """
        if not self.find_action:
            logger.warning("Find action not available for state detection")
            return []

        # Create object collection from state images
        collection = ObjectCollection()
        for state_image in state.state_images:
            collection.state_images.append(state_image)

        # Perform find operation
        try:
            result = ActionResult()
            self.find_action.perform(result, collection)
            if result and not result.is_empty():
                return cast(list[Match], result.get_match_list())
        except Exception as e:
            logger.error(f"Error searching for state {state.name}: {e}")

        return []

    def _check_state_text(self, state: State) -> bool:
        """Check if state's text clues are present.

        Args:
            state: State to check

        Returns:
            True if text clues found
        """
        if not state.state_text:
            return False

        # This would integrate with OCR/text finding
        # For now, return False as placeholder
        logger.debug(f"Text checking not yet implemented for state {state.name}")
        return False

    def get_success_rate(self) -> float:
        """Get search success rate.

        Returns:
            Success rate (0.0-1.0)
        """
        if self.total_searches == 0:
            return 0.0
        return self.successful_searches / self.total_searches

    def __repr__(self) -> str:
        """String representation.

        Returns:
            Description of detector state
        """
        active_count = self.state_memory.get_active_state_count()
        success_rate = self.get_success_rate()
        return f"StateDetector(active={active_count}, success_rate={success_rate:.2%})"
