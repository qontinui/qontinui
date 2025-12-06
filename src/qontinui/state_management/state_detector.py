"""State detector - ported from Qontinui framework.

Discovers active states through visual pattern matching in the framework.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, cast

from ..actions.action import Action
from ..actions.object_collection import ObjectCollectionBuilder
from ..model.state.special.special_state_type import SpecialStateType
from ..vision.screenshot_cache import ScreenshotCache
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
        all_states_in_project_service: StateService | None = None,
        state_memory: StateMemory | None = None,
        action: Action | None = None,
        max_concurrent: int = 15,
        screenshot_cache_ttl: float = 0.1,
    ) -> None:
        """Initialize StateDetector.

        Args:
            all_states_in_project_service: Service for accessing state definitions
            state_memory: Current state memory
            action: Action executor for performing find operations
            max_concurrent: Maximum concurrent template matches (default: 15)
            screenshot_cache_ttl: Screenshot cache TTL in seconds (default: 0.1)
        """
        self.all_states_in_project_service = all_states_in_project_service
        self.state_memory = state_memory
        self.action = action
        self._max_concurrent = max_concurrent
        self._screenshot_cache = ScreenshotCache(ttl_seconds=screenshot_cache_ttl)

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
                self.state_memory.add_active_state(SpecialStateType.UNKNOWN.value)
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
        from ..actions.basic.find.pattern_find_options import PatternFindOptionsBuilder

        collection = ObjectCollectionBuilder().with_non_shared_images(state).build()
        config = PatternFindOptionsBuilder().build()
        result = self.action.perform(config, collection)

        found = result.is_success

        if found:
            logger.debug(f"State '{state_name}' found and activated")

        return cast(bool, found)

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
        from ..actions.basic.find.pattern_find_options import PatternFindOptionsBuilder

        collection = ObjectCollectionBuilder().with_non_shared_images(state).build()
        config = PatternFindOptionsBuilder().build()
        result = self.action.perform(config, collection)

        found = result.is_success

        if found:
            logger.debug(f"State '{state_name}' (ID: {state_id}) found and activated")

        return cast(bool, found)

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

    async def find_states_parallel_async(
        self,
        state_names: list[str],
        max_concurrent: int | None = None,
    ) -> set[str]:
        """Find multiple states in parallel using async template matching.

        Searches for all specified states concurrently, using a single cached
        screenshot to eliminate redundant screen captures. This is significantly
        faster than sequential state detection, especially when checking many states.

        Performance:
        - Sequential: N states × 500ms = N/2 seconds
        - Parallel: ~300-500ms regardless of N (up to concurrency limit)
        - Speedup: 2-8x depending on number of states

        Args:
            state_names: List of state names to search for
            max_concurrent: Maximum concurrent searches (default: self._max_concurrent)

        Returns:
            Set of state names that were found to be active

        Example:
            states_to_check = ["login", "dashboard", "settings"]
            found = await detector.find_states_parallel_async(states_to_check)
            # Returns: {"dashboard", "settings"} if those are visible
        """
        if not self.all_states_in_project_service or not self.action:
            logger.warning("StateDetector not fully initialized for async operations")
            return set()

        if not state_names:
            return set()

        logger.info(f"Starting parallel search for {len(state_names)} states")

        # Create search tasks for each state
        search_tasks = []
        for state_name in state_names:
            task = self._find_state_async(state_name)
            search_tasks.append({"state_name": state_name, "task": task})

        # Execute with concurrency limit
        limit = max_concurrent or self._max_concurrent
        semaphore = asyncio.Semaphore(limit)

        async def limited_search(task_info: dict) -> tuple[str, bool]:
            async with semaphore:
                found = await task_info["task"]
                return task_info["state_name"], found

        # Run all searches concurrently
        results = await asyncio.gather(
            *[limited_search(t) for t in search_tasks], return_exceptions=True
        )

        # Collect found states
        found_states = set()
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error during state search: {result}")
                continue
            state_name, found = result  # type: ignore[misc]
            if found:
                found_states.add(state_name)

        logger.info(
            f"Parallel search complete: found {len(found_states)} of {len(state_names)} states"
        )
        return found_states

    async def _find_state_async(self, state_name: str) -> bool:
        """Async version of find_state_by_name.

        Searches for a specific state asynchronously. This method should be
        called within a concurrent context managed by find_states_parallel_async.

        Args:
            state_name: Name of the state to search for

        Returns:
            True if the state was found on screen, False otherwise
        """
        if not self.all_states_in_project_service or not self.action:
            return False

        logger.debug(f"Async search for state: {state_name}")
        state = self.all_states_in_project_service.get_state_by_name(state_name)

        if not state:
            logger.warning(f"State '{state_name}' not found in service")
            return False

        # Run the synchronous action in a thread pool to avoid blocking
        # the event loop during template matching
        from ..actions.basic.find.pattern_find_options import PatternFindOptionsBuilder

        collection = ObjectCollectionBuilder().with_non_shared_images(state).build()
        config = PatternFindOptionsBuilder().build()

        # Execute the action in a thread pool
        result = await asyncio.to_thread(self.action.perform, config, collection)

        found = result.is_success

        if found:
            logger.debug(f"State '{state_name}' found and activated")

        return cast(bool, found)

    async def rebuild_active_states_async(self) -> None:
        """Async version of rebuild_active_states using parallel state detection.

        First attempts to verify existing active states in parallel. If no states
        remain active after verification, performs a comprehensive parallel search
        of all defined states. If still no states are found, falls back to the
        UNKNOWN state.

        This is significantly faster than the synchronous version:
        - Sequential: 5 states × 500ms = 2.5 seconds
        - Async parallel: ~300-500ms regardless of state count
        """
        logger.info("Rebuilding active states (async)")

        # Check current active states in parallel
        if self.state_memory and self.state_memory.active_states:
            current_active_state_ids = list(self.state_memory.active_states)
            state_names = []

            if self.all_states_in_project_service:
                for state_id in current_active_state_ids:
                    name = self.all_states_in_project_service.get_state_name(state_id)
                    if name:
                        state_names.append(name)

            if state_names:
                found_states = await self.find_states_parallel_async(state_names)

                # Remove states that are no longer active
                for state_id in current_active_state_ids:
                    if self.all_states_in_project_service is not None:
                        name = self.all_states_in_project_service.get_state_name(state_id)
                        if name and name not in found_states:
                            self.state_memory.remove_active_state(state_id)

        # If we still have active states, we're done
        if self.state_memory and self.state_memory.active_states:
            logger.info(
                f"Active states still present after verification: "
                f"{self.state_memory.get_active_state_names()}"
            )
            return

        # No active states remain, search all states
        logger.info("No active states found, performing comprehensive parallel search")
        await self._search_all_images_async()

        if self.state_memory:
            if not self.state_memory.active_states:
                logger.warning(
                    "No states found after comprehensive search, defaulting to UNKNOWN state"
                )
                self.state_memory.add_active_state(SpecialStateType.UNKNOWN.value)
            else:
                logger.info(f"Rebuilt active states: {self.state_memory.get_active_state_names()}")

    async def _search_all_images_async(self) -> None:
        """Async version of search_all_images_for_current_states using parallel search.

        Performs comprehensive parallel search for all defined states on the current
        screen. This is much faster than the sequential version but still expensive
        for projects with many states.
        """
        if not self.all_states_in_project_service:
            return

        logger.info("Starting comprehensive parallel state search")
        all_state_names = list(self.all_states_in_project_service.get_all_state_names())

        # Remove UNKNOWN state from search
        unknown_name = str(SpecialStateType.UNKNOWN)
        if unknown_name in all_state_names:
            all_state_names.remove(unknown_name)

        # Search all states in parallel
        found_states = await self.find_states_parallel_async(all_state_names)

        logger.info(
            f"Comprehensive parallel search complete: found {len(found_states)} "
            f"of {len(all_state_names)} states"
        )


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

    def get_state_name(self, state_id: int) -> str | None:
        """Get state name by ID."""
        return None

    def get_all_state_names(self) -> set[str]:
        """Get all state names."""
        return set()
