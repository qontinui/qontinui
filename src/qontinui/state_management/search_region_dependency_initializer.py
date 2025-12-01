"""Search region dependency initializer - ported from Qontinui framework.

Initializes search region dependencies when states are loaded.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..model.state.state import State
    from ..model.state.state_object import StateObject

logger = logging.getLogger(__name__)


class SearchRegionDependencyInitializer:
    """Initializes search region dependencies when states are loaded.

    Port of SearchRegionDependencyInitializer from Qontinui framework class.

    This component listens for state initialization events and registers
    all cross-state search region dependencies with the DynamicRegionResolver.
    """

    def __init__(
        self,
        state_store: StateStore | None = None,
        dynamic_region_resolver: DynamicRegionResolver | None = None,
    ) -> None:
        """Initialize SearchRegionDependencyInitializer.

        Args:
            state_store: Store containing all states
            dynamic_region_resolver: Resolver for dynamic regions
        """
        self.state_store = state_store
        self.dynamic_region_resolver = dynamic_region_resolver
        logger.info("SearchRegionDependencyInitializer constructor called")

    def on_states_registered(self, state_count: int) -> None:
        """Initialize search region dependencies when all states have been registered.

        This listens for the StatesRegisteredEvent to ensure all states are available.

        Args:
            state_count: Number of states registered
        """
        logger.info(
            f"SearchRegionDependencyInitializer: Received StatesRegisteredEvent with {state_count} states"
        )
        self._initialize_dependencies()

    def _initialize_dependencies(self) -> None:
        """Initialize search region dependencies.

        This method collects all state objects and registers their dependencies.
        """
        logger.info(
            "SearchRegionDependencyInitializer: Starting dependency registration"
        )

        if not self.state_store or not self.dynamic_region_resolver:
            logger.warning("Cannot initialize dependencies: missing store or resolver")
            return

        if TYPE_CHECKING:
            pass

        all_objects: list[Any] = (
            []
        )  # Will contain StateImages, StateLocations, and StateRegions

        # Collect all state objects from all states
        for state in self.state_store.get_all_states():
            logger.debug(f"Processing state: {state.__class__.__name__}")
            all_objects.extend(state.state_images)
            all_objects.extend(state.state_locations)
            all_objects.extend(state.state_regions)

        logger.info(
            f"Found {len(all_objects)} state objects to process for dependencies"
        )

        # Register dependencies with the resolver
        self.dynamic_region_resolver.register_dependencies(all_objects)

        logger.info("SearchRegionDependencyInitializer: Completed registration")

    def refresh_dependencies(self) -> None:
        """Re-register dependencies when states are reloaded or updated.

        This can be called manually or triggered by state reload events.
        """
        self._initialize_dependencies()


class StateStore:
    """Placeholder for StateStore.

    Will be implemented when migrating the persistence package.
    """

    def get_all_states(self) -> list[State]:
        """Get all states.

        Returns:
            List of all states
        """
        return []


class DynamicRegionResolver:
    """Placeholder for DynamicRegionResolver.

    Will be implemented when migrating the action internal package.
    """

    def register_dependencies(self, state_objects: list[StateObject]) -> None:
        """Register dependencies.

        Args:
            state_objects: List of state objects
        """
        pass
