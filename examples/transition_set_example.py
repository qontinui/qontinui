"""Example demonstrating the new transition_set pattern with from_transition and to_transition.

This example shows how to use the new Brobot-style transition decorators in Qontinui.
"""

import logging

from qontinui.actions import Action
from qontinui.annotations import from_transition, state, to_transition, transition_set

logger = logging.getLogger(__name__)


# Define states using the existing @state decorator
@state(name="Home", initial=True)
class HomeState:
    """The home/landing state of the application."""

    def __init__(self):
        self.world_button = "world_button.png"
        self.search_button = "search_button.png"
        self.unique_logo = "home_logo.png"


@state(name="World")
class WorldState:
    """The world/map view state."""

    def __init__(self):
        self.back_button = "back_button.png"
        self.island_button = "island_button.png"
        self.world_map = "world_map.png"


@state(name="Island")
class IslandState:
    """The island detail state."""

    def __init__(self):
        self.home_button = "home_button.png"
        self.world_button = "world_nav_button.png"
        self.island_view = "island_view.png"


# Define transitions using the new @transition_set pattern
@transition_set(state=HomeState, description="All transitions to Home state")
class HomeTransitions:
    """Contains all transitions that lead TO the Home state."""

    def __init__(self):
        self.home_state = HomeState()
        self.world_state = WorldState()
        self.island_state = IslandState()
        self.action = Action()

    @from_transition(
        from_state=WorldState, priority=1, description="Navigate from World to Home"
    )
    def from_world(self) -> bool:
        """Navigate from World state back to Home."""
        logger.info("Navigating from World to Home")
        # Click the back button in World state
        return self.action.click(self.world_state.back_button).is_success()

    @from_transition(
        from_state=IslandState, priority=1, description="Navigate from Island to Home"
    )
    def from_island(self) -> bool:
        """Navigate from Island state back to Home."""
        logger.info("Navigating from Island to Home")
        # Click the home button in Island state
        return self.action.click(self.island_state.home_button).is_success()

    @to_transition(description="Verify arrival at Home state", required=True)
    def verify_arrival(self) -> bool:
        """Verify that we have successfully arrived at the Home state."""
        logger.info("Verifying arrival at Home state")

        # Check for presence of home-specific elements
        found_world_button = self.action.find(self.home_state.world_button).is_success()
        found_search_button = self.action.find(
            self.home_state.search_button
        ).is_success()

        if found_world_button or found_search_button:
            logger.info("Successfully confirmed Home state is active")
            return True
        else:
            logger.error("Failed to confirm Home state - home elements not found")
            return False


@transition_set(state=WorldState, description="All transitions to World state")
class WorldTransitions:
    """Contains all transitions that lead TO the World state."""

    def __init__(self):
        self.home_state = HomeState()
        self.world_state = WorldState()
        self.island_state = IslandState()
        self.action = Action()

    @from_transition(
        from_state=HomeState, priority=1, description="Navigate from Home to World"
    )
    def from_home(self) -> bool:
        """Navigate from Home state to World."""
        logger.info("Navigating from Home to World")
        # Click the world button in Home state
        return self.action.click(self.home_state.world_button).is_success()

    @from_transition(
        from_state=IslandState, priority=2, description="Navigate from Island to World"
    )
    def from_island(self) -> bool:
        """Navigate from Island state to World."""
        logger.info("Navigating from Island to World")
        # Click the world navigation button in Island state
        return self.action.click(self.island_state.world_button).is_success()

    @to_transition(description="Verify arrival at World state", timeout=10)
    def verify_arrival(self) -> bool:
        """Verify that we have successfully arrived at the World state."""
        logger.info("Verifying arrival at World state")

        # Check for the world map
        if self.action.find(self.world_state.world_map).is_success():
            logger.info("Successfully confirmed World state is active")
            return True
        else:
            logger.warning("Failed to confirm World state - world map not found")
            return False


@transition_set(state=IslandState, description="All transitions to Island state")
class IslandTransitions:
    """Contains all transitions that lead TO the Island state."""

    def __init__(self):
        self.world_state = WorldState()
        self.island_state = IslandState()
        self.action = Action()

    @from_transition(
        from_state=WorldState, priority=1, description="Navigate from World to Island"
    )
    def from_world(self) -> bool:
        """Navigate from World state to Island."""
        logger.info("Navigating from World to Island")
        # Click on an island in the world map
        return self.action.click(self.world_state.island_button).is_success()

    @to_transition(description="Verify arrival at Island state", required=True)
    def verify_arrival(self) -> bool:
        """Verify that we have successfully arrived at the Island state."""
        logger.info("Verifying arrival at Island state")

        # Check for the island view
        if self.action.find(self.island_state.island_view).is_success():
            logger.info("Successfully confirmed Island state is active")
            return True
        else:
            logger.error("Failed to confirm Island state - island view not found")
            return False


if __name__ == "__main__":
    """
    This example demonstrates:

    1. The @transition_set decorator groups all transitions for a target state
    2. @from_transition methods define how to navigate FROM other states
    3. @to_transition method verifies arrival at the target state
    4. The to_transition is automatically executed after any from_transition

    Key benefits of this pattern:
    - Cleaner separation of navigation logic and verification
    - All transitions to a state are grouped in one class
    - Automatic execution of arrival verification after navigation
    - Support for optional vs required verification
    - Priority-based transition selection when multiple paths exist
    """

    print("Transition set example loaded successfully!")
    print(
        "This example shows how to use the Brobot-style transition decorators in Qontinui."
    )
    print("\nKey concepts:")
    print("- @transition_set: Groups all transitions for a target state")
    print("- @from_transition: Defines navigation FROM a source state")
    print("- @to_transition: Verifies arrival at the target state")
    print("\nThe to_transition is automatically executed after any from_transition.")
    print(
        "This ensures consistent state verification regardless of the navigation path."
    )
