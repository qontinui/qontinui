"""Click action - ported from Qontinui framework.

Performs mouse click operations on GUI elements.
"""

from typing import Any, Optional

from ....model.element.location import Location
from ....model.match import Match as ModelMatch
from ...action_interface import ActionInterface
from ...action_result import ActionResult
from ...action_type import ActionType
from ...object_collection import ObjectCollection
from .click_options import ClickOptions, ClickOptionsBuilder


class Click(ActionInterface):
    """Performs ATOMIC mouse click operations on GUI elements.

    Port of Click from Brobot framework class.

    Click is an ATOMIC action that only performs mouse clicks on locations. It does NOT
    search for images or patterns - that functionality is provided by Find or through
    action chaining when using convenience methods like Action.click(StateImage).

    This separation of concerns ensures:
    - Click remains a pure, atomic action
    - Complex behaviors are achieved through action composition
    - Each action has a single, well-defined responsibility

    Click targets supported:
    - Locations: Direct screen coordinates
    - Regions: Clicks at region center
    - Matches: Results from previous Find operations
    - NOT StateImages directly (use Find first or Action.click convenience method)

    Advanced features:
    - Multi-click support for double-clicks, triple-clicks, etc.
    - Configurable click types (left, right, middle button)
    - Batch clicking on multiple locations
    - Post-click mouse movement to avoid hover effects
    - Precise timing control between clicks
    """

    def __init__(
        self,
        options: ClickOptions | None = None,
        click_location_once: Optional["SingleClickExecutor"] = None,
        time: Optional["TimeProvider"] = None,
        after_click: Optional["PostClickHandler"] = None,
    ):
        """Initialize Click action.

        Args:
            options: Click options for configuration (timing, click type, etc.)
            click_location_once: Executor for single clicks
            time: Time provider for delays
            after_click: Handler for post-click actions
        """
        self.options = options or ClickOptionsBuilder().build()
        self.click_location_once = click_location_once
        self.time = time
        self.after_click = after_click

    def get_action_type(self) -> ActionType:
        """Return the action type.

        Returns:
            ActionType.CLICK
        """
        return ActionType.CLICK

    def execute(self, target: Any = None) -> bool:
        """Convenience method to execute a click action.

        Args:
            target: Target to click (Location, Region, Match, or ObjectCollection)

        Returns:
            True if click was successful
        """
        # Initialize ActionResult with stored ClickOptions
        matches = ActionResult(self.options)
        if target is None:
            return False

        # Convert target to ObjectCollection if needed
        if isinstance(target, ObjectCollection):
            self.perform(matches, target)
        else:
            # Create ObjectCollection from target
            from ...object_collection import ObjectCollectionBuilder

            builder = ObjectCollectionBuilder()
            if isinstance(target, Location):
                builder.with_locations(target)
            elif isinstance(target, ModelMatch):
                builder.with_match_objects_as_regions(target)
            else:
                # Assume it's already compatible
                pass

            obj_coll = builder.build()
            self.perform(matches, obj_coll)

        return matches.success

    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Execute click operations on provided locations, regions, or existing matches.

        Click is an ATOMIC action that only performs mouse clicks. It does NOT search
        for images - that's the responsibility of Find or action chaining.

        This method processes different target types:
        - Locations: Clicks directly at the coordinates
        - Regions: Clicks at the region center
        - Matches: Clicks at match locations (from previous Find operations)
        - DOES NOT process StateImages (those require Find first)

        Args:
            matches: The ActionResult containing configuration and any pre-existing matches
            object_collections: Collections containing locations, regions, or matches to click

        Raises:
            ValueError: If matches does not contain ClickOptions configuration
        """
        # Get the configuration - expecting ClickOptions
        action_config = matches.action_config
        if not isinstance(action_config, ClickOptions):
            raise ValueError("Click requires ClickOptions configuration")

        click_options = action_config
        clicked_count = 0

        # Process all object collections
        for obj_collection in object_collections:
            # Click on any existing matches in the collection (ActionResult objects with match_list)
            for action_result in obj_collection.matches:
                # ActionResult contains a match_list with actual Match objects (from find module)
                for find_match in action_result.match_list:
                    location = find_match.target
                    if location:  # Only click if match has a target location
                        # Use the underlying match_object which is a ModelMatch
                        self._click(location, click_options, find_match.match_object)
                        clicked_count += 1
                        # Pause between different targets
                        if self.time and clicked_count < self._get_total_targets(
                            object_collections
                        ):
                            self.time.wait(click_options.get_pause_between_individual_actions())

            # Click on any locations in the collection
            for state_location in obj_collection.state_locations:
                # Extract the Location from StateLocation
                from ....find.match import Match as FindMatch

                # Create a model Match for tracking
                model_match = ModelMatch(target=state_location.location, score=1.0)
                # Wrap in find Match for compatibility with match_list
                find_match = FindMatch(match_object=model_match)
                self._click(state_location.location, click_options, model_match)
                matches.match_list.append(find_match)
                clicked_count += 1
                if self.time and clicked_count < self._get_total_targets(object_collections):
                    self.time.wait(click_options.get_pause_between_individual_actions())

            # Click on any regions in the collection (at their center)
            for state_region in obj_collection.state_regions:
                from ....find.match import Match as FindMatch

                location = state_region.get_center()
                # Create a model Match for tracking
                model_match = ModelMatch(target=location, score=1.0)
                # Wrap in find Match for compatibility with match_list
                find_match = FindMatch(match_object=model_match)
                self._click(location, click_options, model_match)
                matches.match_list.append(find_match)
                clicked_count += 1
                if self.time and clicked_count < self._get_total_targets(object_collections):
                    self.time.wait(click_options.get_pause_between_individual_actions())

        # Set success based on whether we clicked anything
        matches.success = clicked_count > 0

    def _get_total_targets(self, object_collections: tuple[ObjectCollection, ...]) -> int:
        """Count total number of clickable targets across all collections.

        Args:
            object_collections: Collections to count targets in

        Returns:
            Total number of clickable targets
        """
        count = 0
        for obj_collection in object_collections:
            count += len(obj_collection.matches)
            count += len(obj_collection.state_locations)
            count += len(obj_collection.state_regions)
        return count

    def _click(self, location: Location, click_options: ClickOptions, match: ModelMatch) -> None:
        """Perform multiple clicks on a single location with configurable timing and post-click behavior.

        This method handles the low-level click execution for a single match, including:
        - Repeating clicks based on configuration
        - Tracking click count on the match object
        - Managing timing between repeated clicks
        - Delegating post-click mouse movement to AfterClick when configured

        Example: With timesToRepeatIndividualAction=2, this method will:
        1. Click the location
        2. Increment match's acted-on counter
        3. Move mouse if configured
        4. Pause
        5. Click again
        6. Increment counter again
        7. Move mouse if configured (no pause after last click)

        Args:
            location: The screen coordinates where clicks will be performed. This is the
                     final, adjusted location from the match's target.
            click_options: Configuration containing:
                          - timesToRepeatIndividualAction: number of clicks per location
                          - pauseBetweenIndividualActions: delay between clicks (ms)
                          - moveMouseAfterAction: whether to move mouse after clicking
            match: The Match object being acted upon. This object is modified by
                  incrementing its timesActedOn counter for each click.
        """
        times_to_repeat = click_options.get_times_to_repeat_individual_action()
        for i in range(times_to_repeat):
            # SingleClickExecutor now accepts ActionConfig
            if self.click_location_once:
                self.click_location_once.click(location, click_options)
            match.increment_times_acted_on()
            # TODO: Handle mouse movement after action when PostClickHandler is implemented
            if i < times_to_repeat - 1:
                if self.time:
                    self.time.wait(click_options.get_pause_between_individual_actions())


class SingleClickExecutor:
    """Placeholder for SingleClickExecutor class.

    Executes a single click at a location.
    """

    def click(self, location: Location, click_options: ClickOptions) -> None:
        """Execute a single click.

        Args:
            location: Location to click
            click_options: Click configuration
        """
        print(f"Clicking at {location} with options {click_options}")


class PostClickHandler:
    """Placeholder for PostClickHandler class.

    Handles post-click mouse movement.
    """

    pass


class TimeProvider:
    """Placeholder for TimeProvider class.

    Provides time and delay functionality.
    """

    def wait(self, seconds: float) -> None:
        """Wait for specified duration.

        Args:
            seconds: Duration to wait
        """
        import time

        time.sleep(seconds)


class ActionResultFactory:
    """Placeholder for ActionResultFactory class.

    Creates ActionResult instances.
    """

    def init(
        self, action_config, description: str, object_collections: tuple[Any, ...]
    ) -> ActionResult:
        """Initialize an ActionResult.

        Args:
            action_config: Configuration for the action
            description: Description of the action
            object_collections: Object collections

        Returns:
            New ActionResult instance
        """
        result = ActionResult(action_config)
        result.action_description = description
        return result
