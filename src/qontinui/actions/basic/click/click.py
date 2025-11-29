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
    ) -> None:
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
        import os
        import sys
        import tempfile
        from datetime import datetime

        # Create debug log file
        debug_log_path = os.path.join(tempfile.gettempdir(), "qontinui_click_debug.log")

        def log_debug(msg: str):
            """Write to both stderr and debug file."""
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            log_line = f"[{timestamp}] {msg}\n"
            print(f"[CLICK_DEBUG] {msg}", file=sys.stderr, flush=True)
            try:
                with open(debug_log_path, "a", encoding="utf-8") as f:
                    f.write(log_line)
            except Exception:
                pass

        log_debug(f"execute() called with target type: {type(target).__name__}")

        # Initialize ActionResult with stored ClickOptions
        matches = ActionResult(self.options)
        if target is None:
            log_debug("Target is None, returning False")
            return False

        # Store log function for perform() to use
        self._log_debug = log_debug

        # Convert target to ObjectCollection if needed
        if isinstance(target, ObjectCollection):
            log_debug("Target is ObjectCollection, calling perform()")
            self.perform(matches, target)
        else:
            # Create ObjectCollection from target
            from ...object_collection import ObjectCollectionBuilder

            builder = ObjectCollectionBuilder()
            if isinstance(target, Location):
                log_debug(f"Target is Location: {target}")
                builder.with_locations(target)
            elif isinstance(target, ModelMatch):
                log_debug(f"Target is ModelMatch: {target}")
                builder.with_match_objects_as_regions(target)
            else:
                # Assume it's already compatible
                log_debug("Target is unknown type, assuming compatible")
                pass

            obj_coll = builder.build()
            log_debug("Built ObjectCollection, calling perform()")
            self.perform(matches, obj_coll)

        log_debug(f"execute() returning success={matches.success}")
        log_debug(f"Debug log written to: {debug_log_path}")
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

        # Use stored log function if available (from execute())
        log_func = getattr(self, "_log_debug", lambda msg: None)
        log_func(f"perform() called with {len(object_collections)} object collections")

        # Process all object collections
        for obj_idx, obj_collection in enumerate(object_collections):
            log_func(f"Processing object collection #{obj_idx+1}")
            log_func(f"  - matches: {len(obj_collection.matches)}")
            log_func(f"  - locations: {len(obj_collection.locations)}")
            log_func(f"  - regions: {len(obj_collection.regions)}")
            # Click on any existing matches in the collection (ActionResult objects with match_list)
            for ar_idx, action_result in enumerate(obj_collection.matches):
                log_func(
                    f"  Processing ActionResult #{ar_idx+1} with {len(action_result.matches)} matches"
                )
                # ActionResult contains a match_list with actual Match objects (from find module)
                for fm_idx, find_match in enumerate(action_result.matches):
                    location = find_match.target
                    log_func(f"    Match #{fm_idx+1}: location={location}")
                    if location:  # Only click if match has a target location
                        log_func(f"    Calling _click() at coordinates: {location}")
                        # Use the underlying match_object which is a ModelMatch
                        self._click(location, click_options, find_match.match_object)
                        clicked_count += 1
                        log_func(f"    Click completed, clicked_count={clicked_count}")
                    else:
                        log_func("    Skipping match - no location")
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
                matches.matches.append(find_match)
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
                matches.matches.append(find_match)
                clicked_count += 1
                if self.time and clicked_count < self._get_total_targets(object_collections):
                    self.time.wait(click_options.get_pause_between_individual_actions())

        # Set success based on whether we clicked anything
        object.__setattr__(matches, "success", clicked_count > 0)

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
            # Post-click mouse movement handled by PostClickHandler when configured
            # Currently disabled - enable by passing PostClickHandler instance to Click.__init__
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
        object.__setattr__(result, "action_description", description)
        return result
