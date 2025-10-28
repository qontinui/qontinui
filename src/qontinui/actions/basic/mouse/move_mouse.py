"""Move mouse action - ported from Qontinui framework.

Moves the mouse to one or more locations.
"""

from typing import Optional

from ....model.element.location import Location
from ...action_interface import ActionInterface
from ...action_result import ActionResult
from ...action_type import ActionType
from ...object_collection import ObjectCollection
from ..find.find import Find


class MoveMouse(ActionInterface):
    """Moves the mouse to one or more locations in the Qontinui model-based GUI automation framework.

    Port of MoveMouse from Qontinui framework class.

    MoveMouse is a fundamental action in the Action Model (α) that provides precise cursor
    control for GUI automation. It bridges the gap between visual element identification and
    physical mouse positioning, enabling complex interaction patterns including hover effects,
    drag operations, and tooltip activation.

    Movement patterns supported:
    - Single Target: Direct movement to a specific location
    - Multiple Targets: Sequential movement through multiple locations
    - Pattern-based: Movement to visually identified elements
    - Collection-based: Processing multiple ObjectCollections in sequence

    Processing order:
    - Within an ObjectCollection: Points are visited as recorded by Find operations
      (Images → Matches → Regions → Locations)
    - Between ObjectCollections: Processed in the order they appear in the parameters

    Key features:
    - Visual Integration: Uses Find to locate targets before movement
    - Batch Processing: Can move through multiple locations in one action
    - Timing Control: Configurable pauses between movements
    - State Awareness: Updates framework's understanding of cursor position

    Common use cases:
    - Hovering over elements to trigger tooltips or dropdown menus
    - Positioning cursor for subsequent click or drag operations
    - Following paths through UI elements for gesture-based interactions
    - Moving mouse away from active areas to prevent interference

    In the model-based approach, MoveMouse actions contribute to the framework's spatial
    understanding of the GUI. By tracking cursor movements, the framework can model hover
    states, anticipate UI reactions, and optimize subsequent actions based on current
    cursor position.
    """

    def __init__(
        self,
        find: Find | None = None,
        move_mouse_wrapper: Optional["MoveMouseWrapper"] = None,
        time: Optional["TimeProvider"] = None,
    ) -> None:
        """Initialize MoveMouse action.

        Args:
            find: Find action for locating targets
            move_mouse_wrapper: Wrapper for actual mouse movement
            time: Time provider for delays
        """
        self.find = find
        self.move_mouse_wrapper = move_mouse_wrapper
        self.time = time

    def get_action_type(self) -> ActionType:
        """Return the action type.

        Returns:
            ActionType.MOVE
        """
        return ActionType.MOVE

    def perform(self, matches: ActionResult, *object_collections: ObjectCollection) -> None:
        """Move mouse to locations specified in object collections.

        Args:
            matches: The ActionResult containing configuration and to populate with results
            object_collections: Collections containing targets to move to
        """
        # Get the configuration - MouseMoveOptions or any ActionConfig is acceptable
        # since MoveMouse mainly uses Find and basic timing
        config = matches.get_action_config()

        for obj_coll in object_collections:
            # Check if we have locations directly - no need to find anything
            state_locations = obj_coll.state_locations
            if state_locations:
                # Move directly to the locations without finding
                for state_location in state_locations:
                    location = state_location.location
                    if self.move_mouse_wrapper:
                        self.move_mouse_wrapper.move(location)
                    matches.add_match_location(location)
                    # Create a Match object for success determination
                    from ....find.match import Match
                    from ....model.match import Match as MatchObject

                    match_obj = MatchObject(target=location, score=1.0)
                    match = Match(match_object=match_obj)
                    matches.add(match)

            # Check if we have regions
            state_regions = obj_coll.state_regions
            if state_regions:
                # Move to center of regions without finding
                for state_region in state_regions:
                    location = state_region.get_search_region().get_center()
                    if self.move_mouse_wrapper:
                        self.move_mouse_wrapper.move(location)
                    matches.add_match_location(location)
                    # Create a Match object for success determination
                    from ....find.match import Match
                    from ....model.match import Match as MatchObject

                    match_obj = MatchObject(target=location, score=1.0)
                    match = Match(match_object=match_obj)
                    matches.add(match)

            # Only use find if we have images/patterns to search for
            state_images = obj_coll.state_images
            if state_images:
                if self.find:
                    self.find.perform(matches, obj_coll)
                    for location in matches.get_match_locations():
                        if self.move_mouse_wrapper:
                            self.move_mouse_wrapper.move(location)
                # Find.perform will populate the matchList and success will be determined by ActionSuccessCriteria

            print("finished move. ")

            # Pause between collections
            if self.time and config:
                pause = config.get_pause_after_end()
                if pause > 0:
                    self.time.wait(pause)


class MoveMouseWrapper:
    """Placeholder for MoveMouseWrapper class.

    Wraps the actual mouse movement implementation.
    """

    def move(self, location: Location) -> None:
        """Move mouse to location.

        Args:
            location: Target location
        """
        print(f"Moving mouse to {location}")


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
