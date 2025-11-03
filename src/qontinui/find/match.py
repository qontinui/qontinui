"""Match class - ported from Qontinui framework.

Wrapper around MatchObject with additional functionality.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from ..actions import ActionResult
from ..model.element import Image, Location, Region
from ..model.match import Match as MatchObject


@dataclass
class Match:
    """Enhanced match result with action capabilities.

    Port of Match from Qontinui framework class.
    Extends MatchObject with action methods.
    """

    match_object: MatchObject

    # Exposed properties from match_object
    location: Location | None = field(init=False)
    region: Region | None = field(init=False)
    similarity: float = field(init=False)
    pattern: Image | None = field(init=False)
    state_object_data: Any = field(init=False, default=None)

    def __post_init__(self):
        """Initialize match properties."""
        # Expose key properties directly
        self.location = self.match_object.target  # Match uses 'target' not 'location'
        self.region = self.match_object.get_region()
        self.similarity = self.match_object.score
        self.pattern = self.match_object.search_image
        # state_object_data is stored in metadata, not as direct attribute
        self.state_object_data = getattr(self.match_object.metadata, "state_object_data", None)

    @property
    def center(self) -> Location:
        """Get center location of match."""
        return self.match_object.center

    @property
    def target(self) -> Location:
        """Get target location for actions."""
        target = self.match_object.target
        if target is None:
            return self.center  # Fallback to center if target is None
        return target

    @property
    def score(self) -> float:
        """Get similarity score."""
        return self.match_object.score

    @property
    def confidence(self) -> float:
        """Get confidence score (alias for score/similarity)."""
        return self.match_object.score

    @property
    def x(self) -> int:
        """Get x coordinate of match target."""
        return int(self.target.x)

    @property
    def y(self) -> int:
        """Get y coordinate of match target."""
        return int(self.target.y)

    def get_region(self) -> Region | None:
        """Get the region of this match.

        Returns:
            Region of the match or None
        """
        return self.region

    def get_target(self) -> Location:
        """Get the target location for actions.

        Returns:
            Target location
        """
        return self.target

    def exists(self) -> bool:
        """Check if match exists (was found).

        Returns:
            True if match exists
        """
        if self.match_object is None:
            return False
        return self.similarity > 0

    def click(self) -> ActionResult:
        """Click on this match.

        Returns:
            ActionResult from click action
        """
        if not self.exists():
            result = ActionResult()
            result.success = False
            result.output_text = "Match does not exist"
            return result

        # Use Action system to click at the target location
        from ..actions import Action
        from ..model.element import Location

        action = Action()
        return action.click(Location(x=self.target.x, y=self.target.y))

    def double_click(self) -> ActionResult:
        """Double-click on this match.

        Returns:
            ActionResult from double-click action
        """
        if not self.exists():
            result = ActionResult()
            result.success = False
            result.output_text = "Match does not exist"
            return result

        from ..actions import Action
        from ..actions.basic.click.click_options import ClickOptionsBuilder
        from ..actions.object_collection import ObjectCollectionBuilder
        from ..model.element import Location

        location = Location(x=self.target.x, y=self.target.y)
        collection = ObjectCollectionBuilder().with_locations(location).build()
        click_options = ClickOptionsBuilder().set_number_of_clicks(2).build()
        action = Action()
        return action.perform(click_options, collection)

    def right_click(self) -> ActionResult:
        """Right-click on this match.

        Returns:
            ActionResult from right-click action
        """
        if not self.exists():
            result = ActionResult()
            result.success = False
            result.output_text = "Match does not exist"
            return result

        from ..actions import Action
        from ..actions.basic.click.click_options import ClickOptionsBuilder
        from ..actions.basic.mouse.mouse_press_options import MouseButton, MousePressOptions
        from ..actions.object_collection import ObjectCollectionBuilder
        from ..model.element import Location

        location = Location(x=self.target.x, y=self.target.y)
        collection = ObjectCollectionBuilder().with_locations(location).build()
        mouse_press = MousePressOptions.builder().set_button(MouseButton.RIGHT).build()
        click_options = ClickOptionsBuilder().set_press_options(mouse_press).build()
        action = Action()
        return action.perform(click_options, collection)

    def hover(self) -> ActionResult:
        """Move mouse to this match.

        Returns:
            ActionResult from move action
        """
        if not self.exists():
            result = ActionResult()
            result.success = False
            result.output_text = "Match does not exist"
            return result

        from ..actions import Action
        from ..actions.basic.mouse.mouse_move_options import MouseMoveOptionsBuilder
        from ..actions.object_collection import ObjectCollectionBuilder
        from ..model.element import Location

        location = Location(x=self.target.x, y=self.target.y)
        collection = ObjectCollectionBuilder().with_locations(location).build()
        move_options = MouseMoveOptionsBuilder().build()
        action = Action()
        return action.perform(move_options, collection)

    def drag_to(self, target: "Match") -> ActionResult:
        """Drag from this match to another.

        Args:
            target: Target match to drag to

        Returns:
            ActionResult from drag action
        """
        if not self.exists():
            result = ActionResult()
            result.success = False
            result.output_text = "Source match does not exist"
            return result
        if not target.exists():
            result = ActionResult()
            result.success = False
            result.output_text = "Target match does not exist"
            return result

        from ..actions import Action
        from ..actions.composite.drag import DragOptionsBuilder
        from ..actions.object_collection import ObjectCollectionBuilder
        from ..model.element import Location

        source_loc = Location(x=self.target.x, y=self.target.y)
        target_loc = Location(x=target.target.x, y=target.target.y)
        source_collection = ObjectCollectionBuilder().with_locations(source_loc).build()
        target_collection = ObjectCollectionBuilder().with_locations(target_loc).build()
        drag_options = DragOptionsBuilder().build()
        action = Action()
        return action.perform(drag_options, source_collection, target_collection)

    def type_text(self, text: str) -> ActionResult:
        """Type text at this match location.

        Args:
            text: Text to type

        Returns:
            ActionResult from type action
        """
        if not self.exists():
            result = ActionResult()
            result.success = False
            result.output_text = "Match does not exist"
            return result

        # Click first to focus
        click_result = self.click()
        if not click_result.success:
            return click_result

        from ..actions import Action

        action = Action()
        return action.type_text(text)

    def highlight(self, duration: float = 2.0) -> ActionResult:
        """Highlight this match on screen.

        Args:
            duration: How long to show highlight

        Returns:
            ActionResult from highlight action
        """
        if not self.exists():
            result = ActionResult()
            result.success = False
            result.output_text = "Match does not exist"
            return result

        # This would draw a rectangle around the match
        # Implementation depends on platform
        result = ActionResult()
        result.success = True
        result.defined_regions = [self.region] if self.region else []
        result.duration = timedelta(seconds=duration)
        return result

    def get_text(self) -> str:
        """Extract text from this match region using OCR.

        Returns:
            Extracted text or empty string
        """
        if not self.exists():
            return ""

        # This would use OCR on the match region
        # For now, return placeholder
        return f"Text from {self.region}"

    def distance_to(self, other: "Match") -> float:
        """Calculate distance to another match.

        Args:
            other: Other match

        Returns:
            Distance in pixels or float('inf') if either doesn't exist
        """
        if not self.exists() or not other.exists():
            return float("inf")

        return self.center.distance_to(other.center)

    def is_left_of(self, other: "Match") -> bool:
        """Check if this match is to the left of another.

        Args:
            other: Other match

        Returns:
            True if this match is to the left
        """
        if not self.exists() or not other.exists():
            return False
        if self.region is None or other.region is None:
            return False

        is_left: bool = self.region.right <= other.region.left
        return is_left

    def is_right_of(self, other: "Match") -> bool:
        """Check if this match is to the right of another.

        Args:
            other: Other match

        Returns:
            True if this match is to the right
        """
        if not self.exists() or not other.exists():
            return False
        if self.region is None or other.region is None:
            return False

        is_right: bool = self.region.left >= other.region.right
        return is_right

    def is_above(self, other: "Match") -> bool:
        """Check if this match is above another.

        Args:
            other: Other match

        Returns:
            True if this match is above
        """
        if not self.exists() or not other.exists():
            return False
        if self.region is None or other.region is None:
            return False

        is_above: bool = self.region.bottom <= other.region.top
        return is_above

    def is_below(self, other: "Match") -> bool:
        """Check if this match is below another.

        Args:
            other: Other match

        Returns:
            True if this match is below
        """
        if not self.exists() or not other.exists():
            return False
        if self.region is None or other.region is None:
            return False

        is_below: bool = self.region.top >= other.region.bottom
        return is_below

    def __str__(self) -> str:
        """String representation."""
        if not self.exists():
            return "Match(not found)"
        return str(self.match_object)

    def __repr__(self) -> str:
        """Developer representation."""
        return f"Match({self.match_object!r})"

    def __bool__(self) -> bool:
        """Boolean evaluation - True if match exists."""
        return self.exists()

    @classmethod
    def empty(cls) -> "Match":
        """Create an empty (not found) match.

        Returns:
            Empty Match instance
        """
        match_obj = MatchObject()
        match_obj.target = Location(0, 0)
        match_obj.score = 0.0
        return cls(match_obj)
