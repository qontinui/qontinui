"""Scroll action implementation - ported from Qontinui framework.

Performs scroll operations at specified locations.
"""

import time
from typing import Any, cast

from ....model.element.location import Location
from ...action_interface import ActionInterface
from ...action_result import ActionResult
from ...object_collection import ObjectCollection
from ..find.find import Find
from .scroll_options import ScrollDirection, ScrollOptions


class Scroll(ActionInterface):
    """Performs scroll operations at specified locations.

    Port of Scroll from Qontinui framework action.

    The Scroll action can:
    - Scroll at a specific location found through pattern matching
    - Scroll in different directions (up, down, left, right)
    - Perform multiple scroll clicks
    - Use smooth or discrete scrolling
    """

    def __init__(self, find: Find | None = None) -> None:
        """Initialize Scroll action.

        Args:
            find: The Find action for locating scroll positions
        """
        self.find = find

    def perform(self, action_result: ActionResult, *object_collections: ObjectCollection) -> None:
        """Execute the scroll operation.

        First finds the target location using Find, then performs the
        configured number of scroll actions at that location.

        Args:
            action_result: Result container that will be populated
            object_collections: Collections defining what/where to scroll
        """
        if not isinstance(action_result.action_config, ScrollOptions):
            object.__setattr__(action_result, "success", False)
            return

        scroll_options = action_result.action_config

        # Find the location to scroll at
        location = self._find_scroll_location(action_result, object_collections)
        if not location:
            object.__setattr__(action_result, "success", False)
            object.__setattr__(action_result, "output_text", "Could not find location to scroll")
            return

        # Perform the scroll operations
        success = self._perform_scroll(
            location,
            scroll_options.get_direction(),
            scroll_options.get_clicks(),
            scroll_options.is_smooth(),
            scroll_options.get_delay_between_scrolls(),
        )

        object.__setattr__(action_result, "success", success)
        if success:
            object.__setattr__(
                action_result,
                "output_text",
                f"Scrolled {scroll_options.get_direction().name} {scroll_options.get_clicks()} times",
            )

    def _find_scroll_location(
        self, action_result: ActionResult, object_collections: tuple[Any, ...]
    ) -> Location | None:
        """Find the location to scroll at.

        Args:
            action_result: The action result for storing find results
            object_collections: Collections to search in

        Returns:
            Location to scroll at, or None if not found
        """
        if not object_collections:
            # No specific location, use screen center
            return Location(640, 360)  # Default center position

        # Use Find to locate the target
        if self.find:
            find_result = ActionResult(action_result.action_config)
            self.find.perform(find_result, *object_collections)

            if find_result.is_success and find_result.matches:
                # Use the first match's location
                first_match = find_result.matches[0]
                return cast(Location | None, first_match.get_target())

        return None

    def _perform_scroll(
        self,
        location: Location,
        direction: ScrollDirection,
        clicks: int,
        smooth: bool,
        delay: float,
    ) -> bool:
        """Perform the actual scroll operations.

        Args:
            location: Location to scroll at
            direction: Direction to scroll
            clicks: Number of scroll clicks
            smooth: Whether to use smooth scrolling
            delay: Delay between scrolls in seconds

        Returns:
            True if scrolling succeeded
        """
        # This would interface with actual scrolling backend
        # For now, simulate the scroll with timing

        for i in range(clicks):
            # Simulate scroll action
            self._execute_single_scroll(location, direction, smooth)

            # Delay between scrolls if not the last one
            if i < clicks - 1 and delay > 0:
                time.sleep(delay)

        return True

    def _execute_single_scroll(
        self, location: Location, direction: ScrollDirection, smooth: bool
    ) -> None:
        """Execute a single scroll action.

        Args:
            location: Location to scroll at
            direction: Direction to scroll
            smooth: Whether to use smooth scrolling
        """
        # Placeholder for actual scroll implementation
        # Would interface with platform-specific scrolling API

        # Map direction to scroll delta
        if direction == ScrollDirection.UP:
            _delta_x, _delta_y = 0, -120
        elif direction == ScrollDirection.DOWN:
            _delta_x, _delta_y = 0, 120
        elif direction == ScrollDirection.LEFT:
            _delta_x, _delta_y = -120, 0
        elif direction == ScrollDirection.RIGHT:
            _delta_x, _delta_y = 120, 0
        else:
            _delta_x, _delta_y = 0, 0

        # Placeholder implementation - integrate with HAL scroll API
        # Replace with: MouseWrapper.scroll(_delta_x, _delta_y, location) when HAL is available
        _ = (_delta_x, _delta_y)  # Acknowledge unused variables

        # Simulate scroll timing
        if smooth:
            time.sleep(0.05)  # Smooth scroll takes slightly longer
        else:
            time.sleep(0.01)  # Quick discrete scroll
