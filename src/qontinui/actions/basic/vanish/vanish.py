"""Vanish action implementation - ported from Qontinui framework.

Waits for visual elements to disappear from the screen.
"""

import time

from ...action_interface import ActionInterface
from ...action_result import ActionResult
from ...object_collection import ObjectCollection
from ..find.find import Find
from .vanish_options import VanishOptions


class Vanish(ActionInterface):
    """Waits for visual elements to disappear from the screen.

    Port of Vanish from Qontinui framework action.

    The Vanish action repeatedly checks if a visual element is still present,
    succeeding when the element can no longer be found. This is useful for:
    - Waiting for loading screens to disappear
    - Confirming dialogs have closed
    - Ensuring temporary UI elements are gone
    """

    def __init__(self, find: Find | None = None):
        """Initialize Vanish action.

        Args:
            find: The Find action for checking element presence
        """
        self.find = find

    def perform(self, action_result: ActionResult, *object_collections: ObjectCollection) -> None:
        """Execute the vanish operation.

        Repeatedly checks for the presence of elements until they disappear
        or the maximum wait time is exceeded.

        Args:
            action_result: Result container that will be populated
            object_collections: Collections defining what should vanish
        """
        if not isinstance(action_result.action_config, VanishOptions):
            action_result.success = False
            return

        vanish_options = action_result.action_config

        # Wait for elements to vanish
        vanished = self._wait_for_vanish(
            action_result,
            object_collections,
            vanish_options.get_max_wait_time(),
            vanish_options.get_poll_interval(),
        )

        action_result.success = vanished
        if vanished:
            action_result.text = "Element(s) vanished successfully"
        else:
            action_result.text = (
                f"Element(s) still present after {vanish_options.get_max_wait_time()} seconds"
            )

    def _wait_for_vanish(
        self,
        action_result: ActionResult,
        object_collections: tuple,
        max_wait: float,
        poll_interval: float,
    ) -> bool:
        """Wait for elements to disappear.

        Args:
            action_result: The action result for tracking
            object_collections: Collections to check for vanishing
            max_wait: Maximum time to wait in seconds
            poll_interval: Time between checks in seconds

        Returns:
            True if elements vanished, False if timeout
        """
        if not object_collections or not self.find:
            # Nothing to check for vanishing
            return True

        start_time = time.time()

        while time.time() - start_time < max_wait:
            # Check if elements are still present
            if self._elements_are_gone(action_result, object_collections):
                # Elements have vanished
                action_result.duration = time.time() - start_time
                return True

            # Wait before next check
            time.sleep(poll_interval)

        # Timeout - elements still present
        action_result.duration = max_wait
        return False

    def _elements_are_gone(self, action_result: ActionResult, object_collections: tuple) -> bool:
        """Check if elements are no longer present.

        Args:
            action_result: The action result for tracking
            object_collections: Collections to check

        Returns:
            True if elements are gone, False if still present
        """
        # Use Find to check for presence
        find_result = ActionResult(action_result.action_config)
        self.find.perform(find_result, *object_collections)

        # Elements are gone if Find fails or finds no matches
        return not find_result.is_success() or len(find_result.match_list) == 0
