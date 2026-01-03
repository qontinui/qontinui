"""Vanish action implementation - ported from Qontinui framework.

Waits for visual elements to disappear from the screen.
"""

import time
from datetime import timedelta
from typing import Any

from ....actions.find import FindAction, FindOptions
from ...action_interface import ActionInterface
from ...action_result import ActionResult
from ...object_collection import ObjectCollection
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

    def __init__(self, find_action: FindAction | None = None) -> None:
        """Initialize Vanish action.

        Args:
            find_action: The FindAction for checking element presence
        """
        self.find_action = find_action or FindAction()

    async def perform(
        self, action_result: ActionResult, *object_collections: ObjectCollection
    ) -> None:
        """Execute the vanish operation.

        Repeatedly checks for the presence of elements until they disappear
        or the maximum wait time is exceeded.

        Args:
            action_result: Result container that will be populated
            object_collections: Collections defining what should vanish
        """
        if not isinstance(action_result.action_config, VanishOptions):
            object.__setattr__(action_result, "success", False)
            return

        vanish_options = action_result.action_config

        # Wait for elements to vanish
        vanished = await self._wait_for_vanish(
            action_result,
            object_collections,
            vanish_options.get_max_wait_time(),
            vanish_options.get_poll_interval(),
        )

        object.__setattr__(action_result, "success", vanished)
        if vanished:
            object.__setattr__(action_result, "output_text", "Element(s) vanished successfully")
        else:
            object.__setattr__(
                action_result,
                "output_text",
                f"Element(s) still present after {vanish_options.get_max_wait_time()} seconds",
            )

    async def _wait_for_vanish(
        self,
        action_result: ActionResult,
        object_collections: tuple[Any, ...],
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
        import asyncio

        if not object_collections:
            # Nothing to check for vanishing
            return True

        start_time = time.time()

        while time.time() - start_time < max_wait:
            # Check if elements are still present
            if await self._elements_are_gone(object_collections):
                # Elements have vanished
                object.__setattr__(
                    action_result,
                    "duration",
                    timedelta(seconds=time.time() - start_time),
                )
                return True

            # Wait before next check
            await asyncio.sleep(poll_interval)

        # Timeout - elements still present
        object.__setattr__(action_result, "duration", timedelta(seconds=max_wait))
        return False

    async def _elements_are_gone(self, object_collections: tuple[Any, ...]) -> bool:
        """Check if elements are no longer present.

        Args:
            object_collections: Collections to check

        Returns:
            True if elements are gone, False if still present
        """
        # Collect all patterns for parallel search
        patterns = []
        for obj_coll in object_collections:
            for state_image in obj_coll.state_images:
                pattern = state_image.get_pattern()
                if pattern:
                    patterns.append(pattern)

        if not patterns:
            return True

        options = FindOptions(similarity=0.8)
        results = await self.find_action.find(patterns, options)

        # If any element is found, they haven't vanished
        for result in results:
            if result.found:
                return False

        # No elements found - they have vanished
        return True
