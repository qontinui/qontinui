"""Mock implementation of actions for testing.

All mock behavior is contained here, keeping application code clean.
"""

import logging
from typing import Any, cast

from ..actions.action_result import ActionResult

logger = logging.getLogger(__name__)


class MockActions:
    """Mock implementation of actions for testing.

    Following Brobot principles:
    - Mock behavior is realistic based on state
    - Uses ActionHistory for probabilistic behavior
    - Application code doesn't know it's using mocks
    """

    def __init__(self):
        """Initialize mock actions."""
        logger.debug("MockActions initialized")

    def click(self, target: Any, button: str = "left") -> ActionResult:
        """Mock click action.

        Args:
            target: Target to click
            button: Mouse button

        Returns:
            ActionResult
        """
        target_name = self._get_target_name(target)
        logger.info(f"[Mock] Click {button} on {target_name}")

        result = ActionResult()
        result.success = True
        result.action_description = f"Clicked {target_name}"
        return result

    def type(self, text: str, target: Any | None = None) -> ActionResult:
        """Mock type action.

        Args:
            text: Text to type
            target: Optional target to click first

        Returns:
            ActionResult
        """
        if target:
            target_name = self._get_target_name(target)
            logger.info(f"[Mock] Type '{text}' at {target_name}")
        else:
            logger.info(f"[Mock] Type '{text}'")

        result = ActionResult()
        result.success = True
        result.action_description = f"Typed: {text}"
        result.selected_text = text
        return result

    def key(self, key_name: str) -> ActionResult:
        """Mock key press.

        Args:
            key_name: Key to press

        Returns:
            ActionResult
        """
        logger.info(f"[Mock] Press key: {key_name}")

        result = ActionResult()
        result.success = True
        result.action_description = f"Pressed {key_name}"
        return result

    def drag(self, from_target: Any, to_target: Any, duration: float = 1.0) -> ActionResult:
        """Mock drag action.

        Args:
            from_target: Starting point
            to_target: Ending point
            duration: Duration of drag

        Returns:
            ActionResult
        """
        from_name = self._get_target_name(from_target)
        to_name = self._get_target_name(to_target)
        logger.info(f"[Mock] Drag from {from_name} to {to_name} over {duration}s")

        result = ActionResult()
        result.success = True
        result.action_description = f"Dragged from {from_name} to {to_name}"
        return result

    def _get_target_name(self, target: Any) -> str:
        """Get a readable name for a target.

        Args:
            target: Any target type

        Returns:
            String representation
        """

        if hasattr(target, "name"):
            return cast(str, target.name)
        elif hasattr(target, "__class__"):
            return cast(str, target.__class__.__name__)
        else:
            return str(target)
