"""Unified Actions class that handles both mock and live execution.

This is the main entry point for all actions in Qontinui applications.
Mock vs live execution is determined by application properties.
"""

import logging
from typing import Any, cast

from ..model.element import Location, Pattern, Region
from .action_result import ActionResult
from .fluent import FluentActions
from .pure import PureActions

logger = logging.getLogger(__name__)


class Actions:
    """Unified action interface that automatically handles mock/live execution.

    Following Brobot principles:
    - Application code doesn't know if it's running mock or live
    - Mock/live decision is made at the lowest action level
    - All higher-level code is agnostic to execution mode
    """

    def __init__(self):
        """Initialize Actions with appropriate implementations."""
        self.pure = PureActions()
        self.fluent = FluentActions()

        # Mock implementations (lazy loaded to avoid circular imports)
        self._mock_find = None
        self._mock_actions = None
        self._mock_mode_checked = False

    def _ensure_mock_initialized(self):
        """Lazy load mock implementations to avoid circular imports."""
        if not self._mock_mode_checked:
            self._mock_mode_checked = True
            # Import here to avoid circular imports
            from ..mock import MockActions, MockFind, MockModeManager

            if MockModeManager.is_mock_mode():
                self._mock_find = MockFind()
                self._mock_actions = MockActions()
                logger.info("Actions initialized in MOCK mode")
            else:
                logger.info("Actions initialized in LIVE mode")

    def find(self, target: Pattern, region: Region | None = None) -> ActionResult:
        """Find a pattern on screen.

        Args:
            target: Pattern to find
            region: Optional search region

        Returns:
            ActionResult with matches
        """
        self._ensure_mock_initialized()

        # Import here to avoid circular imports
        from ..mock import MockModeManager

        if MockModeManager.is_mock_mode():
            if self._mock_find:
                result: ActionResult = self._mock_find.find(target, region)
                return result
            # Fallback if mock not initialized
            result = ActionResult()
            result.success = False
            return result
        else:
            # Live implementation would use real computer vision
            # For now, return empty result
            result = ActionResult()
            result.success = True
            return result

    def click(self, target: Any, button: str = "left") -> ActionResult:
        """Click on a target.

        Args:
            target: Pattern, Location, Region, or coordinates to click
            button: Mouse button to use

        Returns:
            ActionResult
        """
        self._ensure_mock_initialized()
        from ..mock import MockModeManager

        if MockModeManager.is_mock_mode():
            if self._mock_actions:
                result: ActionResult = self._mock_actions.click(target, button)
                return result
            result = ActionResult()
            result.success = True
            return result
        else:
            # Live implementation
            if isinstance(target, Pattern):

                # Find pattern first, then click
                find_result = self.find(target)
                if find_result.success and find_result.get_match_list():
                    match = find_result.get_match_list()[0]
                    location = match.center
                    return cast(ActionResult, self.pure.mouse_click(location.x, location.y, button))
                else:
                    result = ActionResult()
                    result.success = False
                    result.output_text = "Pattern not found"
                    return result
            elif isinstance(target, Location):

                return cast(ActionResult, self.pure.mouse_click(target.x, target.y, button))
            elif isinstance(target, Region):

                center = target.get_center()
                return cast(ActionResult, self.pure.mouse_click(center.x, center.y, button))
            else:
                result = ActionResult()
                result.success = False
                result.output_text = f"Unsupported target type: {type(target)}"
                return result

    def type(self, text: str, target: Any | None = None) -> ActionResult:
        """Type text, optionally at a target location.

        Args:
            text: Text to type
            target: Optional target to click before typing

        Returns:
            ActionResult
        """
        self._ensure_mock_initialized()
        from ..mock import MockModeManager

        if MockModeManager.is_mock_mode():
            if self._mock_actions:
                result: ActionResult = self._mock_actions.type(text, target)
                return result
            result = ActionResult()
            result.success = True
            return result
        else:
            # Live implementation
            if target:
                click_result = self.click(target)
                if not click_result.success:
                    return click_result

            # Type the text
            for char in text:
                result = self.pure.type_character(char)
                if not result.success:
                    return result

            result = ActionResult()
            result.success = True
            return result

    def key(self, key_name: str) -> ActionResult:
        """Press a key.

        Args:
            key_name: Name of key to press (e.g., "ENTER", "ESC")

        Returns:
            ActionResult
        """
        self._ensure_mock_initialized()
        from ..mock import MockModeManager

        if MockModeManager.is_mock_mode():
            if self._mock_actions:
                result: ActionResult = self._mock_actions.key(key_name)
                return result
            result = ActionResult()
            result.success = True
            return result
        else:
            # Live implementation

            return cast(ActionResult, self.pure.key_press(key_name.lower()))

    def wait(self, seconds: float) -> ActionResult:
        """Wait for specified time.

        Args:
            seconds: Seconds to wait

        Returns:
            ActionResult
        """
        self._ensure_mock_initialized()
        from ..mock import MockModeManager

        if MockModeManager.is_mock_mode():
            # In mock mode, don't actually wait
            logger.debug(f"[Mock] Wait {seconds}s")
            result = ActionResult()
            result.success = True
            return result
        else:
            # Live implementation

            return cast(ActionResult, self.pure.wait(seconds))

    def wait_for(self, target: Pattern, timeout: float = 5.0) -> ActionResult:
        """Wait for a pattern to appear.

        Args:
            target: Pattern to wait for
            timeout: Timeout in seconds

        Returns:
            ActionResult
        """
        self._ensure_mock_initialized()
        from ..mock import MockModeManager

        if MockModeManager.is_mock_mode():
            # In mock mode, check if pattern would be found
            find_result = self.find(target)
            result = ActionResult()
            result.success = find_result.success
            return result
        else:
            # Live implementation would poll for pattern
            import time

            start_time = time.time()

            while time.time() - start_time < timeout:
                find_result = self.find(target)
                if find_result.success:
                    result = ActionResult()
                    result.success = True
                    return result
                time.sleep(0.5)

            result = ActionResult()
            result.success = False
            result.output_text = f"Pattern {target.name} not found after {timeout}s"
            return result

    def wait_vanish(self, target: Pattern, timeout: float = 30.0) -> ActionResult:
        """Wait for a pattern to disappear.

        Args:
            target: Pattern to wait to vanish
            timeout: Timeout in seconds

        Returns:
            ActionResult
        """
        self._ensure_mock_initialized()
        from ..mock import MockModeManager

        if MockModeManager.is_mock_mode():
            # In mock mode, just return success
            logger.debug(f"[Mock] Wait for {target.name} to vanish")
            result = ActionResult()
            result.success = True
            return result
        else:
            # Live implementation would poll until pattern disappears
            import time

            start_time = time.time()

            while time.time() - start_time < timeout:
                find_result = self.find(target)
                if not find_result.success:
                    result = ActionResult()
                    result.success = True
                    return result
                time.sleep(0.5)

            result = ActionResult()
            result.success = False
            result.output_text = f"Pattern {target.name} still visible after {timeout}s"
            return result

    def drag(self, from_target: Any, to_target: Any, duration: float = 1.0) -> ActionResult:
        """Drag from one target to another.

        Args:
            from_target: Starting point
            to_target: Ending point
            duration: Duration of drag

        Returns:
            ActionResult
        """
        self._ensure_mock_initialized()
        from ..mock import MockModeManager

        if MockModeManager.is_mock_mode():
            if self._mock_actions:
                result: ActionResult = self._mock_actions.drag(from_target, to_target, duration)
                return result
            result = ActionResult()
            result.success = True
            return result
        else:
            # Live implementation
            # Convert targets to locations
            from_loc = self._get_location(from_target)
            to_loc = self._get_location(to_target)

            if not from_loc or not to_loc:
                result = ActionResult()
                result.success = False
                result.output_text = "Could not determine drag locations"
                return result

            # Perform drag

            return cast(
                ActionResult,
                self.pure.mouse_drag(from_loc.x, from_loc.y, to_loc.x, to_loc.y, duration),
            )

    def _get_location(self, target: Any) -> Location | None:
        """Convert various target types to Location.

        Args:
            target: Pattern, Location, Region, or coordinates

        Returns:
            Location or None
        """
        if isinstance(target, Location):
            return target
        elif isinstance(target, Region):
            return target.get_center()
        elif isinstance(target, Pattern):

            find_result = self.find(target)
            if find_result.success and find_result.get_match_list():
                return cast(Location, find_result.get_match_list()[0].center)
        elif isinstance(target, tuple | list) and len(target) == 2:
            return Location(target[0], target[1])

        return None
