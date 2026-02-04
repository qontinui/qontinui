"""Unified Actions class that handles both mock and live execution.

This is the main entry point for all actions in Qontinui applications.
Mock vs live execution is determined by FrameworkSettings configuration.
"""

import logging
from typing import Any, cast

from ..model.element import Location, Pattern, Region
from .action_result import ActionResult, ActionResultBuilder
from .find import FindAction, FindOptions
from .fluent import FluentActions
from .pure import PureActions

logger = logging.getLogger(__name__)


class Actions:
    """Unified action interface that automatically handles mock/live execution.

    Following Brobot principles:
    - Application code doesn't know if it's running mock or live
    - Mock/live decision is made at the lowest action level (FindAction)
    - All higher-level code is agnostic to execution mode
    """

    def __init__(self) -> None:
        """Initialize Actions with appropriate implementations."""
        self.pure = PureActions()
        self.fluent = FluentActions()

        # FindAction handles mock/real routing internally
        self._find_action: FindAction | None = None
        self._mock_actions = None
        self._mock_mode_checked = False

    @property
    def find_action(self) -> FindAction:
        """Get FindAction instance (lazy initialization)."""
        if self._find_action is None:
            self._find_action = FindAction()
        return self._find_action

    def _ensure_mock_initialized(self):
        """Lazy load mock implementations to avoid circular imports."""
        if not self._mock_mode_checked:
            self._mock_mode_checked = True
            from ..mock import MockActions, MockModeManager

            if MockModeManager.is_mock_mode():
                self._mock_actions = MockActions()
                logger.info("Actions initialized in MOCK mode")
            else:
                logger.info("Actions initialized in LIVE mode")

    async def find(self, target: Pattern, region: Region | None = None) -> ActionResult:
        """Find a pattern on screen.

        Args:
            target: Pattern to find
            region: Optional search region

        Returns:
            ActionResult with matches
        """
        from ..find.match import Match as FindMatch

        # FindAction handles mock/real routing internally
        options = FindOptions(search_region=region, find_all=False)
        find_result = await self.find_action.find(target, options)

        # Convert FindResult to ActionResult for compatibility
        # Wrap model.Match into find.Match (which has action methods)
        builder = ActionResultBuilder().with_success(find_result.found)
        if find_result.found:
            for match in find_result.matches:
                wrapped_match = FindMatch(match_object=match)
                builder.add_match(wrapped_match)
        return builder.build()

    async def click(self, target: Any, button: str = "left") -> ActionResult:
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
            return ActionResultBuilder().with_success(True).build()
        else:
            # Live implementation
            if isinstance(target, Pattern):
                # Find pattern first, then click
                find_result = await self.find(target)
                if find_result.success and find_result.matches:
                    match = find_result.matches[0]
                    location = match.center
                    return cast(
                        ActionResult,
                        self.pure.mouse_click(location.x, location.y, button),
                    )
                else:
                    return (
                        ActionResultBuilder()
                        .with_success(False)
                        .with_output_text("Pattern not found")
                        .build()
                    )
            elif isinstance(target, Location):
                return cast(ActionResult, self.pure.mouse_click(target.x, target.y, button))
            elif isinstance(target, Region):
                center = target.get_center()
                return cast(ActionResult, self.pure.mouse_click(center.x, center.y, button))
            else:
                return (
                    ActionResultBuilder()
                    .with_success(False)
                    .with_output_text(f"Unsupported target type: {type(target)}")
                    .build()
                )

    async def type(self, text: str, target: Any | None = None) -> ActionResult:
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
            return ActionResultBuilder().with_success(True).build()
        else:
            # Live implementation
            if target:
                click_result = await self.click(target)
                if not click_result.success:
                    return click_result

            # Type the text
            for char in text:
                result = self.pure.type_character(char)  # type: ignore[assignment]
                if not result.success:
                    return result

            return ActionResultBuilder().with_success(True).build()

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
            return ActionResultBuilder().with_success(True).build()
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
            return ActionResultBuilder().with_success(True).build()
        else:
            # Live implementation

            return cast(ActionResult, self.pure.wait(seconds))

    async def wait_for(self, target: Pattern, timeout: float = 5.0) -> ActionResult:
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
            find_result = await self.find(target)
            return ActionResultBuilder().with_success(find_result.success).build()
        else:
            # Live implementation would poll for pattern
            import time

            start_time = time.time()

            while time.time() - start_time < timeout:
                find_result = await self.find(target)
                if find_result.success:
                    return ActionResultBuilder().with_success(True).build()
                time.sleep(0.5)

            return (
                ActionResultBuilder()
                .with_success(False)
                .with_output_text(f"Pattern {target.name} not found after {timeout}s")
                .build()
            )

    async def wait_vanish(self, target: Pattern, timeout: float = 30.0) -> ActionResult:
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
            return ActionResultBuilder().with_success(True).build()
        else:
            # Live implementation would poll until pattern disappears
            import time

            start_time = time.time()

            while time.time() - start_time < timeout:
                find_result = await self.find(target)
                if not find_result.success:
                    return ActionResultBuilder().with_success(True).build()
                time.sleep(0.5)

            return (
                ActionResultBuilder()
                .with_success(False)
                .with_output_text(f"Pattern {target.name} still visible after {timeout}s")
                .build()
            )

    async def drag(self, from_target: Any, to_target: Any, duration: float = 1.0) -> ActionResult:
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
            return ActionResultBuilder().with_success(True).build()
        else:
            # Live implementation
            # Convert targets to locations
            from_loc = await self._get_location(from_target)
            to_loc = await self._get_location(to_target)

            if not from_loc or not to_loc:
                return (
                    ActionResultBuilder()
                    .with_success(False)
                    .with_output_text("Could not determine drag locations")
                    .build()
                )

            # Perform drag

            return cast(
                ActionResult,
                self.pure.mouse_drag(from_loc.x, from_loc.y, to_loc.x, to_loc.y, duration),  # type: ignore[attr-defined]
            )

    async def _get_location(self, target: Any) -> Location | None:
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
            find_result = await self.find(target)
            if find_result.success and find_result.matches:
                return cast(Location, find_result.matches[0].center)
        elif isinstance(target, tuple | list) and len(target) == 2:
            return Location(target[0], target[1])

        return None
