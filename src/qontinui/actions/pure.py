"""Pure atomic actions following Brobot principles."""

import time
from dataclasses import dataclass
from typing import Any

from ..hal.factory import HALFactory
from ..hal.interfaces import MouseButton


@dataclass
class ActionResult:
    """Result of an action execution."""

    success: bool
    data: Any | None = None
    error: str | None = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class UnifiedInputController:
    """Unified controller wrapping both mouse and keyboard controllers.

    This class combines mouse and keyboard HAL interfaces into a single
    controller object for convenience in action classes.
    """

    def __init__(self) -> None:
        """Initialize with separate mouse and keyboard controllers."""
        self._mouse = HALFactory.get_mouse_controller()
        self._keyboard = HALFactory.get_keyboard_controller()

    # Mouse operations - delegate to mouse controller
    def move_mouse(self, x: int, y: int) -> None:
        """Move mouse to coordinates."""
        self._mouse.move_mouse(x, y)

    def move_mouse_smooth(self, x: int, y: int, duration: float) -> None:
        """Move mouse smoothly to coordinates."""
        self._mouse.move_mouse(x, y, duration)

    def mouse_down(self, button: MouseButton | str) -> None:
        """Press mouse button."""
        self._mouse.mouse_down(
            button=button if isinstance(button, MouseButton) else MouseButton[button.upper()]  # type: ignore[arg-type]
        )

    def mouse_up(self, button: MouseButton | str) -> None:
        """Release mouse button."""
        self._mouse.mouse_up(
            button=button if isinstance(button, MouseButton) else MouseButton[button.upper()]  # type: ignore[arg-type]
        )

    def click_at(self, x: int, y: int, button: MouseButton | str, clicks: int = 1) -> None:
        """Click at coordinates."""
        self._mouse.mouse_click(
            x,
            y,
            button if isinstance(button, MouseButton) else MouseButton[button.upper()],
            clicks,  # type: ignore[arg-type]
        )

    def scroll(self, clicks: int) -> None:
        """Scroll mouse wheel."""
        self._mouse.scroll(clicks)

    def get_mouse_position(self) -> tuple[int, int]:
        """Get current mouse position."""
        pos = self._mouse.get_mouse_position()
        return (pos.x, pos.y)

    # Keyboard operations - delegate to keyboard controller
    def key_down(self, key: str) -> None:
        """Press key down."""
        self._keyboard.key_down(key)

    def key_up(self, key: str) -> None:
        """Release key."""
        self._keyboard.key_up(key)

    def key_press(self, key: str) -> None:
        """Press and release key."""
        self._keyboard.key_press(key)

    def type_text(self, text: str, interval: float = 0.0) -> None:
        """Type text with optional interval between characters."""
        self._keyboard.type_text(text, interval)


class PureActions:
    """Pure, atomic actions that do one thing only.

    Following Brobot principles:
    - Each action is atomic and does exactly one thing
    - No composite actions (like drag = mouseDown + move + mouseUp)
    - Actions return results for chaining
    - No retry logic in base actions

    Now uses HAL (Hardware Abstraction Layer) instead of pyautogui directly.
    """

    def __init__(self) -> None:
        """Initialize pure actions with HAL controller."""
        self.controller = UnifiedInputController()
        self.screen_capture = HALFactory.get_screen_capture()

    # Mouse Actions (Atomic)

    def mouse_down(
        self, x: int | None = None, y: int | None = None, button: str = "left"
    ) -> ActionResult:
        """Press and hold mouse button.

        Args:
            x: X coordinate (None = current position)
            y: Y coordinate (None = current position)
            button: 'left', 'right', or 'middle'

        Returns:
            ActionResult with success status
        """
        try:
            # Convert string button to MouseButton enum if needed
            btn = self._get_button(button)

            if x is not None and y is not None:
                # Move to position first
                self.controller.move_mouse(x, y)

            self.controller.mouse_down(btn)
            return ActionResult(success=True, data={"x": x, "y": y, "button": button})
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    def mouse_up(
        self, x: int | None = None, y: int | None = None, button: str = "left"
    ) -> ActionResult:
        """Release mouse button.

        Args:
            x: X coordinate (None = current position)
            y: Y coordinate (None = current position)
            button: 'left', 'right', or 'middle'

        Returns:
            ActionResult with success status
        """
        try:
            # Convert string button to MouseButton enum if needed
            btn = self._get_button(button)

            if x is not None and y is not None:
                # Move to position first
                self.controller.move_mouse(x, y)

            self.controller.mouse_up(btn)
            return ActionResult(success=True, data={"x": x, "y": y, "button": button})
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    def mouse_move(self, x: int, y: int, duration: float = 0.0) -> ActionResult:
        """Move mouse to coordinates.

        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Duration of movement in seconds

        Returns:
            ActionResult with success status
        """
        try:
            if duration > 0:
                # Smooth movement with duration
                self.controller.move_mouse_smooth(x, y, duration)
            else:
                # Instant movement
                self.controller.move_mouse(x, y)
            return ActionResult(success=True, data=(x, y))
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    def mouse_click(self, x: int, y: int, button: str = "left") -> ActionResult:
        """Single atomic click at position.

        Args:
            x: X coordinate
            y: Y coordinate
            button: 'left', 'right', or 'middle'

        Returns:
            ActionResult with success status
        """
        try:
            # Convert string button to MouseButton enum if needed
            btn = self._get_button(button)
            self.controller.click_at(x, y, btn)
            return ActionResult(success=True, data=(x, y))
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    def mouse_scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> ActionResult:
        """Scroll mouse wheel.

        Args:
            clicks: Number of scroll clicks (positive=up, negative=down)
            x: Optional X coordinate
            y: Optional Y coordinate

        Returns:
            ActionResult with success status
        """
        try:
            if x is not None and y is not None:
                # Move to position first
                self.controller.move_mouse(x, y)

            self.controller.scroll(clicks)
            return ActionResult(success=True, data=clicks)
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    # Keyboard Actions (Atomic)

    def key_down(self, key: str) -> ActionResult:
        """Press and hold key.

        Args:
            key: Key to press

        Returns:
            ActionResult with success status
        """
        try:
            self.controller.key_down(key)
            return ActionResult(success=True, data=key)
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    def key_up(self, key: str) -> ActionResult:
        """Release key.

        Args:
            key: Key to release

        Returns:
            ActionResult with success status
        """
        try:
            self.controller.key_up(key)
            return ActionResult(success=True, data=key)
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    def key_press(self, key: str) -> ActionResult:
        """Press and release key (atomic).

        Args:
            key: Key to press

        Returns:
            ActionResult with success status
        """
        try:
            self.controller.key_press(key)
            return ActionResult(success=True, data=key)
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    def type_text(self, text: str, interval: float = 0.0) -> ActionResult:
        """Type text string.

        Args:
            text: Text to type
            interval: Interval between keystrokes

        Returns:
            ActionResult with success status
        """
        try:
            self.controller.type_text(text, interval)
            return ActionResult(success=True, data=text)
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    def type_character(self, char: str) -> ActionResult:
        """Type a single character.

        Args:
            char: Character to type

        Returns:
            ActionResult with success status
        """
        try:
            self.controller.type_text(char, 0.0)
            return ActionResult(success=True, data=char)
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    # Utility Actions

    def wait(self, duration: float) -> ActionResult:
        """Wait for specified duration.

        Args:
            duration: Duration in seconds

        Returns:
            ActionResult with success status
        """
        try:
            time.sleep(duration)
            return ActionResult(success=True, data=duration)
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    def get_mouse_position(self) -> ActionResult:
        """Get current mouse position.

        Returns:
            ActionResult with position data
        """
        try:
            pos = self.controller.get_mouse_position()
            return ActionResult(success=True, data={"x": pos[0], "y": pos[1]})
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    def screenshot(self, region: tuple[int, int, int, int] | None = None) -> ActionResult:
        """Take screenshot.

        Args:
            region: Optional (x, y, width, height) region

        Returns:
            ActionResult with image data
        """
        try:
            if region:
                image = self.screen_capture.capture_region(*region)
            else:
                image = self.screen_capture.capture_screen()
            return ActionResult(success=True, data=image)
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    # Helper methods

    def _get_button(self, button: str | MouseButton) -> MouseButton:
        """Convert string button to MouseButton enum.

        Args:
            button: Button as string or MouseButton

        Returns:
            MouseButton enum value
        """
        if isinstance(button, MouseButton):
            return button

        button_map = {
            "left": MouseButton.LEFT,
            "right": MouseButton.RIGHT,
            "middle": MouseButton.MIDDLE,
        }

        button_lower = button.lower()
        if button_lower not in button_map:
            raise ValueError(f"Invalid button: {button}")

        return button_map[button_lower]
