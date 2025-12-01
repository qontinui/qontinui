"""PyAutoGUI mouse operations implementation."""

from ...exceptions import InputControlError
from ...logging import get_logger
from ..interfaces.mouse_controller import IMouseController, MouseButton, MousePosition

logger = get_logger(__name__)


class PyAutoGUIMouseOperations(IMouseController):
    """Handles mouse input operations using PyAutoGUI.

    This implementation uses PyAutoGUI as the backend for mouse control,
    which provides cross-platform support without needing X11 on Linux.
    """

    def __init__(self) -> None:
        """Initialize PyAutoGUI mouse operations."""
        import pyautogui

        self._pyautogui = pyautogui
        # Disable PyAutoGUI's fail-safe (moving mouse to corner)
        pyautogui.FAILSAFE = False
        self._button_map = {
            MouseButton.LEFT: "left",
            MouseButton.RIGHT: "right",
            MouseButton.MIDDLE: "middle",
        }

    def _get_pyautogui_button(self, button: MouseButton | str) -> str:
        """Convert button to PyAutoGUI button name.

        Args:
            button: Button to convert (enum or string)

        Returns:
            PyAutoGUI button name
        """
        if isinstance(button, MouseButton):
            return self._button_map.get(button, "left")
        elif isinstance(button, str):
            return button.lower()
        return "left"

    def mouse_move(self, x: int, y: int, duration: float = 0.0) -> bool:
        """Move mouse to absolute position.

        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Movement duration in seconds (0 for instant)

        Returns:
            True if successful

        Raises:
            InputControlError: If mouse move fails
        """
        try:
            self._pyautogui.moveTo(x, y, duration=duration)
            logger.debug(f"Mouse moved to ({x}, {y})")
            return True

        except Exception as e:
            logger.error(f"Mouse move failed: {e}")
            raise InputControlError("mouse_move", str(e)) from e

    def mouse_move_relative(self, dx: int, dy: int, duration: float = 0.0) -> bool:
        """Move mouse relative to current position.

        Args:
            dx: X offset
            dy: Y offset
            duration: Movement duration in seconds

        Returns:
            True if successful

        Raises:
            InputControlError: If relative mouse move fails
        """
        try:
            self._pyautogui.moveRel(dx, dy, duration=duration)
            logger.debug(f"Mouse moved relative by ({dx}, {dy})")
            return True

        except Exception as e:
            logger.error(f"Relative mouse move failed: {e}")
            raise InputControlError("mouse_move_relative", str(e)) from e

    def mouse_click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: MouseButton = MouseButton.LEFT,
        clicks: int = 1,
        interval: float = 0.0,
    ) -> bool:
        """Click mouse button.

        Args:
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)
            button: Mouse button to click
            clicks: Number of clicks
            interval: Interval between clicks in seconds

        Returns:
            True if successful

        Raises:
            InputControlError: If mouse click fails
        """
        try:
            pyautogui_button = self._get_pyautogui_button(button)
            self._pyautogui.click(
                x=x, y=y, clicks=clicks, interval=interval, button=pyautogui_button
            )

            logger.debug(f"Mouse clicked {clicks} time(s) with {button}")
            return True

        except Exception as e:
            logger.error(f"Mouse click failed: {e}")
            raise InputControlError("mouse_click", str(e)) from e

    def mouse_down(
        self,
        x: int | None = None,
        y: int | None = None,
        button: MouseButton = MouseButton.LEFT,
    ) -> bool:
        """Press and hold mouse button.

        Args:
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)
            button: Mouse button to press

        Returns:
            True if successful

        Raises:
            InputControlError: If mouse down fails
        """
        try:
            # Move to position if specified
            if x is not None and y is not None:
                self._pyautogui.moveTo(x, y)

            pyautogui_button = self._get_pyautogui_button(button)
            self._pyautogui.mouseDown(button=pyautogui_button)

            logger.debug(f"Mouse button {button} pressed")
            return True

        except Exception as e:
            logger.error(f"Mouse down failed: {e}")
            raise InputControlError("mouse_down", str(e)) from e

    def mouse_up(
        self,
        x: int | None = None,
        y: int | None = None,
        button: MouseButton = MouseButton.LEFT,
    ) -> bool:
        """Release mouse button.

        Args:
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)
            button: Mouse button to release

        Returns:
            True if successful

        Raises:
            InputControlError: If mouse up fails
        """
        try:
            # Move to position if specified
            if x is not None and y is not None:
                self._pyautogui.moveTo(x, y)

            pyautogui_button = self._get_pyautogui_button(button)
            self._pyautogui.mouseUp(button=pyautogui_button)

            logger.debug(f"Mouse button {button} released")
            return True

        except Exception as e:
            logger.error(f"Mouse up failed: {e}")
            raise InputControlError("mouse_up", str(e)) from e

    def mouse_drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        button: MouseButton = MouseButton.LEFT,
        duration: float = 0.5,
    ) -> bool:
        """Drag mouse from start to end position.

        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            button: Mouse button to hold during drag
            duration: Drag duration in seconds

        Returns:
            True if successful

        Raises:
            InputControlError: If mouse drag fails
        """
        try:
            # Move to start position
            self._pyautogui.moveTo(start_x, start_y)

            pyautogui_button = self._get_pyautogui_button(button)
            self._pyautogui.drag(
                end_x - start_x,
                end_y - start_y,
                duration=duration,
                button=pyautogui_button,
            )

            logger.debug(
                f"Mouse dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})"
            )
            return True

        except Exception as e:
            logger.error(f"Mouse drag failed: {e}")
            raise InputControlError("mouse_drag", str(e)) from e

    def mouse_scroll(
        self, clicks: int, x: int | None = None, y: int | None = None
    ) -> bool:
        """Scroll mouse wheel.

        Args:
            clicks: Number of scroll clicks (positive=up, negative=down)
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)

        Returns:
            True if successful

        Raises:
            InputControlError: If mouse scroll fails
        """
        try:
            self._pyautogui.scroll(clicks, x=x, y=y)

            logger.debug(f"Mouse scrolled {clicks} clicks")
            return True

        except Exception as e:
            logger.error(f"Mouse scroll failed: {e}")
            raise InputControlError("mouse_scroll", str(e)) from e

    def get_mouse_position(self) -> MousePosition:
        """Get current mouse position.

        Returns:
            Current mouse position

        Raises:
            InputControlError: If get position fails
        """
        try:
            pos = self._pyautogui.position()
            return MousePosition(x=int(pos.x), y=int(pos.y))
        except Exception as e:
            logger.error(f"Get mouse position failed: {e}")
            raise InputControlError("get_mouse_position", str(e)) from e

    def click_at(self, x: int, y: int, button: MouseButton = MouseButton.LEFT) -> bool:
        """Click at specific coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button to click

        Returns:
            True if successful
        """
        return self.mouse_click(x, y, button, clicks=1)

    def double_click_at(
        self, x: int, y: int, button: MouseButton = MouseButton.LEFT
    ) -> bool:
        """Double click at specific coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button to click

        Returns:
            True if successful
        """
        return self.mouse_click(x, y, button, clicks=2, interval=0.1)

    def drag(
        self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.0
    ) -> bool:
        """Drag from start to end position.

        Args:
            start_x: Start X coordinate
            start_y: Start Y coordinate
            end_x: End X coordinate
            end_y: End Y coordinate
            duration: Drag duration in seconds

        Returns:
            True if successful
        """
        return self.mouse_drag(
            start_x, start_y, end_x, end_y, MouseButton.LEFT, duration
        )

    def move_mouse(self, x: int, y: int, duration: float = 0.0) -> bool:
        """Move mouse to position (alias for mouse_move).

        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Movement duration in seconds

        Returns:
            True if successful
        """
        return self.mouse_move(x, y, duration)

    def scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> bool:
        """Scroll mouse wheel (alias for mouse_scroll).

        Args:
            clicks: Number of scroll clicks (positive=up, negative=down)
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)

        Returns:
            True if successful
        """
        return self.mouse_scroll(clicks, x, y)
