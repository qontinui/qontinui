"""Mouse input operations."""

import time

from pynput import mouse
from pynput.mouse import Button as PynputButton

from ...exceptions import InputControlError
from ...logging import get_logger
from ...wrappers.time_wrapper import TimeWrapper
from ..interfaces.mouse_controller import IMouseController, MouseButton, MousePosition

logger = get_logger(__name__)


class MouseOperations(IMouseController):
    """Handles all mouse input operations.

    This class encapsulates mouse-specific functionality, separating
    it from keyboard operations and following the Single Responsibility Principle.
    """

    def __init__(self, mouse_controller: mouse.Controller) -> None:
        """Initialize mouse operations.

        Args:
            mouse_controller: Pynput mouse controller instance
        """
        self._mouse = mouse_controller
        # Map button values (strings) to pynput buttons
        self._button_map = {
            "left": PynputButton.left,
            "right": PynputButton.right,
            "middle": PynputButton.middle,
        }

    def _get_pynput_button(self, button: MouseButton | str) -> PynputButton:
        """Convert button to Pynput button.

        Args:
            button: Button to convert (enum or string)

        Returns:
            Pynput button
        """
        # If it's an enum, get its value (string)
        if hasattr(button, "value"):
            button_str = button.value.lower()
        elif isinstance(button, str):
            button_str = button.lower()
        else:
            button_str = str(button).lower()

        return self._button_map.get(button_str, PynputButton.left)

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
            if duration > 0:
                # Smooth movement
                start_x, start_y = self._mouse.position
                steps = max(int(duration * 60), 1)  # 60 FPS
                delay = duration / steps

                for i in range(steps + 1):
                    progress = i / steps
                    current_x = int(start_x + (x - start_x) * progress)
                    current_y = int(start_y + (y - start_y) * progress)
                    self._mouse.position = (current_x, current_y)
                    if i < steps:
                        time.sleep(delay)
            else:
                # Instant movement
                self._mouse.position = (x, y)

            logger.debug(f"Mouse moved to ({x}, {y})")
            return True

        except (OSError, RuntimeError, ValueError, TypeError) as e:
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
            current_x, current_y = self._mouse.position
            return self.mouse_move(current_x + dx, current_y + dy, duration)

        except (OSError, RuntimeError, ValueError, TypeError) as e:
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
            # Move to position if specified
            if x is not None and y is not None:
                self.mouse_move(x, y)

            pynput_button = self._get_pynput_button(button)

            for i in range(clicks):
                self._mouse.click(pynput_button)
                if i < clicks - 1 and interval > 0:
                    TimeWrapper().wait(seconds=interval)

            logger.debug(f"Mouse clicked {clicks} time(s) with {button}")
            return True

        except (OSError, RuntimeError, ValueError, TypeError) as e:
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
                self.mouse_move(x, y)

            pynput_button = self._get_pynput_button(button)
            self._mouse.press(pynput_button)

            logger.debug(f"Mouse button {button} pressed")
            return True

        except (OSError, RuntimeError, ValueError, TypeError) as e:
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
                self.mouse_move(x, y)

            pynput_button = self._get_pynput_button(button)
            self._mouse.release(pynput_button)

            logger.debug(f"Mouse button {button} released")
            return True

        except (OSError, RuntimeError, ValueError, TypeError) as e:
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
            self.mouse_move(start_x, start_y)

            # Press button
            pynput_button = self._get_pynput_button(button)
            self._mouse.press(pynput_button)

            # Move to end position
            TimeWrapper().wait(seconds=0.1)  # Small delay before drag
            self.mouse_move(end_x, end_y, duration)

            # Release button
            TimeWrapper().wait(seconds=0.1)  # Small delay before release
            self._mouse.release(pynput_button)

            logger.debug(
                f"Mouse dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})"
            )
            return True

        except (OSError, RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Mouse drag failed: {e}")
            # Try to release button if pressed
            try:
                self._mouse.release(self._get_pynput_button(button))
            except (OSError, RuntimeError):
                # OK to ignore release errors in error handler
                pass
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
            # Move to position if specified
            if x is not None and y is not None:
                self.mouse_move(x, y)

            # Scroll (negative for down, positive for up)
            self._mouse.scroll(0, clicks)

            logger.debug(f"Mouse scrolled {clicks} clicks")
            return True

        except (OSError, RuntimeError, ValueError, TypeError) as e:
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
            x, y = self._mouse.position
            return MousePosition(x=int(x), y=int(y))
        except (OSError, RuntimeError, ValueError, TypeError) as e:
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
