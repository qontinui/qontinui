"""Mouse wrapper for mock/live automation switching.

Based on Brobot's wrapper pattern - provides stable API that routes
to mock or live implementation based on execution mode.
"""

import logging

from ..hal.factory import HALFactory
from ..hal.interfaces.input_controller import MouseButton, MousePosition
from ..mock.mock_mode_manager import MockModeManager
from ..mock.mock_input import MockInput

logger = logging.getLogger(__name__)


class Mouse:
    """Wrapper for mouse operations that routes to mock or live implementation.

    This wrapper provides a stable API for mouse operations while allowing
    the underlying implementation to switch between mock and live modes.

    In mock mode:
    - All operations complete instantly (no real mouse movement)
    - State tracked for testing verification

    In live mode:
    - Uses HAL input controller for real mouse operations
    - Actual system mouse control

    Example usage:
        Mouse.move(100, 200)  # Move to coordinates (real or simulated)
        Mouse.click(300, 400)  # Click at coordinates
    """

    _mock_input = MockInput()

    @classmethod
    def move(cls, x: int, y: int, duration: float = 0.0) -> bool:
        """Move mouse to absolute position.

        In mock mode: Updates mock position instantly
        In live mode: Moves actual mouse cursor

        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Movement duration in seconds (0 for instant)

        Returns:
            True if successful
        """
        if MockModeManager.is_mock_mode():
            result = cls._mock_input.mouse_move(x, y, duration)
            logger.debug(f"[MOCK] Mouse moved to ({x}, {y})")
            return result
        else:
            controller = HALFactory.get_input_controller()
            result = controller.mouse_move(x, y, duration)
            logger.debug(f"[LIVE] Mouse moved to ({x}, {y})")
            return result

    @classmethod
    def move_relative(cls, dx: int, dy: int, duration: float = 0.0) -> bool:
        """Move mouse relative to current position.

        Args:
            dx: X offset
            dy: Y offset
            duration: Movement duration in seconds

        Returns:
            True if successful
        """
        if MockModeManager.is_mock_mode():
            return cls._mock_input.mouse_move_relative(dx, dy, duration)
        else:
            controller = HALFactory.get_input_controller()
            return controller.mouse_move_relative(dx, dy, duration)

    @classmethod
    def click(
        cls,
        x: int | None = None,
        y: int | None = None,
        button: MouseButton = MouseButton.LEFT,
        clicks: int = 1,
    ) -> bool:
        """Click mouse button.

        Args:
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)
            button: Mouse button to click
            clicks: Number of clicks

        Returns:
            True if successful
        """
        if MockModeManager.is_mock_mode():
            result = cls._mock_input.mouse_click(x, y, button, clicks)
            logger.debug(f"[MOCK] Mouse clicked {button.value} at ({x}, {y}) x{clicks}")
            return result
        else:
            controller = HALFactory.get_input_controller()
            result = controller.mouse_click(x, y, button, clicks)
            logger.debug(f"[LIVE] Mouse clicked {button.value} at ({x}, {y}) x{clicks}")
            return result

    @classmethod
    def click_at(cls, x: int, y: int, button: MouseButton = MouseButton.LEFT) -> bool:
        """Click at specific coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button to click

        Returns:
            True if successful
        """
        if MockModeManager.is_mock_mode():
            return cls._mock_input.click_at(x, y, button)
        else:
            controller = HALFactory.get_input_controller()
            return controller.click_at(x, y, button)

    @classmethod
    def double_click_at(cls, x: int, y: int, button: MouseButton = MouseButton.LEFT) -> bool:
        """Double click at specific coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button to click

        Returns:
            True if successful
        """
        if MockModeManager.is_mock_mode():
            return cls._mock_input.double_click_at(x, y, button)
        else:
            controller = HALFactory.get_input_controller()
            return controller.double_click_at(x, y, button)

    @classmethod
    def down(
        cls, x: int | None = None, y: int | None = None, button: MouseButton = MouseButton.LEFT
    ) -> bool:
        """Press and hold mouse button.

        Args:
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)
            button: Mouse button to press

        Returns:
            True if successful
        """
        if MockModeManager.is_mock_mode():
            return cls._mock_input.mouse_down(x, y, button)
        else:
            controller = HALFactory.get_input_controller()
            return controller.mouse_down(x, y, button)

    @classmethod
    def up(
        cls, x: int | None = None, y: int | None = None, button: MouseButton = MouseButton.LEFT
    ) -> bool:
        """Release mouse button.

        Args:
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)
            button: Mouse button to release

        Returns:
            True if successful
        """
        if MockModeManager.is_mock_mode():
            return cls._mock_input.mouse_up(x, y, button)
        else:
            controller = HALFactory.get_input_controller()
            return controller.mouse_up(x, y, button)

    @classmethod
    def drag(
        cls,
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
        """
        if MockModeManager.is_mock_mode():
            result = cls._mock_input.mouse_drag(start_x, start_y, end_x, end_y, button, duration)
            logger.debug(f"[MOCK] Mouse dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})")
            return result
        else:
            controller = HALFactory.get_input_controller()
            result = controller.mouse_drag(start_x, start_y, end_x, end_y, button, duration)
            logger.debug(f"[LIVE] Mouse dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})")
            return result

    @classmethod
    def scroll(cls, clicks: int, x: int | None = None, y: int | None = None) -> bool:
        """Scroll mouse wheel.

        Args:
            clicks: Number of scroll clicks (positive=up, negative=down)
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)

        Returns:
            True if successful
        """
        if MockModeManager.is_mock_mode():
            result = cls._mock_input.mouse_scroll(clicks, x, y)
            logger.debug(f"[MOCK] Mouse scrolled {clicks} clicks")
            return result
        else:
            controller = HALFactory.get_input_controller()
            result = controller.mouse_scroll(clicks, x, y)
            logger.debug(f"[LIVE] Mouse scrolled {clicks} clicks")
            return result

    @classmethod
    def position(cls) -> MousePosition:
        """Get current mouse position.

        Returns:
            Current mouse position
        """
        if MockModeManager.is_mock_mode():
            return cls._mock_input.get_mouse_position()
        else:
            controller = HALFactory.get_input_controller()
            return controller.get_mouse_position()

    @classmethod
    def reset_mock(cls) -> None:
        """Reset mock mouse state (for test cleanup).

        Only affects mock mode.
        """
        cls._mock_input.reset()
        logger.debug("Mock mouse reset")
