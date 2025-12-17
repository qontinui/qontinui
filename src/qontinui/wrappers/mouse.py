"""Mouse wrapper for mock/live automation switching.

Based on Brobot's wrapper pattern - provides stable API that routes
to mock or live implementation based on execution mode.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..hal.interfaces.input_controller import IInputController

from ..hal.interfaces.input_controller import MouseButton, MousePosition
from ..mock.mock_input import MockInput
from ..mock.mock_mode_manager import MockModeManager
from ..reporting import EventType, emit_event

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
    _controller: IInputController | None = None
    _controller_lock = threading.Lock()

    @classmethod
    def _get_controller(cls) -> IInputController:
        """Lazy initialization of input controller.

        Uses double-check locking pattern for thread-safe singleton.
        """
        if cls._controller is None:
            with cls._controller_lock:
                if cls._controller is None:
                    from ..hal.factory import HALFactory

                    cls._controller = HALFactory.get_input_controller()
        return cls._controller

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
        is_mock = MockModeManager.is_mock_mode()

        # Debug logging to file for troubleshooting
        import datetime

        debug_log_path = (
            r"C:\Users\Joshua\AppData\Local\Temp\qontinui_mouse_wrapper_debug.log"
        )
        try:
            with open(debug_log_path, "a") as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                f.write(
                    f"[{ts}] Mouse.move() called: x={x}, y={y}, duration={duration}, is_mock={is_mock}\n"
                )
        except Exception:
            pass

        if is_mock:
            result = cls._mock_input.mouse_move(x, y, duration)
            logger.debug(f"[MOCK] Mouse moved to ({x}, {y})")
        else:
            controller = cls._get_controller()
            # Debug: log controller info
            try:
                with open(debug_log_path, "a") as f:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(
                        f"[{ts}] Mouse.move() calling controller.mouse_move(), controller type: {type(controller).__name__}\n"
                    )
            except Exception:
                pass
            result = controller.mouse_move(x, y, duration)
            try:
                with open(debug_log_path, "a") as f:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(
                        f"[{ts}] Mouse.move() controller.mouse_move() returned: {result}\n"
                    )
            except Exception:
                pass
            logger.debug(f"[LIVE] Mouse moved to ({x}, {y})")

        # Emit event after successful move
        if result:
            emit_event(
                EventType.MOUSE_MOVED,
                data={
                    "x": x,
                    "y": y,
                    "duration": duration,
                    "mode": "mock" if is_mock else "live",
                },
            )

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
            controller = cls._get_controller()
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
        is_mock = MockModeManager.is_mock_mode()

        if is_mock:
            result = cls._mock_input.mouse_click(x, y, button, clicks)
            logger.debug(f"[MOCK] Mouse clicked {button.value} at ({x}, {y}) x{clicks}")
        else:
            controller = cls._get_controller()
            result = controller.mouse_click(x, y, button, clicks)
            logger.debug(f"[LIVE] Mouse clicked {button.value} at ({x}, {y}) x{clicks}")

        # Emit event after successful click
        if result:
            emit_event(
                EventType.MOUSE_CLICKED,
                data={
                    "x": x,
                    "y": y,
                    "button": button.value,
                    "clicks": clicks,
                    "click_type": "double" if clicks > 1 else "single",
                    "target_type": "coordinates",
                    "timestamp": time.time(),
                    "mode": "mock" if is_mock else "live",
                },
            )

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
            controller = cls._get_controller()
            return controller.click_at(x, y, button)

    @classmethod
    def double_click_at(
        cls, x: int, y: int, button: MouseButton = MouseButton.LEFT
    ) -> bool:
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
            controller = cls._get_controller()
            return controller.double_click_at(x, y, button)

    @classmethod
    def down(
        cls,
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
        """
        # Debug logging to file for troubleshooting
        import datetime

        debug_log_path = (
            r"C:\Users\Joshua\AppData\Local\Temp\qontinui_mouse_wrapper_debug.log"
        )
        try:
            with open(debug_log_path, "a") as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                is_mock = MockModeManager.is_mock_mode()
                f.write(
                    f"[{ts}] Mouse.down() called: x={x}, y={y}, button={button}, is_mock={is_mock}\n"
                )
        except Exception:
            pass

        if MockModeManager.is_mock_mode():
            return cls._mock_input.mouse_down(x, y, button)
        else:
            controller = cls._get_controller()
            # Debug: log controller info
            try:
                with open(debug_log_path, "a") as f:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(
                        f"[{ts}] Mouse.down() calling controller.mouse_down(), controller type: {type(controller).__name__}\n"
                    )
            except Exception:
                pass
            result = controller.mouse_down(x, y, button)
            try:
                with open(debug_log_path, "a") as f:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(
                        f"[{ts}] Mouse.down() controller.mouse_down() returned: {result}\n"
                    )
            except Exception:
                pass
            return result

    @classmethod
    def up(
        cls,
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
        """
        # Debug logging to file for troubleshooting
        import datetime

        debug_log_path = (
            r"C:\Users\Joshua\AppData\Local\Temp\qontinui_mouse_wrapper_debug.log"
        )
        try:
            with open(debug_log_path, "a") as f:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                is_mock = MockModeManager.is_mock_mode()
                f.write(
                    f"[{ts}] Mouse.up() called: x={x}, y={y}, button={button}, is_mock={is_mock}\n"
                )
        except Exception:
            pass

        if MockModeManager.is_mock_mode():
            return cls._mock_input.mouse_up(x, y, button)
        else:
            controller = cls._get_controller()
            # Debug: log controller info
            try:
                with open(debug_log_path, "a") as f:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(
                        f"[{ts}] Mouse.up() calling controller.mouse_up(), controller type: {type(controller).__name__}\n"
                    )
            except Exception:
                pass
            result = controller.mouse_up(x, y, button)
            try:
                with open(debug_log_path, "a") as f:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(
                        f"[{ts}] Mouse.up() controller.mouse_up() returned: {result}\n"
                    )
            except Exception:
                pass
            return result

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
        is_mock = MockModeManager.is_mock_mode()

        if is_mock:
            result = cls._mock_input.mouse_drag(
                start_x, start_y, end_x, end_y, button, duration
            )
            logger.debug(
                f"[MOCK] Mouse dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})"
            )
        else:
            controller = cls._get_controller()
            result = controller.mouse_drag(
                start_x, start_y, end_x, end_y, button, duration
            )
            logger.debug(
                f"[LIVE] Mouse dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})"
            )

        # Emit event after successful drag
        if result:
            emit_event(
                EventType.MOUSE_DRAGGED,
                data={
                    "from_x": start_x,
                    "from_y": start_y,
                    "to_x": end_x,
                    "to_y": end_y,
                    "button": button.value,
                    "duration": duration,
                    "mode": "mock" if is_mock else "live",
                },
            )

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
            controller = cls._get_controller()
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
            controller = cls._get_controller()
            return controller.get_mouse_position()

    @classmethod
    def reset_mock(cls) -> None:
        """Reset mock mouse state (for test cleanup).

        Only affects mock mode.
        """
        cls._mock_input.reset()
        logger.debug("Mock mouse reset")
