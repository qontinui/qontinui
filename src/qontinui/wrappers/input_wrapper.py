"""InputWrapper - Routes mouse/keyboard operations to mock or real implementations (Brobot pattern).

This wrapper provides the routing layer for input operations (mouse and keyboard),
delegating to either MockMouse/MockKeyboard (no-op tracking) or HAL implementations
(real input automation) based on ExecutionMode.

Architecture:
    FluentActions/Click/Type (high-level)
      ↓
    MouseWrapper/KeyboardWrapper (this layer) ← Routes based on ExecutionMode
      ↓
    ├─ if mock → MockMouse/MockKeyboard → No-op (just logging/tracking)
    └─ if real → HAL Layer → PyAutoGUI/pynput → Real input
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import BaseWrapper

if TYPE_CHECKING:
    from ..hal.interfaces.input_controller import MouseButton, MousePosition

logger = logging.getLogger(__name__)


class MouseWrapper(BaseWrapper):
    """Wrapper for mouse operations.

    Routes mouse operations to either mock or real implementations based on
    ExecutionMode. This follows the Brobot pattern where high-level code is
    agnostic to whether it's running in mock or real mode.

    Example:
        # Initialize wrapper
        wrapper = MouseWrapper()

        # Click (automatically routed to mock or real)
        wrapper.click(100, 200)

        # High-level code doesn't know or care whether this:
        # - MockMouse.click() → Just logs the action
        # - HAL PyAutoGUI.click() → Actually moves/clicks mouse

    Attributes:
        mock_mouse: MockMouse instance for no-op tracking
        hal_input: Input controller for real mode
    """

    def __init__(self) -> None:
        """Initialize MouseWrapper.

        Sets up both mock and real implementations. The actual implementation
        used is determined at runtime based on ExecutionMode.
        """
        super().__init__()

        # Lazy initialization
        self._mock_mouse = None
        self._hal_input = None

        logger.debug("MouseWrapper initialized")

    @property
    def mock_mouse(self):
        """Get MockMouse instance (lazy initialization).

        Returns:
            MockMouse instance
        """
        if self._mock_mouse is None:
            from ..mock.mock_mouse import MockMouse

            self._mock_mouse = MockMouse()
            logger.debug("MockMouse initialized")
        return self._mock_mouse

    @property
    def hal_input(self):
        """Get HAL input controller (lazy initialization).

        Returns:
            IInputController implementation
        """
        if self._hal_input is None:
            from ..hal.factory import HALFactory

            self._hal_input = HALFactory.get_input_controller()
            logger.debug("HAL input controller initialized")
        return self._hal_input

    def move(self, x: int, y: int, duration: float = 0.0) -> bool:
        """Move mouse to position.

        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Movement duration in seconds

        Returns:
            True if successful
        """
        if self.is_mock_mode():
            logger.debug(f"MouseWrapper.move (MOCK): ({x}, {y})")
            return self.mock_mouse.move(x, y, duration)  # type: ignore[no-any-return]
        else:
            logger.debug(f"MouseWrapper.move (REAL): ({x}, {y})")
            return self.hal_input.mouse_move(x, y, duration)  # type: ignore[no-any-return]

    def click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: MouseButton | None = None,
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
        if self.is_mock_mode():
            logger.debug(f"MouseWrapper.click (MOCK): ({x}, {y}), clicks={clicks}")
            return self.mock_mouse.click(x, y, button, clicks)  # type: ignore[no-any-return]
        else:
            logger.debug(f"MouseWrapper.click (REAL): ({x}, {y}), clicks={clicks}")
            from ..hal.interfaces.input_controller import MouseButton as MB

            btn = button if button else MB.LEFT
            return self.hal_input.mouse_click(x, y, btn, clicks)  # type: ignore[no-any-return]

    def double_click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: MouseButton | None = None,
    ) -> bool:
        """Double click mouse button.

        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button to click

        Returns:
            True if successful
        """
        return self.click(x, y, button, clicks=2)

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float = 0.5,
    ) -> bool:
        """Drag mouse from start to end.

        Args:
            start_x: Start X coordinate
            start_y: Start Y coordinate
            end_x: End X coordinate
            end_y: End Y coordinate
            duration: Drag duration

        Returns:
            True if successful
        """
        if self.is_mock_mode():
            logger.debug(f"MouseWrapper.drag (MOCK): ({start_x},{start_y}) → ({end_x},{end_y})")
            return self.mock_mouse.drag(start_x, start_y, end_x, end_y, duration)  # type: ignore[no-any-return]
        else:
            logger.debug(f"MouseWrapper.drag (REAL): ({start_x},{start_y}) → ({end_x},{end_y})")
            return self.hal_input.drag(start_x, start_y, end_x, end_y, duration)  # type: ignore[no-any-return]

    def scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> bool:
        """Scroll mouse wheel.

        Args:
            clicks: Number of clicks (positive=up, negative=down)
            x: X coordinate (optional)
            y: Y coordinate (optional)

        Returns:
            True if successful
        """
        if self.is_mock_mode():
            logger.debug(f"MouseWrapper.scroll (MOCK): clicks={clicks}")
            return self.mock_mouse.scroll(clicks, x, y)  # type: ignore[no-any-return]
        else:
            logger.debug(f"MouseWrapper.scroll (REAL): clicks={clicks}")
            return self.hal_input.scroll(clicks, x, y)  # type: ignore[no-any-return]

    def get_position(self) -> MousePosition:
        """Get current mouse position.

        Returns:
            Current mouse position
        """
        if self.is_mock_mode():
            logger.debug("MouseWrapper.get_position (MOCK)")
            return self.mock_mouse.get_position()  # type: ignore[no-any-return]
        else:
            logger.debug("MouseWrapper.get_position (REAL)")
            return self.hal_input.get_mouse_position()  # type: ignore[no-any-return]


class KeyboardWrapper(BaseWrapper):
    """Wrapper for keyboard operations.

    Routes keyboard operations to either mock or real implementations based on
    ExecutionMode. This follows the Brobot pattern where high-level code is
    agnostic to whether it's running in mock or real mode.

    Example:
        # Initialize wrapper
        wrapper = KeyboardWrapper()

        # Type text (automatically routed to mock or real)
        wrapper.type_text("Hello World")

        # High-level code doesn't know or care whether this:
        # - MockKeyboard.type_text() → Just logs the action
        # - HAL PyAutoGUI.type_text() → Actually types the text

    Attributes:
        mock_keyboard: MockKeyboard instance for no-op tracking
        hal_input: Input controller for real mode
    """

    def __init__(self) -> None:
        """Initialize KeyboardWrapper.

        Sets up both mock and real implementations. The actual implementation
        used is determined at runtime based on ExecutionMode.
        """
        super().__init__()

        # Lazy initialization
        self._mock_keyboard = None
        self._hal_input = None

        logger.debug("KeyboardWrapper initialized")

    @property
    def mock_keyboard(self):
        """Get MockKeyboard instance (lazy initialization).

        Returns:
            MockKeyboard instance
        """
        if self._mock_keyboard is None:
            from ..mock.mock_keyboard import MockKeyboard

            self._mock_keyboard = MockKeyboard()
            logger.debug("MockKeyboard initialized")
        return self._mock_keyboard

    @property
    def hal_input(self):
        """Get HAL input controller (lazy initialization).

        Returns:
            IInputController implementation
        """
        if self._hal_input is None:
            from ..hal.factory import HALFactory

            self._hal_input = HALFactory.get_input_controller()
            logger.debug("HAL input controller initialized")
        return self._hal_input

    def type_text(self, text: str, interval: float = 0.0) -> bool:
        """Type text string.

        Args:
            text: Text to type
            interval: Interval between characters

        Returns:
            True if successful
        """
        if self.is_mock_mode():
            logger.debug(f"KeyboardWrapper.type_text (MOCK): '{text[:50]}...'")
            return self.mock_keyboard.type_text(text, interval)  # type: ignore[no-any-return]
        else:
            logger.debug(f"KeyboardWrapper.type_text (REAL): '{text[:50]}...'")
            return self.hal_input.type_text(text, interval)  # type: ignore[no-any-return]

    def press(self, key: str, presses: int = 1) -> bool:
        """Press key.

        Args:
            key: Key to press
            presses: Number of presses

        Returns:
            True if successful
        """
        if self.is_mock_mode():
            logger.debug(f"KeyboardWrapper.press (MOCK): {key} x{presses}")
            return self.mock_keyboard.press(key, presses)  # type: ignore[no-any-return]
        else:
            logger.debug(f"KeyboardWrapper.press (REAL): {key} x{presses}")
            return self.hal_input.key_press(key, presses)  # type: ignore[no-any-return]

    def hotkey(self, *keys: str) -> bool:
        """Press key combination.

        Args:
            *keys: Keys to press together

        Returns:
            True if successful
        """
        if self.is_mock_mode():
            logger.debug(f"KeyboardWrapper.hotkey (MOCK): {'+'.join(keys)}")
            return self.mock_keyboard.hotkey(*keys)  # type: ignore[no-any-return]
        else:
            logger.debug(f"KeyboardWrapper.hotkey (REAL): {'+'.join(keys)}")
            return self.hal_input.hotkey(*keys)  # type: ignore[no-any-return]

    def key_down(self, key: str) -> bool:
        """Press and hold key.

        Args:
            key: Key to press down

        Returns:
            True if successful
        """
        if self.is_mock_mode():
            logger.debug(f"KeyboardWrapper.key_down (MOCK): {key}")
            return self.mock_keyboard.key_down(key)  # type: ignore[no-any-return]
        else:
            logger.debug(f"KeyboardWrapper.key_down (REAL): {key}")
            return self.hal_input.key_down(key)  # type: ignore[no-any-return]

    def key_up(self, key: str) -> bool:
        """Release key.

        Args:
            key: Key to release

        Returns:
            True if successful
        """
        if self.is_mock_mode():
            logger.debug(f"KeyboardWrapper.key_up (MOCK): {key}")
            return self.mock_keyboard.key_up(key)  # type: ignore[no-any-return]
        else:
            logger.debug(f"KeyboardWrapper.key_up (REAL): {key}")
            return self.hal_input.key_up(key)  # type: ignore[no-any-return]
