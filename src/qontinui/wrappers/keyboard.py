"""Keyboard wrapper for mock/live automation switching.

Based on Brobot's wrapper pattern - provides stable API that routes
to mock or live implementation based on execution mode.
"""

import logging

from ..hal.factory import HALFactory
from ..hal.interfaces.input_controller import Key
from ..mock.mock_mode_manager import MockModeManager
from ..mock.mock_input import MockInput

logger = logging.getLogger(__name__)


class Keyboard:
    """Wrapper for keyboard operations that routes to mock or live implementation.

    This wrapper provides a stable API for keyboard operations while allowing
    the underlying implementation to switch between mock and live modes.

    In mock mode:
    - All operations complete instantly (no real keyboard input)
    - State tracked for testing verification

    In live mode:
    - Uses HAL input controller for real keyboard operations
    - Actual system keyboard control

    Example usage:
        Keyboard.type("Hello World")  # Type text (real or simulated)
        Keyboard.press(Key.ENTER)  # Press key
        Keyboard.hotkey(Key.CTRL, "c")  # Key combination
    """

    _mock_input = MockInput()

    @classmethod
    def type(cls, text: str, interval: float = 0.0) -> bool:
        """Type text string.

        In mock mode: Records text instantly
        In live mode: Types actual text

        Args:
            text: Text to type
            interval: Interval between characters in seconds

        Returns:
            True if successful
        """
        if MockModeManager.is_mock_mode():
            result = cls._mock_input.type_text(text, interval)
            logger.debug(f"[MOCK] Typed text: '{text}'")
            return result
        else:
            controller = HALFactory.get_input_controller()
            result = controller.type_text(text, interval)
            logger.debug(f"[LIVE] Typed text: '{text}'")
            return result

    @classmethod
    def press(cls, key: str | Key, presses: int = 1, interval: float = 0.0) -> bool:
        """Press key (down and up).

        Args:
            key: Key to press (string or Key enum)
            presses: Number of key presses
            interval: Interval between presses in seconds

        Returns:
            True if successful
        """
        if MockModeManager.is_mock_mode():
            result = cls._mock_input.key_press(key, presses, interval)
            key_str = key.value if isinstance(key, Key) else key
            logger.debug(f"[MOCK] Pressed key: '{key_str}' x{presses}")
            return result
        else:
            controller = HALFactory.get_input_controller()
            result = controller.key_press(key, presses, interval)
            key_str = key.value if isinstance(key, Key) else key
            logger.debug(f"[LIVE] Pressed key: '{key_str}' x{presses}")
            return result

    @classmethod
    def down(cls, key: str | Key) -> bool:
        """Press and hold key.

        Args:
            key: Key to press down

        Returns:
            True if successful
        """
        if MockModeManager.is_mock_mode():
            result = cls._mock_input.key_down(key)
            key_str = key.value if isinstance(key, Key) else key
            logger.debug(f"[MOCK] Key down: '{key_str}'")
            return result
        else:
            controller = HALFactory.get_input_controller()
            result = controller.key_down(key)
            key_str = key.value if isinstance(key, Key) else key
            logger.debug(f"[LIVE] Key down: '{key_str}'")
            return result

    @classmethod
    def up(cls, key: str | Key) -> bool:
        """Release key.

        Args:
            key: Key to release

        Returns:
            True if successful
        """
        if MockModeManager.is_mock_mode():
            result = cls._mock_input.key_up(key)
            key_str = key.value if isinstance(key, Key) else key
            logger.debug(f"[MOCK] Key up: '{key_str}'")
            return result
        else:
            controller = HALFactory.get_input_controller()
            result = controller.key_up(key)
            key_str = key.value if isinstance(key, Key) else key
            logger.debug(f"[LIVE] Key up: '{key_str}'")
            return result

    @classmethod
    def hotkey(cls, *keys: str | Key) -> bool:
        """Press key combination.

        Args:
            *keys: Keys to press together (e.g., Key.CTRL, 'c')

        Returns:
            True if successful
        """
        if MockModeManager.is_mock_mode():
            result = cls._mock_input.hotkey(*keys)
            key_strs = [k.value if isinstance(k, Key) else k for k in keys]
            logger.debug(f"[MOCK] Hotkey: {'+'.join(key_strs)}")
            return result
        else:
            controller = HALFactory.get_input_controller()
            result = controller.hotkey(*keys)
            key_strs = [k.value if isinstance(k, Key) else k for k in keys]
            logger.debug(f"[LIVE] Hotkey: {'+'.join(key_strs)}")
            return result

    @classmethod
    def is_pressed(cls, key: str | Key) -> bool:
        """Check if key is currently pressed.

        Args:
            key: Key to check

        Returns:
            True if key is pressed
        """
        if MockModeManager.is_mock_mode():
            return cls._mock_input.is_key_pressed(key)
        else:
            controller = HALFactory.get_input_controller()
            return controller.is_key_pressed(key)

    @classmethod
    def reset_mock(cls) -> None:
        """Reset mock keyboard state (for test cleanup).

        Only affects mock mode.
        """
        cls._mock_input.reset()
        logger.debug("Mock keyboard reset")
