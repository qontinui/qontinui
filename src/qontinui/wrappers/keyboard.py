"""Keyboard wrapper for mock/live automation switching.

Based on Brobot's wrapper pattern - provides stable API that routes
to mock or live implementation based on execution mode.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..hal.container import HALContainer
    from ..hal.interfaces.input_controller import IInputController

from ..hal.interfaces.input_controller import Key
from ..mock.mock_input import MockInput
from ..mock.mock_mode_manager import MockModeManager
from ..reporting import EventType, emit_event

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
        hal = initialize_hal()
        keyboard = Keyboard(hal)
        keyboard.type("Hello World")
        keyboard.press(Key.ENTER)
    """

    _mock_input = MockInput()
    _controller: IInputController | None = None
    _controller_lock = threading.Lock()

    def __init__(self, hal: HALContainer | None = None) -> None:
        """Initialize keyboard wrapper with optional HAL container.

        Args:
            hal: HAL container providing input controller. If None,
                instance methods will fall back to class methods.
        """
        self._hal = hal
        self._instance_controller = hal.input_controller if hal else None  # type: ignore[attr-defined]

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

    def _get_active_controller(self) -> IInputController:
        """Get the active controller (instance or class-level)."""
        if self._instance_controller:
            return self._instance_controller  # type: ignore[no-any-return]
        return self._get_controller()  # type: ignore[no-any-return]

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
        logger.debug(f"About to type text: '{text}', interval={interval}")

        is_mock = MockModeManager.is_mock_mode()
        logger.debug(f"is_mock={is_mock}")

        if is_mock:
            result = cls._mock_input.type_text(text, interval)
            logger.debug(f"[MOCK] Typed text: '{text}'")
        else:
            logger.debug("Getting input controller from lazy init")
            controller = cls._get_controller()
            logger.debug(f"Got controller: {controller}")
            logger.debug(f"About to call controller.type_text('{text}', {interval})")
            result = controller.type_text(text, interval)
            logger.debug(f"controller.type_text returned: {result}")
            logger.debug(f"[LIVE] Typed text: '{text}'")

        # Emit event after successful typing
        if result:
            emit_event(
                EventType.TEXT_TYPED,
                data={
                    "text": text,
                    "length": len(text),
                    "interval": interval,
                    "mode": "mock" if is_mock else "live",
                },
            )

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
            controller = cls._get_controller()
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
            controller = cls._get_controller()
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
            controller = cls._get_controller()
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
            controller = cls._get_controller()
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
            controller = cls._get_controller()
            return controller.is_key_pressed(key)

    @classmethod
    def reset_mock(cls) -> None:
        """Reset mock keyboard state (for test cleanup).

        Only affects mock mode.
        """
        cls._mock_input.reset()
        logger.debug("Mock keyboard reset")
