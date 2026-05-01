"""Keyboard input operations."""

from typing import Any, cast

from pynput import keyboard
from pynput.keyboard import Key as PynputKey

from ...exceptions import InputControlError
from ...logging import get_logger
from ...wrappers.time_wrapper import TimeWrapper
from ..interfaces.keyboard_controller import IKeyboardController, Key

logger = get_logger(__name__)


class KeyboardOperations(IKeyboardController):
    """Handles all keyboard input operations.

    This class encapsulates keyboard-specific functionality, separating
    it from mouse operations and following the Single Responsibility Principle.
    """

    def __init__(self, keyboard_controller: keyboard.Controller) -> None:
        """Initialize keyboard operations.

        Args:
            keyboard_controller: Pynput keyboard controller instance
        """
        self._keyboard = keyboard_controller
        self._key_map = self._build_key_map()

    def _build_key_map(self) -> dict[Key, Any]:
        """Build mapping from Key enum to Pynput keys.

        Returns:
            Mapping dictionary
        """
        return {
            # Modifier keys
            Key.ALT: PynputKey.alt,
            Key.ALT_L: PynputKey.alt_l,
            Key.ALT_R: PynputKey.alt_r,
            Key.SHIFT: PynputKey.shift,
            Key.SHIFT_L: PynputKey.shift_l,
            Key.SHIFT_R: PynputKey.shift_r,
            Key.CTRL: PynputKey.ctrl,
            Key.CTRL_L: PynputKey.ctrl_l,
            Key.CTRL_R: PynputKey.ctrl_r,
            Key.CMD: PynputKey.cmd,
            Key.WIN: PynputKey.cmd,  # Windows key maps to cmd
            # Navigation keys
            Key.UP: PynputKey.up,
            Key.DOWN: PynputKey.down,
            Key.LEFT: PynputKey.left,
            Key.RIGHT: PynputKey.right,
            Key.HOME: PynputKey.home,
            Key.END: PynputKey.end,
            Key.PAGE_UP: PynputKey.page_up,
            Key.PAGE_DOWN: PynputKey.page_down,
            # Action keys
            Key.ENTER: PynputKey.enter,
            Key.TAB: PynputKey.tab,
            Key.SPACE: PynputKey.space,
            Key.BACKSPACE: PynputKey.backspace,
            Key.DELETE: PynputKey.delete,
            Key.ESCAPE: PynputKey.esc,
            # Function keys
            Key.F1: PynputKey.f1,
            Key.F2: PynputKey.f2,
            Key.F3: PynputKey.f3,
            Key.F4: PynputKey.f4,
            Key.F5: PynputKey.f5,
            Key.F6: PynputKey.f6,
            Key.F7: PynputKey.f7,
            Key.F8: PynputKey.f8,
            Key.F9: PynputKey.f9,
            Key.F10: PynputKey.f10,
            Key.F11: PynputKey.f11,
            Key.F12: PynputKey.f12,
        }

    def _get_pynput_key(self, key: str | Key) -> PynputKey | str:
        """Convert key to Pynput key.

        Args:
            key: Key to convert

        Returns:
            Pynput key or string
        """
        if isinstance(key, Key):
            return cast(PynputKey | str, self._key_map.get(key, str(key.value)))
        elif isinstance(key, str):
            # Check if it's a special key name
            key_lower = key.lower()
            for enum_key, pynput_key in self._key_map.items():
                if enum_key.value == key_lower:  # type: ignore[attr-defined]
                    return cast(PynputKey | str, pynput_key)
            # Return as regular character
            return key
        return key

    def key_press(
        self, key: str | Key, presses: int = 1, interval: float = 0.0
    ) -> bool:
        """Press key (down and up).

        Args:
            key: Key to press (string or Key enum)
            presses: Number of key presses
            interval: Interval between presses in seconds

        Returns:
            True if successful

        Raises:
            InputControlError: If key press fails
        """
        try:
            pynput_key = self._get_pynput_key(key)

            for i in range(presses):
                self._keyboard.press(pynput_key)
                self._keyboard.release(pynput_key)
                if i < presses - 1 and interval > 0:
                    TimeWrapper().wait(seconds=interval)

            logger.debug(f"Key '{key}' pressed {presses} time(s)")
            return True

        except (OSError, RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Key press failed: {e}")
            raise InputControlError("key_press", str(e)) from e

    def key_down(self, key: str | Key) -> bool:
        """Press and hold key.

        Args:
            key: Key to press down

        Returns:
            True if successful

        Raises:
            InputControlError: If key down fails
        """
        try:
            pynput_key = self._get_pynput_key(key)
            self._keyboard.press(pynput_key)

            logger.debug(f"Key '{key}' pressed down")
            return True

        except (OSError, RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Key down failed: {e}")
            raise InputControlError("key_down", str(e)) from e

    def key_up(self, key: str | Key) -> bool:
        """Release key.

        Args:
            key: Key to release

        Returns:
            True if successful

        Raises:
            InputControlError: If key up fails
        """
        try:
            pynput_key = self._get_pynput_key(key)
            self._keyboard.release(pynput_key)

            logger.debug(f"Key '{key}' released")
            return True

        except (OSError, RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Key up failed: {e}")
            raise InputControlError("key_up", str(e)) from e

    def type_text(self, text: str, interval: float = 0.0) -> bool:
        """Type text string.

        Handles special characters like newlines by using key_press() for them.
        Pynput's type() method may not properly simulate Enter key for \n,
        so we handle special characters explicitly.

        Args:
            text: Text to type
            interval: Interval between characters in seconds

        Returns:
            True if successful

        Raises:
            InputControlError: If type text fails
        """
        try:
            # Map of special characters to their key names
            special_chars = {
                "\n": Key.ENTER,
                "\r": Key.ENTER,
                "\t": Key.TAB,
            }

            # Process text character by character to handle special chars
            buffer = ""
            for char in text:
                if char in special_chars:
                    # First, type any buffered regular text
                    if buffer:
                        if interval > 0:
                            for c in buffer:
                                self._keyboard.type(c)
                                TimeWrapper().wait(seconds=interval)
                        else:
                            self._keyboard.type(buffer)
                        buffer = ""
                    # Then press the special key
                    pynput_key = self._get_pynput_key(special_chars[char])
                    self._keyboard.press(pynput_key)
                    self._keyboard.release(pynput_key)
                    if interval > 0:
                        TimeWrapper().wait(seconds=interval)
                else:
                    buffer += char

            # Type any remaining buffered text
            if buffer:
                if interval > 0:
                    for c in buffer:
                        self._keyboard.type(c)
                        TimeWrapper().wait(seconds=interval)
                else:
                    self._keyboard.type(buffer)

            logger.debug(f"Typed text: '{text[:20]}...' ({len(text)} chars)")
            return True

        except (OSError, RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Type text failed: {e}")
            raise InputControlError("type_text", str(e)) from e

    def hotkey(self, *keys: str | Key) -> bool:
        """Press key combination.

        Args:
            *keys: Keys to press together (e.g., 'ctrl', 'a')

        Returns:
            True if successful

        Raises:
            InputControlError: If hotkey fails
        """
        try:
            # Convert all keys
            pynput_keys = [self._get_pynput_key(k) for k in keys]

            # Press all keys
            for key in pynput_keys:
                self._keyboard.press(key)

            # Small delay
            TimeWrapper().wait(seconds=0.05)

            # Release all keys in reverse order
            for key in reversed(pynput_keys):
                self._keyboard.release(key)

            logger.debug(f"Hotkey pressed: {'+'.join(str(k) for k in keys)}")
            return True

        except (OSError, RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Hotkey failed: {e}")
            # Try to release any pressed keys
            for orig_key in keys:
                try:
                    # Type annotation to clarify for mypy
                    qontinui_key: str | Key = orig_key
                    pynput_key: PynputKey | str = self._get_pynput_key(qontinui_key)
                    self._keyboard.release(pynput_key)
                except (OSError, RuntimeError):
                    # OK to ignore release errors in error handler
                    pass
            raise InputControlError("hotkey", str(e)) from e

    def is_key_pressed(self, key: str | Key) -> bool:
        """Check if key is currently pressed.

        Note: Pynput doesn't provide direct key state checking,
        so this is a best-effort implementation.

        Args:
            key: Key to check

        Returns:
            True if key is pressed (not fully reliable)
        """
        try:
            # This is a limitation of pynput - it doesn't provide
            # a way to check current key state without listening
            logger.debug("Key state checking not fully supported by pynput")
            return False

        except (OSError, RuntimeError, ValueError, TypeError) as e:
            logger.error(f"Key state check failed: {e}")
            return False
