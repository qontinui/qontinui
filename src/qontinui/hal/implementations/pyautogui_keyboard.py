"""PyAutoGUI keyboard operations implementation."""

from ...exceptions import InputControlError
from ...logging import get_logger
from ..interfaces.keyboard_controller import IKeyboardController, Key

logger = get_logger(__name__)


class PyAutoGUIKeyboardOperations(IKeyboardController):
    """Handles keyboard input operations using PyAutoGUI.

    This implementation uses PyAutoGUI as the backend for keyboard control,
    which provides cross-platform support without needing X11 on Linux.
    """

    def __init__(self) -> None:
        """Initialize PyAutoGUI keyboard operations."""
        import pyautogui

        self._pyautogui = pyautogui
        # Disable PyAutoGUI's fail-safe (moving mouse to corner)
        pyautogui.FAILSAFE = False
        self._key_map = self._build_key_map()

    def _build_key_map(self) -> dict[Key, str]:
        """Build mapping from Key enum to PyAutoGUI key names.

        Returns:
            Mapping dictionary
        """
        return {
            # Modifier keys
            Key.ALT: "alt",
            Key.ALT_L: "altleft",
            Key.ALT_R: "altright",
            Key.SHIFT: "shift",
            Key.SHIFT_L: "shiftleft",
            Key.SHIFT_R: "shiftright",
            Key.CTRL: "ctrl",
            Key.CTRL_L: "ctrlleft",
            Key.CTRL_R: "ctrlright",
            Key.CMD: "command",
            Key.WIN: "win",
            # Navigation keys
            Key.UP: "up",
            Key.DOWN: "down",
            Key.LEFT: "left",
            Key.RIGHT: "right",
            Key.HOME: "home",
            Key.END: "end",
            Key.PAGE_UP: "pageup",
            Key.PAGE_DOWN: "pagedown",
            # Action keys
            Key.ENTER: "enter",
            Key.TAB: "tab",
            Key.SPACE: "space",
            Key.BACKSPACE: "backspace",
            Key.DELETE: "delete",
            Key.ESCAPE: "escape",
            # Function keys
            Key.F1: "f1",
            Key.F2: "f2",
            Key.F3: "f3",
            Key.F4: "f4",
            Key.F5: "f5",
            Key.F6: "f6",
            Key.F7: "f7",
            Key.F8: "f8",
            Key.F9: "f9",
            Key.F10: "f10",
            Key.F11: "f11",
            Key.F12: "f12",
        }

    def _get_pyautogui_key(self, key: str | Key) -> str:
        """Convert key to PyAutoGUI key name.

        Args:
            key: Key to convert

        Returns:
            PyAutoGUI key name
        """
        if isinstance(key, Key):
            return self._key_map.get(key, key.value)
        elif isinstance(key, str):
            # Check if it's a special key name
            key_lower = key.lower()
            for enum_key, pyautogui_key in self._key_map.items():
                if enum_key.value == key_lower:
                    return pyautogui_key
            # Return as regular character
            return key
        return str(key)

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
            pyautogui_key = self._get_pyautogui_key(key)
            self._pyautogui.press(pyautogui_key, presses=presses, interval=interval)

            logger.debug(f"Key '{key}' pressed {presses} time(s)")
            return True

        except Exception as e:
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
            pyautogui_key = self._get_pyautogui_key(key)
            self._pyautogui.keyDown(pyautogui_key)

            logger.debug(f"Key '{key}' pressed down")
            return True

        except Exception as e:
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
            pyautogui_key = self._get_pyautogui_key(key)
            self._pyautogui.keyUp(pyautogui_key)

            logger.debug(f"Key '{key}' released")
            return True

        except Exception as e:
            logger.error(f"Key up failed: {e}")
            raise InputControlError("key_up", str(e)) from e

    def type_text(self, text: str, interval: float = 0.0) -> bool:
        """Type text string.

        Handles special characters like newlines by using press() for them.
        PyAutoGUI's typewrite() can only handle printable characters, so we
        need to split the text and handle special characters separately.

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
                "\n": "enter",
                "\r": "enter",
                "\t": "tab",
            }

            # Process text character by character to handle special chars
            buffer = ""
            for char in text:
                if char in special_chars:
                    # First, type any buffered regular text
                    if buffer:
                        self._pyautogui.typewrite(buffer, interval=interval)
                        buffer = ""
                    # Then press the special key
                    self._pyautogui.press(special_chars[char])
                    if interval > 0:
                        import time

                        time.sleep(interval)
                else:
                    buffer += char

            # Type any remaining buffered text
            if buffer:
                self._pyautogui.typewrite(buffer, interval=interval)

            logger.debug(f"Typed text: '{text[:20]}...' ({len(text)} chars)")
            return True

        except Exception as e:
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
            # Convert all keys to PyAutoGUI format
            pyautogui_keys = [self._get_pyautogui_key(k) for k in keys]
            self._pyautogui.hotkey(*pyautogui_keys)

            logger.debug(f"Hotkey pressed: {'+'.join(str(k) for k in keys)}")
            return True

        except Exception as e:
            logger.error(f"Hotkey failed: {e}")
            raise InputControlError("hotkey", str(e)) from e

    def is_key_pressed(self, key: str | Key) -> bool:
        """Check if key is currently pressed.

        Note: PyAutoGUI doesn't provide direct key state checking,
        so this returns False.

        Args:
            key: Key to check

        Returns:
            False (not supported)
        """
        logger.debug("Key state checking not supported by PyAutoGUI")
        return False
