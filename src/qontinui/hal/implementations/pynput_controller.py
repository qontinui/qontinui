"""Pynput-based input controller implementation."""

from typing import Any, cast

from pynput import keyboard, mouse
from pynput.keyboard import Key as PynputKey
from pynput.mouse import Button as PynputButton

from ...logging import get_logger
from ...wrappers import Time
from ..config import HALConfig
from ..interfaces.input_controller import IInputController, Key, MouseButton, MousePosition

logger = get_logger(__name__)


class PynputController(IInputController):
    """Input controller implementation using Pynput.

    Pynput provides direct system-level input control without GUI automation
    dependencies, offering better performance and reliability.
    """

    def __init__(self, config: HALConfig | None = None):
        """Initialize Pynput controller.

        Args:
            config: HAL configuration
        """
        self.config = config or HALConfig()
        self.mouse_controller = mouse.Controller()
        self.keyboard_controller = keyboard.Controller()

        # Button mapping
        self._button_map = {
            MouseButton.LEFT: PynputButton.left,
            MouseButton.RIGHT: PynputButton.right,
            MouseButton.MIDDLE: PynputButton.middle,
            "left": PynputButton.left,
            "right": PynputButton.right,
            "middle": PynputButton.middle,
        }

        # Key mapping for special keys
        self._key_map = self._build_key_map()

        logger.info("pynput_controller_initialized")

    def _build_key_map(self) -> dict[Key, Any]:
        """Build mapping from Key enum to Pynput keys."""
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

    def _get_pynput_button(self, button: MouseButton | str) -> PynputButton:
        """Convert button to Pynput button.

        Args:
            button: Button to convert

        Returns:
            Pynput button
        """
        if isinstance(button, str):
            button = button.lower()
        return self._button_map.get(button, PynputButton.left)

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

    # Mouse operations

    def mouse_move(self, x: int, y: int, duration: float = 0.0) -> bool:
        """Move mouse to absolute position.

        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Movement duration in seconds (0 for instant)

        Returns:
            True if successful
        """
        try:
            if duration > 0:
                # Smooth movement
                start_x, start_y = self.mouse_controller.position
                steps = max(int(duration * 60), 1)  # 60 FPS
                delay = duration / steps

                for i in range(steps + 1):
                    progress = i / steps
                    current_x = int(start_x + (x - start_x) * progress)
                    current_y = int(start_y + (y - start_y) * progress)
                    self.mouse_controller.position = (current_x, current_y)
                    if i < steps:
                        Time.wait(delay)
            else:
                # Instant movement
                self.mouse_controller.position = (x, y)

            logger.debug(f"Mouse moved to ({x}, {y})")
            return True

        except Exception as e:
            logger.error(f"Mouse move failed: {e}")
            return False

    def mouse_move_relative(self, dx: int, dy: int, duration: float = 0.0) -> bool:
        """Move mouse relative to current position.

        Args:
            dx: X offset
            dy: Y offset
            duration: Movement duration in seconds

        Returns:
            True if successful
        """
        try:
            current_x, current_y = self.mouse_controller.position
            return self.mouse_move(current_x + dx, current_y + dy, duration)

        except Exception as e:
            logger.error(f"Relative mouse move failed: {e}")
            return False

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
        """
        try:
            # Move to position if specified
            if x is not None and y is not None:
                self.mouse_move(x, y)

            pynput_button = self._get_pynput_button(button)

            for i in range(clicks):
                self.mouse_controller.click(pynput_button)
                if i < clicks - 1 and interval > 0:
                    Time.wait(interval)

            logger.debug(f"Mouse clicked {clicks} time(s) with {button}")
            return True

        except Exception as e:
            logger.error(f"Mouse click failed: {e}")
            return False

    def mouse_down(
        self, x: int | None = None, y: int | None = None, button: MouseButton = MouseButton.LEFT
    ) -> bool:
        """Press and hold mouse button.

        Args:
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)
            button: Mouse button to press

        Returns:
            True if successful
        """
        try:
            # Move to position if specified
            if x is not None and y is not None:
                self.mouse_move(x, y)

            pynput_button = self._get_pynput_button(button)
            self.mouse_controller.press(pynput_button)

            logger.debug(f"Mouse button {button} pressed")
            return True

        except Exception as e:
            logger.error(f"Mouse down failed: {e}")
            return False

    def mouse_up(
        self, x: int | None = None, y: int | None = None, button: MouseButton = MouseButton.LEFT
    ) -> bool:
        """Release mouse button.

        Args:
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)
            button: Mouse button to release

        Returns:
            True if successful
        """
        try:
            # Move to position if specified
            if x is not None and y is not None:
                self.mouse_move(x, y)

            pynput_button = self._get_pynput_button(button)
            self.mouse_controller.release(pynput_button)

            logger.debug(f"Mouse button {button} released")
            return True

        except Exception as e:
            logger.error(f"Mouse up failed: {e}")
            return False

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
        """
        try:
            # Move to start position
            self.mouse_move(start_x, start_y)

            # Press button
            pynput_button = self._get_pynput_button(button)
            self.mouse_controller.press(pynput_button)

            # Move to end position
            Time.wait(0.1)  # Small delay before drag
            self.mouse_move(end_x, end_y, duration)

            # Release button
            Time.wait(0.1)  # Small delay before release
            self.mouse_controller.release(pynput_button)

            logger.debug(f"Mouse dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})")
            return True

        except Exception as e:
            logger.error(f"Mouse drag failed: {e}")
            # Try to release button if pressed
            try:
                self.mouse_controller.release(self._get_pynput_button(button))
            except Exception:
                pass
            return False

    def mouse_scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> bool:
        """Scroll mouse wheel.

        Args:
            clicks: Number of scroll clicks (positive=up, negative=down)
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)

        Returns:
            True if successful
        """
        try:
            # Move to position if specified
            if x is not None and y is not None:
                self.mouse_move(x, y)

            # Scroll (negative for down, positive for up)
            self.mouse_controller.scroll(0, clicks)

            logger.debug(f"Mouse scrolled {clicks} clicks")
            return True

        except Exception as e:
            logger.error(f"Mouse scroll failed: {e}")
            return False

    def get_mouse_position(self) -> MousePosition:
        """Get current mouse position.

        Returns:
            Current mouse position
        """
        try:
            x, y = self.mouse_controller.position
            return MousePosition(x=int(x), y=int(y))
        except Exception as e:
            logger.error(f"Get mouse position failed: {e}")
            return MousePosition(x=0, y=0)

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

    def double_click_at(self, x: int, y: int, button: MouseButton = MouseButton.LEFT) -> bool:
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
        return self.mouse_drag(start_x, start_y, end_x, end_y, MouseButton.LEFT, duration)

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

    # Keyboard operations

    def key_press(self, key: str | Key, presses: int = 1, interval: float = 0.0) -> bool:
        """Press key (down and up).

        Args:
            key: Key to press (string or Key enum)
            presses: Number of key presses
            interval: Interval between presses in seconds

        Returns:
            True if successful
        """
        try:
            pynput_key = self._get_pynput_key(key)

            for i in range(presses):
                self.keyboard_controller.press(pynput_key)
                self.keyboard_controller.release(pynput_key)
                if i < presses - 1 and interval > 0:
                    Time.wait(interval)

            logger.debug(f"Key '{key}' pressed {presses} time(s)")
            return True

        except Exception as e:
            logger.error(f"Key press failed: {e}")
            return False

    def key_down(self, key: str | Key) -> bool:
        """Press and hold key.

        Args:
            key: Key to press down

        Returns:
            True if successful
        """
        try:
            pynput_key = self._get_pynput_key(key)
            self.keyboard_controller.press(pynput_key)

            logger.debug(f"Key '{key}' pressed down")
            return True

        except Exception as e:
            logger.error(f"Key down failed: {e}")
            return False

    def key_up(self, key: str | Key) -> bool:
        """Release key.

        Args:
            key: Key to release

        Returns:
            True if successful
        """
        try:
            pynput_key = self._get_pynput_key(key)
            self.keyboard_controller.release(pynput_key)

            logger.debug(f"Key '{key}' released")
            return True

        except Exception as e:
            logger.error(f"Key up failed: {e}")
            return False

    def type_text(self, text: str, interval: float = 0.0) -> bool:
        """Type text string.

        Args:
            text: Text to type
            interval: Interval between characters in seconds

        Returns:
            True if successful
        """
        try:
            if interval > 0:
                for char in text:
                    self.keyboard_controller.type(char)
                    Time.wait(interval)
            else:
                self.keyboard_controller.type(text)

            logger.debug(f"Typed text: '{text[:20]}...' ({len(text)} chars)")
            return True

        except Exception as e:
            logger.error(f"Type text failed: {e}")
            return False

    def hotkey(self, *keys: str | Key) -> bool:
        """Press key combination.

        Args:
            *keys: Keys to press together (e.g., 'ctrl', 'a')

        Returns:
            True if successful
        """
        try:
            # Convert all keys
            pynput_keys = [self._get_pynput_key(k) for k in keys]

            # Press all keys
            for key in pynput_keys:
                self.keyboard_controller.press(key)

            # Small delay
            Time.wait(0.05)

            # Release all keys in reverse order
            for key in reversed(pynput_keys):
                self.keyboard_controller.release(key)

            logger.debug(f"Hotkey pressed: {'+'.join(str(k) for k in keys)}")
            return True

        except Exception as e:
            logger.error(f"Hotkey failed: {e}")
            # Try to release any pressed keys
            for orig_key in keys:
                try:
                    # Type annotation to clarify for mypy
                    qontinui_key: str | Key = orig_key
                    pynput_key: PynputKey | str = self._get_pynput_key(qontinui_key)
                    self.keyboard_controller.release(pynput_key)
                except Exception:
                    pass
            return False

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

        except Exception as e:
            logger.error(f"Key state check failed: {e}")
            return False
