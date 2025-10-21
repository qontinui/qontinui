"""Mock input controller for testing without actual input operations.

Based on Brobot's mock pattern - simulates input operations instantly.
"""

import logging
from datetime import datetime

from ..hal.interfaces.input_controller import IInputController, Key, MouseButton, MousePosition

logger = logging.getLogger(__name__)


class MockInput(IInputController):
    """Mock implementation of input controller for testing.

    All operations complete instantly without actual mouse/keyboard control.
    Tracks state for verification in tests.
    """

    def __init__(self):
        """Initialize mock input controller."""
        self._mouse_position = MousePosition(x=0, y=0)
        self._pressed_keys: set[str] = set()
        self._action_history: list[dict] = []
        logger.debug("MockInput initialized")

    def _record_action(self, action_type: str, **kwargs) -> None:
        """Record action for history tracking."""
        self._action_history.append({
            "type": action_type,
            "timestamp": datetime.now(),
            **kwargs
        })

    # Mouse operations

    def mouse_move(self, x: int, y: int, duration: float = 0.0) -> bool:
        """Mock move mouse to absolute position (instant)."""
        logger.debug(f"[MOCK] Mouse move to ({x}, {y})")
        self._mouse_position = MousePosition(x=x, y=y)
        self._record_action("mouse_move", x=x, y=y, duration=duration)
        return True

    def mouse_move_relative(self, dx: int, dy: int, duration: float = 0.0) -> bool:
        """Mock move mouse relative to current position (instant)."""
        new_x = self._mouse_position.x + dx
        new_y = self._mouse_position.y + dy
        logger.debug(f"[MOCK] Mouse move relative ({dx}, {dy}) to ({new_x}, {new_y})")
        self._mouse_position = MousePosition(x=new_x, y=new_y)
        self._record_action("mouse_move_relative", dx=dx, dy=dy, duration=duration)
        return True

    def mouse_click(
        self,
        x: int | None = None,
        y: int | None = None,
        button: MouseButton = MouseButton.LEFT,
        clicks: int = 1,
        interval: float = 0.0,
    ) -> bool:
        """Mock mouse click (instant)."""
        pos_x = x if x is not None else self._mouse_position.x
        pos_y = y if y is not None else self._mouse_position.y

        if x is not None and y is not None:
            self._mouse_position = MousePosition(x=x, y=y)

        logger.debug(f"[MOCK] Mouse click {button.value} at ({pos_x}, {pos_y}) x{clicks}")
        self._record_action("mouse_click", x=pos_x, y=pos_y, button=button.value, clicks=clicks)
        return True

    def mouse_down(
        self, x: int | None = None, y: int | None = None, button: MouseButton = MouseButton.LEFT
    ) -> bool:
        """Mock press and hold mouse button."""
        pos_x = x if x is not None else self._mouse_position.x
        pos_y = y if y is not None else self._mouse_position.y

        if x is not None and y is not None:
            self._mouse_position = MousePosition(x=x, y=y)

        logger.debug(f"[MOCK] Mouse down {button.value} at ({pos_x}, {pos_y})")
        self._record_action("mouse_down", x=pos_x, y=pos_y, button=button.value)
        return True

    def mouse_up(
        self, x: int | None = None, y: int | None = None, button: MouseButton = MouseButton.LEFT
    ) -> bool:
        """Mock release mouse button."""
        pos_x = x if x is not None else self._mouse_position.x
        pos_y = y if y is not None else self._mouse_position.y

        if x is not None and y is not None:
            self._mouse_position = MousePosition(x=x, y=y)

        logger.debug(f"[MOCK] Mouse up {button.value} at ({pos_x}, {pos_y})")
        self._record_action("mouse_up", x=pos_x, y=pos_y, button=button.value)
        return True

    def mouse_drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        button: MouseButton = MouseButton.LEFT,
        duration: float = 0.5,
    ) -> bool:
        """Mock drag mouse from start to end position (instant)."""
        logger.debug(f"[MOCK] Mouse drag from ({start_x}, {start_y}) to ({end_x}, {end_y})")
        self._mouse_position = MousePosition(x=end_x, y=end_y)
        self._record_action("mouse_drag", start_x=start_x, start_y=start_y,
                          end_x=end_x, end_y=end_y, button=button.value, duration=duration)
        return True

    def mouse_scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> bool:
        """Mock scroll mouse wheel (instant)."""
        pos_x = x if x is not None else self._mouse_position.x
        pos_y = y if y is not None else self._mouse_position.y

        logger.debug(f"[MOCK] Mouse scroll {clicks} clicks at ({pos_x}, {pos_y})")
        self._record_action("mouse_scroll", clicks=clicks, x=pos_x, y=pos_y)
        return True

    def get_mouse_position(self) -> MousePosition:
        """Get current mock mouse position."""
        return self._mouse_position

    def click_at(self, x: int, y: int, button: MouseButton = MouseButton.LEFT) -> bool:
        """Mock click at specific coordinates (instant)."""
        return self.mouse_click(x=x, y=y, button=button, clicks=1)

    def double_click_at(self, x: int, y: int, button: MouseButton = MouseButton.LEFT) -> bool:
        """Mock double click at specific coordinates (instant)."""
        return self.mouse_click(x=x, y=y, button=button, clicks=2)

    def drag(
        self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.0
    ) -> bool:
        """Mock drag from start to end position (instant)."""
        return self.mouse_drag(start_x, start_y, end_x, end_y, duration=duration)

    def move_mouse(self, x: int, y: int, duration: float = 0.0) -> bool:
        """Mock move mouse to position (instant)."""
        return self.mouse_move(x, y, duration)

    def scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> bool:
        """Mock scroll mouse wheel (instant)."""
        return self.mouse_scroll(clicks, x, y)

    # Keyboard operations

    def key_press(self, key: str | Key, presses: int = 1, interval: float = 0.0) -> bool:
        """Mock press key (instant)."""
        key_str = key.value if isinstance(key, Key) else key
        logger.debug(f"[MOCK] Key press '{key_str}' x{presses}")
        self._record_action("key_press", key=key_str, presses=presses)
        return True

    def key_down(self, key: str | Key) -> bool:
        """Mock press and hold key."""
        key_str = key.value if isinstance(key, Key) else key
        logger.debug(f"[MOCK] Key down '{key_str}'")
        self._pressed_keys.add(key_str)
        self._record_action("key_down", key=key_str)
        return True

    def key_up(self, key: str | Key) -> bool:
        """Mock release key."""
        key_str = key.value if isinstance(key, Key) else key
        logger.debug(f"[MOCK] Key up '{key_str}'")
        self._pressed_keys.discard(key_str)
        self._record_action("key_up", key=key_str)
        return True

    def type_text(self, text: str, interval: float = 0.0) -> bool:
        """Mock type text string (instant)."""
        logger.debug(f"[MOCK] Type text: '{text}'")
        self._record_action("type_text", text=text)
        return True

    def hotkey(self, *keys: str | Key) -> bool:
        """Mock press key combination (instant)."""
        key_strs = [k.value if isinstance(k, Key) else k for k in keys]
        logger.debug(f"[MOCK] Hotkey: {'+'.join(key_strs)}")
        self._record_action("hotkey", keys=key_strs)
        return True

    def is_key_pressed(self, key: str | Key) -> bool:
        """Check if key is currently pressed in mock state."""
        key_str = key.value if isinstance(key, Key) else key
        return key_str in self._pressed_keys

    # Mock-specific methods for testing

    def reset(self) -> None:
        """Reset mock state (for test cleanup)."""
        self._mouse_position = MousePosition(x=0, y=0)
        self._pressed_keys.clear()
        self._action_history.clear()
        logger.debug("MockInput reset")

    def get_action_history(self) -> list[dict]:
        """Get history of all actions performed."""
        return self._action_history.copy()

    def get_last_action(self) -> dict | None:
        """Get the last action performed."""
        return self._action_history[-1] if self._action_history else None
