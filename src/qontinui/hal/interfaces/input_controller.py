"""Input controller interface definition."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class MouseButton(Enum):
    """Mouse button enumeration."""

    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class Key(Enum):
    """Special keyboard keys."""

    # Modifier keys
    ALT = "alt"
    ALT_L = "altleft"
    ALT_R = "altright"
    SHIFT = "shift"
    SHIFT_L = "shiftleft"
    SHIFT_R = "shiftright"
    CTRL = "ctrl"
    CTRL_L = "ctrlleft"
    CTRL_R = "ctrlright"
    CMD = "cmd"
    WIN = "win"

    # Navigation keys
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    HOME = "home"
    END = "end"
    PAGE_UP = "pageup"
    PAGE_DOWN = "pagedown"

    # Action keys
    ENTER = "enter"
    TAB = "tab"
    SPACE = "space"
    BACKSPACE = "backspace"
    DELETE = "delete"
    ESCAPE = "escape"

    # Function keys
    F1 = "f1"
    F2 = "f2"
    F3 = "f3"
    F4 = "f4"
    F5 = "f5"
    F6 = "f6"
    F7 = "f7"
    F8 = "f8"
    F9 = "f9"
    F10 = "f10"
    F11 = "f11"
    F12 = "f12"


@dataclass
class MousePosition:
    """Current mouse position."""

    x: int
    y: int


class IInputController(ABC):
    """Interface for input control operations."""

    # Mouse operations

    @abstractmethod
    def mouse_move(self, x: int, y: int, duration: float = 0.0) -> bool:
        """Move mouse to absolute position.

        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Movement duration in seconds (0 for instant)

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def mouse_move_relative(self, dx: int, dy: int, duration: float = 0.0) -> bool:
        """Move mouse relative to current position.

        Args:
            dx: X offset
            dy: Y offset
            duration: Movement duration in seconds

        Returns:
            True if successful
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        """
        pass

    @abstractmethod
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
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def mouse_scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> bool:
        """Scroll mouse wheel.

        Args:
            clicks: Number of scroll clicks (positive=up, negative=down)
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def get_mouse_position(self) -> MousePosition:
        """Get current mouse position.

        Returns:
            Current mouse position
        """
        pass

    @abstractmethod
    def click_at(self, x: int, y: int, button: MouseButton = MouseButton.LEFT) -> bool:
        """Click at specific coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button to click

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def double_click_at(self, x: int, y: int, button: MouseButton = MouseButton.LEFT) -> bool:
        """Double click at specific coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            button: Mouse button to click

        Returns:
            True if successful
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def move_mouse(self, x: int, y: int, duration: float = 0.0) -> bool:
        """Move mouse to position (alias for mouse_move).

        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Movement duration in seconds

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def scroll(self, clicks: int, x: int | None = None, y: int | None = None) -> bool:
        """Scroll mouse wheel (alias for mouse_scroll).

        Args:
            clicks: Number of scroll clicks (positive=up, negative=down)
            x: X coordinate (None for current position)
            y: Y coordinate (None for current position)

        Returns:
            True if successful
        """
        pass

    # Keyboard operations

    @abstractmethod
    def key_press(self, key: str | Key, presses: int = 1, interval: float = 0.0) -> bool:
        """Press key (down and up).

        Args:
            key: Key to press (string or Key enum)
            presses: Number of key presses
            interval: Interval between presses in seconds

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def key_down(self, key: str | Key) -> bool:
        """Press and hold key.

        Args:
            key: Key to press down

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def key_up(self, key: str | Key) -> bool:
        """Release key.

        Args:
            key: Key to release

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def type_text(self, text: str, interval: float = 0.0) -> bool:
        """Type text string.

        Args:
            text: Text to type
            interval: Interval between characters in seconds

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def hotkey(self, *keys: str | Key) -> bool:
        """Press key combination.

        Args:
            *keys: Keys to press together (e.g., 'ctrl', 'a')

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def is_key_pressed(self, key: str | Key) -> bool:
        """Check if key is currently pressed.

        Args:
            key: Key to check

        Returns:
            True if key is pressed
        """
        pass
