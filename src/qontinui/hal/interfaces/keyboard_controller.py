"""Keyboard controller interface definition."""

from abc import ABC, abstractmethod
from enum import Enum


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


class IKeyboardController(ABC):
    """Interface for keyboard control operations.

    This interface defines keyboard-only operations following the
    Single Responsibility Principle. Implementations should focus
    solely on keyboard input control.
    """

    @abstractmethod
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
