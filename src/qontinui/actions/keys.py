"""Special key enums - ported from SikuliX/Brobot concepts.

This module provides enums for special keyboard keys that cannot be typed
directly as strings, similar to SikuliX's Key class.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class Key(Enum):
    """Special keyboard keys enum.

    Similar to SikuliX's Key class, provides string representations
    of special keyboard keys for use in automation.

    Each value is the string that should be sent to the underlying
    automation library (e.g., pyautogui, pynput) to trigger that key.
    """

    # Navigation keys
    ENTER = "\n"
    RETURN = "\r"
    TAB = "\t"
    SPACE = " "
    BACKSPACE = "\b"
    DELETE = "delete"

    # Arrow keys
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

    # Modifier keys (also in KeyModifier enum for specific modifier operations)
    SHIFT = "shift"
    CTRL = "ctrl"
    CONTROL = "ctrl"  # Alias
    ALT = "alt"
    META = "meta"  # Windows/Command key
    WIN = "win"  # Windows key
    CMD = "cmd"  # Mac Command key

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
    F13 = "f13"
    F14 = "f14"
    F15 = "f15"

    # Navigation cluster
    HOME = "home"
    END = "end"
    PAGE_UP = "pageup"
    PAGE_DOWN = "pagedown"
    INSERT = "insert"

    # Special keys
    ESC = "escape"
    ESCAPE = "escape"  # Alias
    PRINT_SCREEN = "printscreen"
    SCROLL_LOCK = "scrolllock"
    PAUSE = "pause"
    BREAK = "break"
    CAPS_LOCK = "capslock"
    NUM_LOCK = "numlock"

    # Numpad keys
    NUM0 = "num0"
    NUM1 = "num1"
    NUM2 = "num2"
    NUM3 = "num3"
    NUM4 = "num4"
    NUM5 = "num5"
    NUM6 = "num6"
    NUM7 = "num7"
    NUM8 = "num8"
    NUM9 = "num9"
    NUM_MULTIPLY = "multiply"
    NUM_ADD = "add"
    NUM_SEPARATOR = "separator"
    NUM_SUBTRACT = "subtract"
    NUM_DECIMAL = "decimal"
    NUM_DIVIDE = "divide"

    # Media keys (may not be supported on all platforms)
    VOLUME_UP = "volumeup"
    VOLUME_DOWN = "volumedown"
    VOLUME_MUTE = "volumemute"
    MEDIA_PLAY_PAUSE = "playpause"
    MEDIA_STOP = "stop"
    MEDIA_NEXT_TRACK = "nexttrack"
    MEDIA_PREV_TRACK = "prevtrack"

    # Browser control keys
    BROWSER_BACK = "browserback"
    BROWSER_FORWARD = "browserforward"
    BROWSER_REFRESH = "browserrefresh"
    BROWSER_STOP = "browserstop"
    BROWSER_SEARCH = "browsersearch"
    BROWSER_FAVORITES = "browserfavorites"
    BROWSER_HOME = "browserhome"

    # System keys
    SLEEP = "sleep"
    APPS = "apps"  # Context menu key

    def __str__(self) -> str:
        """Return the key value as a string."""
        return self.value

    @classmethod
    def is_modifier(cls, key: Key) -> bool:
        """Check if a key is a modifier key.

        Args:
            key: Key to check

        Returns:
            True if the key is a modifier
        """
        modifiers = {
            cls.SHIFT,
            cls.CTRL,
            cls.CONTROL,
            cls.ALT,
            cls.META,
            cls.WIN,
            cls.CMD,
        }
        return key in modifiers

    @classmethod
    def is_function_key(cls, key: Key) -> bool:
        """Check if a key is a function key (F1-F15).

        Args:
            key: Key to check

        Returns:
            True if the key is a function key
        """
        function_keys = {
            cls.F1,
            cls.F2,
            cls.F3,
            cls.F4,
            cls.F5,
            cls.F6,
            cls.F7,
            cls.F8,
            cls.F9,
            cls.F10,
            cls.F11,
            cls.F12,
            cls.F13,
            cls.F14,
            cls.F15,
        }
        return key in function_keys

    @classmethod
    def is_navigation_key(cls, key: Key) -> bool:
        """Check if a key is a navigation key.

        Args:
            key: Key to check

        Returns:
            True if the key is a navigation key
        """
        nav_keys = {
            cls.UP,
            cls.DOWN,
            cls.LEFT,
            cls.RIGHT,
            cls.HOME,
            cls.END,
            cls.PAGE_UP,
            cls.PAGE_DOWN,
        }
        return key in nav_keys

    @classmethod
    def from_string(cls, key_string: str) -> Key:
        """Get a Key enum from its string representation.

        Args:
            key_string: String representation of the key

        Returns:
            Corresponding Key enum

        Raises:
            ValueError: If no matching key is found
        """
        # First try exact match with enum name
        try:
            return cls[key_string.upper()]
        except KeyError:
            pass

        # Then try matching by value
        for key in cls:
            if key.value == key_string.lower():
                return key

        raise ValueError(f"No Key enum found for '{key_string}'")


class KeyCombo:
    """Helper class for creating keyboard combinations.

    Similar to SikuliX's approach for key combinations.
    """

    def __init__(self, *keys) -> None:
        """Initialize a key combination.

        Args:
            *keys: Keys or modifiers to combine
        """
        self.keys = list(keys)
        self.modifiers = []
        self.main_key = None

        # Import KeyModifier here to avoid circular import
        try:
            from .action_options import KeyModifier

            key_modifier_type = KeyModifier
        except ImportError:
            key_modifier_type = None  # type: ignore[assignment]

        # Separate modifiers from main key
        for key in keys:
            if isinstance(key, Key) and Key.is_modifier(key):
                self.modifiers.append(key)
            elif key_modifier_type is not None and isinstance(key, key_modifier_type):
                # Convert KeyModifier to Key
                modifier_key = Key.from_string(key.value)
                self.modifiers.append(modifier_key)
            else:
                self.main_key = key

    def __str__(self) -> str:
        """String representation of the key combination."""
        parts = []
        for mod in self.modifiers:
            parts.append(str(mod))
        if self.main_key:
            parts.append(str(self.main_key))
        return "+".join(parts)

    @staticmethod
    def ctrl(key) -> KeyCombo:
        """Create a Ctrl+key combination."""
        return KeyCombo(Key.CTRL, key)

    @staticmethod
    def alt(key) -> KeyCombo:
        """Create an Alt+key combination."""
        return KeyCombo(Key.ALT, key)

    @staticmethod
    def shift(key) -> KeyCombo:
        """Create a Shift+key combination."""
        return KeyCombo(Key.SHIFT, key)

    @staticmethod
    def meta(key) -> KeyCombo:
        """Create a Meta/Win/Cmd+key combination."""
        return KeyCombo(Key.META, key)


# Common key combinations as constants (similar to SikuliX)
class KeyCombos:
    """Common keyboard combinations as class attributes."""

    # Text editing
    SELECT_ALL = KeyCombo.ctrl("a")
    COPY = KeyCombo.ctrl("c")
    CUT = KeyCombo.ctrl("x")
    PASTE = KeyCombo.ctrl("v")
    UNDO = KeyCombo.ctrl("z")
    REDO = KeyCombo.ctrl("y")
    FIND = KeyCombo.ctrl("f")
    REPLACE = KeyCombo.ctrl("h")
    SAVE = KeyCombo.ctrl("s")
    SAVE_AS = KeyCombo.ctrl(KeyCombo.shift("s"))

    # Navigation
    NEXT_TAB = KeyCombo.ctrl(Key.TAB)
    PREV_TAB = KeyCombo.ctrl(KeyCombo.shift(Key.TAB))
    CLOSE_TAB = KeyCombo.ctrl("w")
    NEW_TAB = KeyCombo.ctrl("t")

    # System
    TASK_MANAGER = KeyCombo.ctrl(KeyCombo.shift(Key.ESC))
    ALT_TAB = KeyCombo.alt(Key.TAB)
    ALT_F4 = KeyCombo.alt(Key.F4)

    # Mac specific (using meta/cmd)
    MAC_COPY = KeyCombo.meta("c")
    MAC_PASTE = KeyCombo.meta("v")
    MAC_CUT = KeyCombo.meta("x")
    MAC_QUIT = KeyCombo.meta("q")
