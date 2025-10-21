"""Default configuration values for all action options.

Similar to Brobot's properties file, this module defines default values
for timing, thresholds, and behavior of all actions.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class MouseActionDefaults:
    """Default values for mouse actions."""

    # Click timing
    click_hold_duration: float = 0.05  # Seconds to hold button down during click
    click_release_delay: float = 0.1   # Delay after releasing button
    click_safety_release: bool = True  # Release all buttons before clicking

    # Double-click timing
    double_click_interval: float = 0.1  # Delay between two clicks

    # Drag timing
    drag_start_delay: float = 0.1      # Delay after pressing button before moving
    drag_end_delay: float = 0.1        # Delay before releasing button after moving
    drag_default_duration: float = 1.0  # Default drag duration in seconds

    # Move timing
    move_default_duration: float = 0.0  # Default movement duration (0 = instant)

    # Button release timing
    safety_release_delay: float = 0.2  # Delay after safety release of all buttons


@dataclass
class KeyboardActionDefaults:
    """Default values for keyboard actions."""

    # Key press timing
    key_hold_duration: float = 0.05    # Seconds to hold key down
    key_release_delay: float = 0.05    # Delay after releasing key

    # Typing
    typing_interval: float = 0.05      # Delay between characters when typing

    # Hotkey timing
    hotkey_hold_duration: float = 0.05 # How long to hold hotkey combination
    hotkey_press_interval: float = 0.01 # Delay between pressing each key in combination


@dataclass
class FindActionDefaults:
    """Default values for find/match actions."""

    # Image matching
    default_similarity_threshold: float = 0.7  # Default match threshold
    default_timeout: float = 10.0              # Default search timeout in seconds
    default_retry_count: int = 3               # Default number of retries

    # Search behavior
    search_interval: float = 0.5               # Delay between search attempts


@dataclass
class WaitActionDefaults:
    """Default values for wait/pause actions."""

    # Standard waits
    default_wait_duration: float = 1.0         # Default wait time in seconds
    pause_before_action: float = 0.0           # Default pause before any action
    pause_after_action: float = 0.0            # Default pause after any action


@dataclass
class ActionDefaults:
    """Container for all action default configurations."""

    def __init__(self):
        self.mouse = MouseActionDefaults()
        self.keyboard = KeyboardActionDefaults()
        self.find = FindActionDefaults()
        self.wait = WaitActionDefaults()

    def get(self, category: str, option: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            category: Configuration category (mouse, keyboard, find, wait)
            option: Option name
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        category_obj = getattr(self, category, None)
        if category_obj is None:
            return default
        return getattr(category_obj, option, default)

    def set(self, category: str, option: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            category: Configuration category
            option: Option name
            value: Value to set
        """
        category_obj = getattr(self, category, None)
        if category_obj is not None:
            setattr(category_obj, option, value)

    def to_dict(self) -> dict:
        """Convert to dictionary representation.

        Returns:
            Dictionary of all configuration values
        """
        return {
            "mouse": {
                "click_hold_duration": self.mouse.click_hold_duration,
                "click_release_delay": self.mouse.click_release_delay,
                "click_safety_release": self.mouse.click_safety_release,
                "double_click_interval": self.mouse.double_click_interval,
                "drag_start_delay": self.mouse.drag_start_delay,
                "drag_end_delay": self.mouse.drag_end_delay,
                "drag_default_duration": self.mouse.drag_default_duration,
                "move_default_duration": self.mouse.move_default_duration,
                "safety_release_delay": self.mouse.safety_release_delay,
            },
            "keyboard": {
                "key_hold_duration": self.keyboard.key_hold_duration,
                "key_release_delay": self.keyboard.key_release_delay,
                "typing_interval": self.keyboard.typing_interval,
                "hotkey_hold_duration": self.keyboard.hotkey_hold_duration,
                "hotkey_press_interval": self.keyboard.hotkey_press_interval,
            },
            "find": {
                "default_similarity_threshold": self.find.default_similarity_threshold,
                "default_timeout": self.find.default_timeout,
                "default_retry_count": self.find.default_retry_count,
                "search_interval": self.find.search_interval,
            },
            "wait": {
                "default_wait_duration": self.wait.default_wait_duration,
                "pause_before_action": self.wait.pause_before_action,
                "pause_after_action": self.wait.pause_after_action,
            },
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ActionDefaults":
        """Create from dictionary.

        Args:
            config_dict: Dictionary of configuration values

        Returns:
            ActionDefaults instance
        """
        defaults = cls()

        # Load mouse settings
        if "mouse" in config_dict:
            for key, value in config_dict["mouse"].items():
                if hasattr(defaults.mouse, key):
                    setattr(defaults.mouse, key, value)

        # Load keyboard settings
        if "keyboard" in config_dict:
            for key, value in config_dict["keyboard"].items():
                if hasattr(defaults.keyboard, key):
                    setattr(defaults.keyboard, key, value)

        # Load find settings
        if "find" in config_dict:
            for key, value in config_dict["find"].items():
                if hasattr(defaults.find, key):
                    setattr(defaults.find, key, value)

        # Load wait settings
        if "wait" in config_dict:
            for key, value in config_dict["wait"].items():
                if hasattr(defaults.wait, key):
                    setattr(defaults.wait, key, value)

        return defaults


# Global instance (can be overridden by loading from file)
_action_defaults = ActionDefaults()


def get_defaults() -> ActionDefaults:
    """Get the global action defaults instance.

    Returns:
        ActionDefaults instance
    """
    return _action_defaults


def set_defaults(defaults: ActionDefaults) -> None:
    """Set the global action defaults instance.

    Args:
        defaults: New defaults instance
    """
    global _action_defaults
    _action_defaults = defaults


def load_defaults_from_file(file_path: str) -> ActionDefaults:
    """Load defaults from a configuration file.

    Args:
        file_path: Path to configuration file (JSON or YAML)

    Returns:
        ActionDefaults instance
    """
    import json
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(path, "r") as f:
        config_dict = json.load(f)

    defaults = ActionDefaults.from_dict(config_dict)
    set_defaults(defaults)
    return defaults
