"""Mouse press options - ported from Qontinui framework.

Options for mouse press operations.
"""

from dataclasses import dataclass
from enum import Enum, auto


class MouseButton(Enum):
    """Mouse button types."""

    LEFT = auto()
    """Left mouse button"""

    RIGHT = auto()
    """Right mouse button"""

    MIDDLE = auto()
    """Middle mouse button"""


@dataclass
class MousePressOptions:
    """Options for mouse press operations.

    Port of MousePressOptions from Qontinui framework.

    This class encapsulates parameters for mouse button operations
    including which button to press and timing parameters.
    """

    button: MouseButton = MouseButton.LEFT
    """The mouse button to press"""

    press_duration: float = 0.0
    """Duration to hold the button pressed in seconds"""

    pause_after_press: float = 0.0
    """Pause after pressing the button in seconds"""

    pause_after_release: float = 0.0
    """Pause after releasing the button in seconds"""

    @classmethod
    def builder(cls) -> "MousePressOptionsBuilder":
        """Create a builder for MousePressOptions.

        Returns:
            A new builder instance
        """
        return MousePressOptionsBuilder()

    def to_builder(self) -> "MousePressOptionsBuilder":
        """Convert this instance to a builder for modification.

        Returns:
            A builder pre-populated with this instance's values
        """
        builder = MousePressOptionsBuilder()
        builder.button = self.button
        builder.press_duration = self.press_duration
        builder.pause_after_press = self.pause_after_press
        builder.pause_after_release = self.pause_after_release
        return builder


class MousePressOptionsBuilder:
    """Builder for MousePressOptions."""

    def __init__(self) -> None:
        self.button = MouseButton.LEFT
        self.press_duration = 0.0
        self.pause_after_press = 0.0
        self.pause_after_release = 0.0

    def set_button(self, button: MouseButton) -> "MousePressOptionsBuilder":
        """Set the mouse button to press."""
        self.button = button
        return self

    def set_press_duration(self, duration: float) -> "MousePressOptionsBuilder":
        """Set duration to hold the button pressed."""
        self.press_duration = max(0.0, duration)
        return self

    def set_pause_after_press(self, pause: float) -> "MousePressOptionsBuilder":
        """Set pause after pressing the button."""
        self.pause_after_press = max(0.0, pause)
        return self

    def set_pause_after_release(self, pause: float) -> "MousePressOptionsBuilder":
        """Set pause after releasing the button."""
        self.pause_after_release = max(0.0, pause)
        return self

    def build(self) -> MousePressOptions:
        """Build the MousePressOptions instance.

        Returns:
            A new MousePressOptions with the configured values
        """
        return MousePressOptions(
            button=self.button,
            press_duration=self.press_duration,
            pause_after_press=self.pause_after_press,
            pause_after_release=self.pause_after_release,
        )
