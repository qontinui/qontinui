"""Mouse press options for fine-grained click control.

Port of MousePressOptions from Qontinui framework.
"""

from dataclasses import dataclass

from ....model.action.mouse_button import MouseButton


@dataclass
class MousePressOptions:
    """Options for controlling mouse press and release timing.

    Port of MousePressOptions from Qontinui framework class.

    Provides fine-grained control over the timing of mouse button
    press and release operations, allowing for precise simulation
    of different click patterns.
    """

    button: MouseButton = MouseButton.LEFT
    pause_before_mouse_down: float = 0.0  # Pause before pressing button
    pause_after_mouse_down: float = 0.0  # Pause after pressing button
    pause_before_mouse_up: float = 0.0  # Pause before releasing button
    pause_after_mouse_up: float = 0.0  # Pause after releasing button

    def with_button(self, button: MouseButton) -> "MousePressOptions":
        """Set mouse button.

        Args:
            button: Mouse button to use

        Returns:
            Self for fluent interface
        """
        self.button = button
        return self

    def pause_before_press(self, seconds: float) -> "MousePressOptions":
        """Set pause before mouse press.

        Args:
            seconds: Pause duration in seconds

        Returns:
            Self for fluent interface
        """
        self.pause_before_mouse_down = seconds
        return self

    def pause_after_press(self, seconds: float) -> "MousePressOptions":
        """Set pause after mouse press.

        Args:
            seconds: Pause duration in seconds

        Returns:
            Self for fluent interface
        """
        self.pause_after_mouse_down = seconds
        return self

    def pause_before_release(self, seconds: float) -> "MousePressOptions":
        """Set pause before mouse release.

        Args:
            seconds: Pause duration in seconds

        Returns:
            Self for fluent interface
        """
        self.pause_before_mouse_up = seconds
        return self

    def pause_after_release(self, seconds: float) -> "MousePressOptions":
        """Set pause after mouse release.

        Args:
            seconds: Pause duration in seconds

        Returns:
            Self for fluent interface
        """
        self.pause_after_mouse_up = seconds
        return self

    def total_pause_time(self) -> float:
        """Calculate total pause time for this press operation.

        Returns:
            Sum of all pause durations
        """
        return (
            self.pause_before_mouse_down
            + self.pause_after_mouse_down
            + self.pause_before_mouse_up
            + self.pause_after_mouse_up
        )

    @staticmethod
    def default() -> "MousePressOptions":
        """Create default mouse press options.

        Returns:
            Default MousePressOptions
        """
        return MousePressOptions()

    @staticmethod
    def with_hold_time(hold_duration: float) -> "MousePressOptions":
        """Create options with specified hold duration.

        Args:
            hold_duration: Time to hold button pressed

        Returns:
            MousePressOptions with hold time
        """
        return MousePressOptions(pause_after_mouse_down=hold_duration)

    def __repr__(self) -> str:
        """String representation.

        Returns:
            Description of options
        """
        parts = [f"button={self.button.name}"]
        if self.pause_before_mouse_down > 0:
            parts.append(f"before_press={self.pause_before_mouse_down}s")
        if self.pause_after_mouse_down > 0:
            parts.append(f"after_press={self.pause_after_mouse_down}s")
        if self.pause_before_mouse_up > 0:
            parts.append(f"before_release={self.pause_before_mouse_up}s")
        if self.pause_after_mouse_up > 0:
            parts.append(f"after_release={self.pause_after_mouse_up}s")
        return f"MousePressOptions({', '.join(parts)})"
