"""Mouse up options - ported from Qontinui framework.

Configuration for mouse up actions.
"""

from ...action_config import ActionConfig, ActionConfigBuilder
from .mouse_press_options import MouseButton, MousePressOptions


class MouseUpOptions(ActionConfig):
    """Configuration for mouse up actions.

    Port of MouseUpOptions from Qontinui framework.

    This class encapsulates parameters for releasing mouse buttons.
    Used as part of drag operations or for custom mouse interactions.
    """

    def __init__(self, builder: "MouseUpOptionsBuilder") -> None:
        """Initialize MouseUpOptions from builder.

        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self.press_options: MousePressOptions = builder.press_options

    def get_press_options(self) -> MousePressOptions:
        """Get the mouse press options."""
        return self.press_options


class MouseUpOptionsBuilder(ActionConfigBuilder):
    """Builder for constructing MouseUpOptions with a fluent API.

    Port of MouseUpOptions from Qontinui framework.Builder.
    """

    def __init__(self, original: MouseUpOptions | None = None) -> None:
        """Initialize builder.

        Args:
            original: Optional MouseUpOptions instance to copy values from
        """
        super().__init__(original)

        if original:
            self.press_options = original.press_options
        else:
            self.press_options = MousePressOptions.builder().set_button(MouseButton.LEFT).build()

    def set_press_options(self, press_options: MousePressOptions) -> "MouseUpOptionsBuilder":
        """Set the mouse press options.

        Args:
            press_options: Configuration for mouse button and timing

        Returns:
            This builder instance for chaining
        """
        self.press_options = press_options
        return self

    def build(self) -> MouseUpOptions:
        """Build the immutable MouseUpOptions object.

        Returns:
            A new instance of MouseUpOptions
        """
        return MouseUpOptions(self)

    def _self(self) -> "MouseUpOptionsBuilder":
        """Return self for fluent interface.

        Returns:
            This builder instance
        """
        return self
