"""Mouse down options - ported from Qontinui framework.

Configuration for mouse down actions.
"""

from ...action_config import ActionConfig, ActionConfigBuilder
from .mouse_press_options import MouseButton, MousePressOptions


class MouseDownOptions(ActionConfig):
    """Configuration for mouse down actions.

    Port of MouseDownOptions from Qontinui framework.

    This class encapsulates parameters for pressing and holding mouse buttons.
    Used as part of drag operations or for custom mouse interactions.
    """

    def __init__(self, builder: "MouseDownOptionsBuilder"):
        """Initialize MouseDownOptions from builder.

        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self.press_options: MousePressOptions = builder.press_options

    def get_press_options(self) -> MousePressOptions:
        """Get the mouse press options."""
        return self.press_options


class MouseDownOptionsBuilder(ActionConfigBuilder):
    """Builder for constructing MouseDownOptions with a fluent API.

    Port of MouseDownOptions from Qontinui framework.Builder.
    """

    def __init__(self, original: MouseDownOptions | None = None):
        """Initialize builder.

        Args:
            original: Optional MouseDownOptions instance to copy values from
        """
        super().__init__(original)

        if original:
            self.press_options = original.press_options
        else:
            self.press_options = MousePressOptions.builder().set_button(MouseButton.LEFT).build()

    def set_press_options(self, press_options: MousePressOptions) -> "MouseDownOptionsBuilder":
        """Set the mouse press options.

        Args:
            press_options: Configuration for mouse button and timing

        Returns:
            This builder instance for chaining
        """
        self.press_options = press_options
        return self

    def build(self) -> MouseDownOptions:
        """Build the immutable MouseDownOptions object.

        Returns:
            A new instance of MouseDownOptions
        """
        return MouseDownOptions(self)

    def _self(self) -> "MouseDownOptionsBuilder":
        """Return self for fluent interface.

        Returns:
            This builder instance
        """
        return self
