"""Mouse move options - ported from Qontinui framework.

Configuration for mouse movement actions.
"""

from ...action_config import ActionConfig, ActionConfigBuilder


class MouseMoveOptions(ActionConfig):
    """Configuration for mouse movement actions.

    Port of MouseMoveOptions from Qontinui framework class.

    This class encapsulates all parameters for mouse movement actions, including
    movement speed and instant movement behavior. It is an immutable object and
    must be constructed using its inner Builder.

    This specialized configuration enhances API clarity by only exposing options
    relevant to mouse movement operations.

    Example usage:
        instant_move = MouseMoveOptionsBuilder().set_instant().build()
        slow_move = MouseMoveOptionsBuilder().set_speed(1.5).build()
    """

    def __init__(self, builder: "MouseMoveOptionsBuilder") -> None:
        """Initialize MouseMoveOptions from builder.

        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self.move_instantly: bool = builder.move_instantly
        self.move_speed: float = builder.move_speed

    def get_move_instantly(self) -> bool:
        """Get whether to move instantly.

        Returns:
            True for instant movement, False for animated movement
        """
        return self.move_instantly

    def get_move_speed(self) -> float:
        """Get the movement speed.

        Returns:
            Movement duration in seconds
        """
        return self.move_speed


class MouseMoveOptionsBuilder(ActionConfigBuilder):
    """Builder for constructing MouseMoveOptions with a fluent API.

    Port of MouseMoveOptions from Qontinui framework.Builder.
    """

    def __init__(self, original: MouseMoveOptions | None = None) -> None:
        """Initialize builder.

        Args:
            original: Optional MouseMoveOptions instance to copy values from
        """
        super().__init__(original)

        if original:
            self.move_instantly = original.move_instantly
            self.move_speed = original.move_speed
        else:
            self.move_instantly = False
            self.move_speed = 0.5

    def set_instant(self) -> "MouseMoveOptionsBuilder":
        """Configure for instant movement.

        Returns:
            This builder instance for chaining
        """
        self.move_instantly = True
        return self

    def set_speed(self, speed: float) -> "MouseMoveOptionsBuilder":
        """Set movement speed for animated movement.

        Args:
            speed: Movement duration in seconds

        Returns:
            This builder instance for chaining
        """
        self.move_speed = speed
        self.move_instantly = False
        return self

    def build(self) -> MouseMoveOptions:
        """Build the immutable MouseMoveOptions object.

        Returns:
            A new instance of MouseMoveOptions
        """
        return MouseMoveOptions(self)

    def _self(self) -> "MouseMoveOptionsBuilder":
        """Return self for fluent interface.

        Returns:
            This builder instance
        """
        return self
