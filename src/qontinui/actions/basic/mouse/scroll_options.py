"""Scroll options - ported from Qontinui framework.

Configuration for mouse wheel scrolling actions.
"""

from enum import Enum, auto

from ...action_config import ActionConfig, ActionConfigBuilder


class Direction(Enum):
    """Defines the direction of mouse wheel scrolling."""

    UP = auto()
    """Scroll upward (toward the top of the page/content)."""

    DOWN = auto()
    """Scroll downward (toward the bottom of the page/content)."""


class ScrollOptions(ActionConfig):
    """Configuration for mouse wheel scrolling actions.

    Port of ScrollOptions from Qontinui framework.

    This class encapsulates all parameters for scrolling the mouse wheel,
    including direction and number of scroll steps. It is an immutable object
    and must be constructed using its inner Builder.

    By providing a specialized configuration class, the Qontinui API ensures that only
    relevant options are available for scroll operations, enhancing type safety
    and ease of use.
    """

    def __init__(self, builder: "ScrollOptionsBuilder") -> None:
        """Initialize ScrollOptions from builder.

        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self.direction: Direction = builder.direction
        self.scroll_steps: int = builder.scroll_steps

    def get_direction(self) -> Direction:
        """Get the scroll direction."""
        return self.direction

    def get_scroll_steps(self) -> int:
        """Get the number of scroll steps."""
        return self.scroll_steps


class ScrollOptionsBuilder(ActionConfigBuilder):
    """Builder for constructing ScrollOptions with a fluent API.

    Port of ScrollOptions from Qontinui framework.Builder.
    """

    def __init__(self, original: ScrollOptions | None = None) -> None:
        """Initialize builder.

        Args:
            original: Optional ScrollOptions instance to copy values from
        """
        super().__init__(original)

        if original:
            self.direction = original.direction
            self.scroll_steps = original.scroll_steps
        else:
            self.direction = Direction.DOWN
            self.scroll_steps = 3

    def set_direction(self, direction: Direction) -> "ScrollOptionsBuilder":
        """Set the scroll direction.

        Args:
            direction: The direction to scroll (UP or DOWN)

        Returns:
            This builder instance for chaining
        """
        self.direction = direction
        return self

    def set_scroll_steps(self, scroll_steps: int) -> "ScrollOptionsBuilder":
        """Set the number of scroll steps (or "clicks" of the wheel).

        Each step represents one notch of the mouse wheel.

        Args:
            scroll_steps: The number of scroll steps. Must be positive

        Returns:
            This builder instance for chaining
        """
        self.scroll_steps = max(1, scroll_steps)
        return self

    def build(self) -> ScrollOptions:
        """Build the immutable ScrollOptions object.

        Returns:
            A new instance of ScrollOptions
        """
        return ScrollOptions(self)

    def _self(self) -> "ScrollOptionsBuilder":
        """Return self for fluent interface.

        Returns:
            This builder instance
        """
        return self
