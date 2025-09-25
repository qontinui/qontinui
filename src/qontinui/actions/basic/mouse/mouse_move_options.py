"""Mouse move options - ported from Qontinui framework.

Configuration for mouse movement actions.
"""

from dataclasses import dataclass

from ...action_config import ActionConfig


@dataclass
class MouseMoveOptions(ActionConfig):
    """Configuration for mouse movement actions.

    Port of MouseMoveOptions from Qontinui framework class.

    Configures mouse movement behavior and timing.
    """

    # Movement configuration
    move_instantly: bool = False  # True for instant movement, False for animated
    move_speed: float = 0.5  # Speed of animated movement (seconds)

    def instantly(self) -> "MouseMoveOptions":
        """Configure for instant movement.

        Returns:
            Self for fluent interface
        """
        self.move_instantly = True
        return self

    def with_speed(self, speed: float) -> "MouseMoveOptions":
        """Set movement speed.

        Args:
            speed: Movement duration in seconds

        Returns:
            Self for fluent interface
        """
        self.move_speed = speed
        self.move_instantly = False
        return self
