"""Match adjustment options - ported from Qontinui framework.

Options for post-processing found matches.
"""

from dataclasses import dataclass


@dataclass
class MatchAdjustmentOptions:
    """Options for adjusting match regions after they are found.

    Port of MatchAdjustmentOptions from Qontinui framework.

    This class allows for resizing match regions or targeting specific
    points within matches after they are found by the find operation.
    """

    # Adjustment factors for match region
    x_adjustment: float = 0.0
    """Horizontal adjustment in pixels"""

    y_adjustment: float = 0.0
    """Vertical adjustment in pixels"""

    width_factor: float = 1.0
    """Factor to multiply match width by"""

    height_factor: float = 1.0
    """Factor to multiply match height by"""

    # Target point within match
    target_x_offset: float = 0.5
    """X offset within match (0.0 = left, 0.5 = center, 1.0 = right)"""

    target_y_offset: float = 0.5
    """Y offset within match (0.0 = top, 0.5 = center, 1.0 = bottom)"""

    @classmethod
    def builder(cls) -> "MatchAdjustmentOptionsBuilder":
        """Create a builder for MatchAdjustmentOptions.

        Returns:
            A new builder instance
        """
        return MatchAdjustmentOptionsBuilder()

    def to_builder(self) -> "MatchAdjustmentOptionsBuilder":
        """Convert this instance to a builder for modification.

        Returns:
            A builder pre-populated with this instance's values
        """
        builder = MatchAdjustmentOptionsBuilder()
        builder.x_adjustment = self.x_adjustment
        builder.y_adjustment = self.y_adjustment
        builder.width_factor = self.width_factor
        builder.height_factor = self.height_factor
        builder.target_x_offset = self.target_x_offset
        builder.target_y_offset = self.target_y_offset
        return builder


class MatchAdjustmentOptionsBuilder:
    """Builder for MatchAdjustmentOptions."""

    def __init__(self):
        self.x_adjustment = 0.0
        self.y_adjustment = 0.0
        self.width_factor = 1.0
        self.height_factor = 1.0
        self.target_x_offset = 0.5
        self.target_y_offset = 0.5

    def set_x_adjustment(self, value: float) -> "MatchAdjustmentOptionsBuilder":
        """Set horizontal adjustment."""
        self.x_adjustment = value
        return self

    def set_y_adjustment(self, value: float) -> "MatchAdjustmentOptionsBuilder":
        """Set vertical adjustment."""
        self.y_adjustment = value
        return self

    def set_width_factor(self, value: float) -> "MatchAdjustmentOptionsBuilder":
        """Set width multiplication factor."""
        self.width_factor = value
        return self

    def set_height_factor(self, value: float) -> "MatchAdjustmentOptionsBuilder":
        """Set height multiplication factor."""
        self.height_factor = value
        return self

    def set_target_x_offset(self, value: float) -> "MatchAdjustmentOptionsBuilder":
        """Set target X offset within match."""
        self.target_x_offset = value
        return self

    def set_target_y_offset(self, value: float) -> "MatchAdjustmentOptionsBuilder":
        """Set target Y offset within match."""
        self.target_y_offset = value
        return self

    def build(self) -> MatchAdjustmentOptions:
        """Build the MatchAdjustmentOptions instance.

        Returns:
            A new MatchAdjustmentOptions with the configured values
        """
        return MatchAdjustmentOptions(
            x_adjustment=self.x_adjustment,
            y_adjustment=self.y_adjustment,
            width_factor=self.width_factor,
            height_factor=self.height_factor,
            target_x_offset=self.target_x_offset,
            target_y_offset=self.target_y_offset,
        )
