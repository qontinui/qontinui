"""Match fusion options - ported from Qontinui framework.

Options for combining adjacent matches.
"""

from dataclasses import dataclass
from enum import Enum, auto


class FusionMethod(Enum):
    """Method for determining if matches should be fused."""

    ABSOLUTE = auto()
    """Use absolute pixel distance"""

    RELATIVE = auto()
    """Use relative distance based on match size"""


@dataclass
class MatchFusionOptions:
    """Options for combining adjacent matches into single matches.

    Port of MatchFusionOptions from Qontinui framework.

    This class controls how nearby matches are combined during find operations.
    Fusion can help when a single logical element is detected as multiple
    separate matches due to variations in the pattern or image quality.
    """

    fusion_method: FusionMethod = FusionMethod.ABSOLUTE
    """Method for determining fusion distance"""

    max_fusion_distance_x: int = 20
    """Maximum horizontal distance for fusion (pixels or relative)"""

    max_fusion_distance_y: int = 20
    """Maximum vertical distance for fusion (pixels or relative)"""

    fusion_threshold: float = 0.8
    """Minimum similarity for matches to be fused"""

    @classmethod
    def builder(cls) -> "MatchFusionOptionsBuilder":
        """Create a builder for MatchFusionOptions.

        Returns:
            A new builder instance
        """
        return MatchFusionOptionsBuilder()

    def to_builder(self) -> "MatchFusionOptionsBuilder":
        """Convert this instance to a builder for modification.

        Returns:
            A builder pre-populated with this instance's values
        """
        builder = MatchFusionOptionsBuilder()
        builder.fusion_method = self.fusion_method
        builder.max_fusion_distance_x = self.max_fusion_distance_x
        builder.max_fusion_distance_y = self.max_fusion_distance_y
        builder.fusion_threshold = self.fusion_threshold
        return builder


class MatchFusionOptionsBuilder:
    """Builder for MatchFusionOptions."""

    def __init__(self) -> None:
        self.fusion_method = FusionMethod.ABSOLUTE
        self.max_fusion_distance_x = 20
        self.max_fusion_distance_y = 20
        self.fusion_threshold = 0.8

    def set_fusion_method(self, method: FusionMethod) -> "MatchFusionOptionsBuilder":
        """Set the fusion method."""
        self.fusion_method = method
        return self

    def set_max_fusion_distance_x(self, distance: int) -> "MatchFusionOptionsBuilder":
        """Set maximum horizontal fusion distance."""
        self.max_fusion_distance_x = distance
        return self

    def set_max_fusion_distance_y(self, distance: int) -> "MatchFusionOptionsBuilder":
        """Set maximum vertical fusion distance."""
        self.max_fusion_distance_y = distance
        return self

    def set_fusion_threshold(self, threshold: float) -> "MatchFusionOptionsBuilder":
        """Set minimum similarity for fusion."""
        self.fusion_threshold = threshold
        return self

    def build(self) -> MatchFusionOptions:
        """Build the MatchFusionOptions instance.

        Returns:
            A new MatchFusionOptions with the configured values
        """
        return MatchFusionOptions(
            fusion_method=self.fusion_method,
            max_fusion_distance_x=self.max_fusion_distance_x,
            max_fusion_distance_y=self.max_fusion_distance_y,
            fusion_threshold=self.fusion_threshold,
        )
