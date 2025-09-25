"""Color find options - ported from Qontinui framework.

Configuration for color-based find operations.
"""

from dataclasses import dataclass, field
from enum import Enum, auto

from .base_find_options import BaseFindOptions, FindStrategy


class ColorStrategy(Enum):
    """Defines the color analysis strategy.

    Port of Color from Qontinui framework enum.
    """

    KMEANS = auto()  # Find RGB color cluster centers using k-means algorithm
    MU = auto()  # Min, max, mean, and standard deviation of HSV values
    CLASSIFICATION = auto()  # Multi-class classification by color profiles


@dataclass
class AreaFilteringOptions:
    """Configuration for filtering results by pixel area.

    Port of AreaFilteringOptions from Qontinui framework.

    Filters matches based on their minimum and maximum size.
    """

    min_area: int = 1  # Minimum pixels for valid match
    max_area: int = -1  # Maximum pixels (-1 = no limit)

    def validate(self) -> bool:
        """Validate area filtering options.

        Returns:
            True if valid
        """
        if self.min_area < 0:
            return False
        if self.max_area != -1 and self.max_area < self.min_area:
            return False
        return True


@dataclass
class HSVBinOptions:
    """Configuration for HSV histogram binning.

    Port of HSVBinOptions from Qontinui framework.

    Controls the number of bins for hue, saturation, and value channels.
    """

    hue_bins: int = 12
    saturation_bins: int = 2
    value_bins: int = 1

    def validate(self) -> bool:
        """Validate HSV bin options.

        Returns:
            True if valid
        """
        return self.hue_bins > 0 and self.saturation_bins > 0 and self.value_bins > 0


@dataclass
class ColorFindOptions(BaseFindOptions):
    """Configuration for color-based find operations.

    Port of ColorFindOptions from Qontinui framework class.

    Configures color-based pattern matching using k-means clustering,
    mean color statistics, or multi-class classification.
    """

    # Color analysis configuration
    color_strategy: ColorStrategy = ColorStrategy.MU
    diameter: int = 5  # Width/height of color boxes to find
    kmeans: int = 2  # Number of k-means clusters

    # Filtering options
    area_filtering: AreaFilteringOptions = field(default_factory=AreaFilteringOptions)
    bin_options: HSVBinOptions = field(default_factory=HSVBinOptions)

    def get_strategy(self) -> FindStrategy:
        """Get the find strategy for color finding.

        Returns:
            COLOR strategy
        """
        return FindStrategy.COLOR

    def with_color_strategy(self, strategy: ColorStrategy) -> "ColorFindOptions":
        """Set color analysis strategy.

        Args:
            strategy: Color strategy to use

        Returns:
            Self for fluent interface
        """
        self.color_strategy = strategy
        return self

    def with_diameter(self, diameter: int) -> "ColorFindOptions":
        """Set diameter of color boxes.

        Args:
            diameter: Diameter in pixels

        Returns:
            Self for fluent interface
        """
        self.diameter = diameter
        return self

    def with_kmeans(self, clusters: int) -> "ColorFindOptions":
        """Set number of k-means clusters.

        Args:
            clusters: Number of clusters

        Returns:
            Self for fluent interface
        """
        self.kmeans = clusters
        return self

    def with_area_filtering(self, min_area: int = 1, max_area: int = -1) -> "ColorFindOptions":
        """Configure area filtering.

        Args:
            min_area: Minimum area in pixels
            max_area: Maximum area in pixels (-1 for no limit)

        Returns:
            Self for fluent interface
        """
        self.area_filtering = AreaFilteringOptions(min_area, max_area)
        return self

    def with_hsv_bins(
        self, hue: int = 12, saturation: int = 2, value: int = 1
    ) -> "ColorFindOptions":
        """Configure HSV histogram bins.

        Args:
            hue: Number of hue bins
            saturation: Number of saturation bins
            value: Number of value bins

        Returns:
            Self for fluent interface
        """
        self.bin_options = HSVBinOptions(hue, saturation, value)
        return self

    def validate(self) -> bool:
        """Validate color find configuration.

        Returns:
            True if valid
        """
        if not super().validate():
            return False
        if self.diameter < 0:
            return False
        if self.kmeans <= 0:
            return False
        if not self.area_filtering.validate():
            return False
        if not self.bin_options.validate():
            return False
        return True

    @staticmethod
    def kmeans_defaults() -> "ColorFindOptions":
        """Create options for k-means clustering.

        Returns:
            ColorFindOptions configured for k-means
        """
        return ColorFindOptions(color_strategy=ColorStrategy.KMEANS, kmeans=3, diameter=10)

    @staticmethod
    def classification_defaults() -> "ColorFindOptions":
        """Create options for classification.

        Returns:
            ColorFindOptions configured for classification
        """
        return ColorFindOptions(
            color_strategy=ColorStrategy.CLASSIFICATION,
            diameter=5,
            area_filtering=AreaFilteringOptions(min_area=100, max_area=5000),
        )
