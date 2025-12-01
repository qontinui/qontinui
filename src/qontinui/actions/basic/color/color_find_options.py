"""Color find options - ported from Qontinui framework.

Configuration for color-based find operations.
"""

from dataclasses import dataclass
from enum import Enum, auto

from ..find.base_find_options import BaseFindOptions, BaseFindOptionsBuilder
from ..find.find_strategy import FindStrategy


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


class AreaFilteringOptionsBuilder:
    """Builder for constructing AreaFilteringOptions with a fluent API.

    Example usage:
        options = AreaFilteringOptionsBuilder()
            .set_min_area(100)
            .set_max_area(5000)
            .build()
    """

    def __init__(self, original: AreaFilteringOptions | None = None) -> None:
        """Initialize builder.

        Args:
            original: Optional AreaFilteringOptions instance to copy values from
        """
        if original:
            self.min_area = original.min_area
            self.max_area = original.max_area
        else:
            self.min_area = 1
            self.max_area = -1

    def set_min_area(self, min_area: int) -> "AreaFilteringOptionsBuilder":
        """Set minimum area in pixels.

        Args:
            min_area: Minimum area for valid match

        Returns:
            This builder instance for chaining
        """
        self.min_area = min_area
        return self

    def set_max_area(self, max_area: int) -> "AreaFilteringOptionsBuilder":
        """Set maximum area in pixels.

        Args:
            max_area: Maximum area (-1 for no limit)

        Returns:
            This builder instance for chaining
        """
        self.max_area = max_area
        return self

    def build(self) -> AreaFilteringOptions:
        """Build the immutable AreaFilteringOptions object.

        Returns:
            A new instance of AreaFilteringOptions
        """
        return AreaFilteringOptions(min_area=self.min_area, max_area=self.max_area)


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


class HSVBinOptionsBuilder:
    """Builder for constructing HSVBinOptions with a fluent API.

    Example usage:
        options = HSVBinOptionsBuilder()
            .set_hue_bins(18)
            .set_saturation_bins(3)
            .set_value_bins(2)
            .build()
    """

    def __init__(self, original: HSVBinOptions | None = None) -> None:
        """Initialize builder.

        Args:
            original: Optional HSVBinOptions instance to copy values from
        """
        if original:
            self.hue_bins = original.hue_bins
            self.saturation_bins = original.saturation_bins
            self.value_bins = original.value_bins
        else:
            self.hue_bins = 12
            self.saturation_bins = 2
            self.value_bins = 1

    def set_hue_bins(self, hue_bins: int) -> "HSVBinOptionsBuilder":
        """Set number of hue bins.

        Args:
            hue_bins: Number of bins for hue channel

        Returns:
            This builder instance for chaining
        """
        self.hue_bins = hue_bins
        return self

    def set_saturation_bins(self, saturation_bins: int) -> "HSVBinOptionsBuilder":
        """Set number of saturation bins.

        Args:
            saturation_bins: Number of bins for saturation channel

        Returns:
            This builder instance for chaining
        """
        self.saturation_bins = saturation_bins
        return self

    def set_value_bins(self, value_bins: int) -> "HSVBinOptionsBuilder":
        """Set number of value bins.

        Args:
            value_bins: Number of bins for value channel

        Returns:
            This builder instance for chaining
        """
        self.value_bins = value_bins
        return self

    def build(self) -> HSVBinOptions:
        """Build the immutable HSVBinOptions object.

        Returns:
            A new instance of HSVBinOptions
        """
        return HSVBinOptions(
            hue_bins=self.hue_bins,
            saturation_bins=self.saturation_bins,
            value_bins=self.value_bins,
        )


class ColorFindOptions(BaseFindOptions):
    """Configuration for color-based find operations.

    Port of ColorFindOptions from Qontinui framework class.

    Configures color-based pattern matching using k-means clustering,
    mean color statistics, or multi-class classification.

    It is an immutable object and must be constructed using its Builder.

    Example usage:
        options = ColorFindOptionsBuilder()
            .set_color_strategy(ColorStrategy.KMEANS)
            .set_diameter(10)
            .set_kmeans(3)
            .build()
    """

    def __init__(self, builder: "ColorFindOptionsBuilder") -> None:
        """Initialize ColorFindOptions from builder.

        Args:
            builder: The builder instance containing configuration values
        """
        super().__init__(builder)
        self.color_strategy: ColorStrategy = builder.color_strategy
        self.diameter: int = builder.diameter
        self.kmeans: int = builder.kmeans
        self.area_filtering: AreaFilteringOptions = builder.area_filtering
        self.bin_options: HSVBinOptions = builder.bin_options

    def get_find_strategy(self) -> FindStrategy:
        """Get the find strategy for color finding.

        Returns:
            COLOR strategy
        """
        return FindStrategy.COLOR

    def get_color_strategy(self) -> ColorStrategy:
        """Get the color analysis strategy.

        Returns:
            The color strategy to use
        """
        return self.color_strategy

    def get_diameter(self) -> int:
        """Get the diameter of color boxes.

        Returns:
            Diameter in pixels
        """
        return self.diameter

    def get_kmeans(self) -> int:
        """Get the number of k-means clusters.

        Returns:
            Number of clusters
        """
        return self.kmeans

    def get_area_filtering(self) -> AreaFilteringOptions:
        """Get area filtering options.

        Returns:
            Area filtering configuration
        """
        return self.area_filtering

    def get_bin_options(self) -> HSVBinOptions:
        """Get HSV bin options.

        Returns:
            HSV bin configuration
        """
        return self.bin_options

    def validate(self) -> bool:
        """Validate color find configuration.

        Returns:
            True if valid
        """
        if self.diameter < 0:
            return False
        if self.kmeans <= 0:
            return False
        if not self.area_filtering.validate():
            return False
        if not self.bin_options.validate():
            return False
        return True


class ColorFindOptionsBuilder(BaseFindOptionsBuilder["ColorFindOptionsBuilder"]):
    """Builder for constructing ColorFindOptions with a fluent API.

    Port of ColorFindOptions from Qontinui framework.Builder.
    """

    def __init__(self, original: ColorFindOptions | None = None) -> None:
        """Initialize builder.

        Args:
            original: Optional ColorFindOptions instance to copy values from
        """
        super().__init__(original)

        if original:
            self.color_strategy = original.color_strategy
            self.diameter = original.diameter
            self.kmeans = original.kmeans
            self.area_filtering = AreaFilteringOptions(
                min_area=original.area_filtering.min_area,
                max_area=original.area_filtering.max_area,
            )
            self.bin_options = HSVBinOptions(
                hue_bins=original.bin_options.hue_bins,
                saturation_bins=original.bin_options.saturation_bins,
                value_bins=original.bin_options.value_bins,
            )
        else:
            self.color_strategy = ColorStrategy.MU
            self.diameter = 5
            self.kmeans = 2
            self.area_filtering = AreaFilteringOptions()
            self.bin_options = HSVBinOptions()

    def set_color_strategy(self, strategy: ColorStrategy) -> "ColorFindOptionsBuilder":
        """Set color analysis strategy.

        Args:
            strategy: Color strategy to use

        Returns:
            This builder instance for chaining
        """
        self.color_strategy = strategy
        return self._self()

    def set_diameter(self, diameter: int) -> "ColorFindOptionsBuilder":
        """Set diameter of color boxes.

        Args:
            diameter: Diameter in pixels

        Returns:
            This builder instance for chaining
        """
        self.diameter = diameter
        return self._self()

    def set_kmeans(self, clusters: int) -> "ColorFindOptionsBuilder":
        """Set number of k-means clusters.

        Args:
            clusters: Number of clusters

        Returns:
            This builder instance for chaining
        """
        self.kmeans = clusters
        return self._self()

    def set_area_filtering(
        self, min_area: int = 1, max_area: int = -1
    ) -> "ColorFindOptionsBuilder":
        """Configure area filtering.

        Args:
            min_area: Minimum area in pixels
            max_area: Maximum area in pixels (-1 for no limit)

        Returns:
            This builder instance for chaining
        """
        self.area_filtering = AreaFilteringOptions(min_area, max_area)
        return self._self()

    def set_hsv_bins(
        self, hue: int = 12, saturation: int = 2, value: int = 1
    ) -> "ColorFindOptionsBuilder":
        """Configure HSV histogram bins.

        Args:
            hue: Number of hue bins
            saturation: Number of saturation bins
            value: Number of value bins

        Returns:
            This builder instance for chaining
        """
        self.bin_options = HSVBinOptions(hue, saturation, value)
        return self._self()

    def build(self) -> ColorFindOptions:
        """Build the immutable ColorFindOptions object.

        Returns:
            A new instance of ColorFindOptions
        """
        return ColorFindOptions(self)

    @staticmethod
    def kmeans_defaults() -> "ColorFindOptionsBuilder":
        """Create builder configured for k-means clustering.

        Returns:
            ColorFindOptionsBuilder configured for k-means
        """
        return (
            ColorFindOptionsBuilder()
            .set_color_strategy(ColorStrategy.KMEANS)
            .set_kmeans(3)
            .set_diameter(10)
        )

    @staticmethod
    def classification_defaults() -> "ColorFindOptionsBuilder":
        """Create builder configured for classification.

        Returns:
            ColorFindOptionsBuilder configured for classification
        """
        return (
            ColorFindOptionsBuilder()
            .set_color_strategy(ColorStrategy.CLASSIFICATION)
            .set_diameter(5)
            .set_area_filtering(min_area=100, max_area=5000)
        )
