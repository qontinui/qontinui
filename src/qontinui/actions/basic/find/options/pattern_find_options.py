"""Pattern find options - ported from Qontinui framework.

Configuration for pattern/template matching operations.
"""

from dataclasses import dataclass, field
from enum import Enum, auto

from .....model.element.pattern import Pattern
from .....model.state.state_image import StateImage
from .base_find_options import BaseFindOptions, FindStrategy


class MatchMethod(Enum):
    """Template matching methods.

    Port of match from Qontinui framework methods.
    """

    CORRELATION = auto()  # Cross-correlation
    CORRELATION_NORMED = auto()  # Normalized cross-correlation
    CORRELATION_COEFFICIENT = auto()  # Correlation coefficient
    CORRELATION_COEFFICIENT_NORMED = auto()  # Normalized correlation coefficient
    SQUARED_DIFFERENCE = auto()  # Squared difference
    SQUARED_DIFFERENCE_NORMED = auto()  # Normalized squared difference


@dataclass
class PatternFindOptions(BaseFindOptions):
    """Configuration for pattern/template matching.

    Port of PatternFindOptions from Qontinui framework class.

    Configures how to find visual patterns using template matching
    or other image-based search methods.
    """

    # Patterns to search for
    patterns: list[Pattern] = field(default_factory=list)
    state_images: list[StateImage] = field(default_factory=list)

    # Matching configuration
    match_method: MatchMethod = MatchMethod.CORRELATION_COEFFICIENT_NORMED
    scale_invariant: bool = False  # Search at multiple scales
    rotation_invariant: bool = False  # Search at multiple rotations

    # Scale search parameters
    min_scale: float = 0.8  # Minimum scale factor
    max_scale: float = 1.2  # Maximum scale factor
    scale_step: float = 0.05  # Scale increment

    # Rotation search parameters
    min_rotation: float = -15.0  # Minimum rotation in degrees
    max_rotation: float = 15.0  # Maximum rotation in degrees
    rotation_step: float = 5.0  # Rotation increment

    # Color handling
    use_grayscale: bool = True  # Convert to grayscale for matching
    use_color_reduction: bool = False  # Reduce colors before matching
    color_tolerance: float = 0.1  # Color matching tolerance

    # Edge detection preprocessing
    use_edges: bool = False  # Use edge detection
    edge_threshold1: float = 50.0  # Canny edge lower threshold
    edge_threshold2: float = 150.0  # Canny edge upper threshold

    # Performance optimization
    downsample_factor: float = 1.0  # Downsample images for speed
    use_image_pyramid: bool = False  # Use pyramid for multi-scale
    early_termination_threshold: float = 0.95  # Stop if match exceeds this

    # Match filtering
    non_max_suppression: bool = True  # Remove overlapping matches
    nms_threshold: float = 0.5  # Overlap threshold for NMS
    min_distance_between_matches: int = 10  # Minimum pixel distance

    def get_strategy(self) -> FindStrategy:
        """Get the find strategy for pattern matching.

        Returns:
            TEMPLATE strategy
        """
        return FindStrategy.TEMPLATE

    def add_pattern(self, pattern: Pattern) -> "PatternFindOptions":
        """Add a pattern to search for.

        Args:
            pattern: Pattern to find

        Returns:
            Self for fluent interface
        """
        self.patterns.append(pattern)
        return self

    def add_patterns(self, *patterns: Pattern) -> "PatternFindOptions":
        """Add multiple patterns.

        Args:
            *patterns: Patterns to find

        Returns:
            Self for fluent interface
        """
        self.patterns.extend(patterns)
        return self

    def add_state_image(self, state_image: StateImage) -> "PatternFindOptions":
        """Add a state image to search for.

        Args:
            state_image: State image to find

        Returns:
            Self for fluent interface
        """
        self.state_images.append(state_image)
        # Also add its patterns
        if hasattr(state_image, "patterns"):
            self.patterns.extend(state_image.patterns)
        return self

    def add_state_images(self, *state_images: StateImage) -> "PatternFindOptions":
        """Add multiple state images.

        Args:
            *state_images: State images to find

        Returns:
            Self for fluent interface
        """
        for si in state_images:
            self.add_state_image(si)
        return self

    def with_method(self, method: MatchMethod) -> "PatternFindOptions":
        """Set matching method.

        Args:
            method: Template matching method

        Returns:
            Self for fluent interface
        """
        self.match_method = method
        return self

    def enable_scale_invariant(
        self, min_scale: float = 0.8, max_scale: float = 1.2
    ) -> "PatternFindOptions":
        """Enable scale-invariant search.

        Args:
            min_scale: Minimum scale factor
            max_scale: Maximum scale factor

        Returns:
            Self for fluent interface
        """
        self.scale_invariant = True
        self.min_scale = min_scale
        self.max_scale = max_scale
        return self

    def enable_rotation_invariant(
        self, min_rotation: float = -15.0, max_rotation: float = 15.0
    ) -> "PatternFindOptions":
        """Enable rotation-invariant search.

        Args:
            min_rotation: Minimum rotation in degrees
            max_rotation: Maximum rotation in degrees

        Returns:
            Self for fluent interface
        """
        self.rotation_invariant = True
        self.min_rotation = min_rotation
        self.max_rotation = max_rotation
        return self

    def with_edge_detection(
        self, threshold1: float = 50.0, threshold2: float = 150.0
    ) -> "PatternFindOptions":
        """Enable edge detection preprocessing.

        Args:
            threshold1: Lower threshold for Canny
            threshold2: Upper threshold for Canny

        Returns:
            Self for fluent interface
        """
        self.use_edges = True
        self.edge_threshold1 = threshold1
        self.edge_threshold2 = threshold2
        return self

    def with_color_tolerance(self, tolerance: float) -> "PatternFindOptions":
        """Set color matching tolerance.

        Args:
            tolerance: Color tolerance (0.0-1.0)

        Returns:
            Self for fluent interface
        """
        self.color_tolerance = tolerance
        self.use_grayscale = False  # Use color matching
        return self

    def downsample(self, factor: float) -> "PatternFindOptions":
        """Set downsampling factor for speed.

        Args:
            factor: Downsample factor (1.0 = no downsampling)

        Returns:
            Self for fluent interface
        """
        self.downsample_factor = factor
        return self

    def with_nms(self, threshold: float = 0.5) -> "PatternFindOptions":
        """Configure non-maximum suppression.

        Args:
            threshold: Overlap threshold for NMS

        Returns:
            Self for fluent interface
        """
        self.non_max_suppression = True
        self.nms_threshold = threshold
        return self

    def validate(self) -> bool:
        """Validate pattern configuration.

        Returns:
            True if valid
        """
        if not super().validate():
            return False
        if not self.patterns and not self.state_images:
            return False  # Need something to search for
        if self.min_scale <= 0 or self.max_scale <= 0:
            return False
        if self.min_scale > self.max_scale:
            return False
        if self.downsample_factor <= 0 or self.downsample_factor > 1:
            return False
        return True

    @staticmethod
    def default() -> "PatternFindOptions":
        """Create default pattern find options.

        Returns:
            Default PatternFindOptions
        """
        return PatternFindOptions()

    @staticmethod
    def fast() -> "PatternFindOptions":
        """Create fast pattern find options.

        Optimized for speed over accuracy.

        Returns:
            Fast PatternFindOptions
        """
        return PatternFindOptions(
            downsample_factor=0.5,
            use_grayscale=True,
            similarity=0.8,
            early_termination_threshold=0.9,
        )

    @staticmethod
    def accurate() -> "PatternFindOptions":
        """Create accurate pattern find options.

        Optimized for accuracy over speed.

        Returns:
            Accurate PatternFindOptions
        """
        return PatternFindOptions(
            scale_invariant=True,
            rotation_invariant=True,
            use_grayscale=False,
            similarity=0.95,
            downsample_factor=1.0,
        )
