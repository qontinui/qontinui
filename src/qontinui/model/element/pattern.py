"""Pattern class with mask support for pattern optimization and matching.

This is the unified Pattern class that combines basic pattern matching
with advanced mask-based optimization capabilities. Port of Brobot's Pattern class.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np

from .pattern_factory import PatternFactory
from .pattern_optimizer import PatternOptimizer
from .similarity_calculator import SimilarityCalculator

if TYPE_CHECKING:
    from ..search_regions import SearchRegions


@dataclass
class Pattern:
    """
    A pattern with an associated mask for matching.
    This is the core class for Pattern Optimization.
    Port of Brobot's Pattern class with mask-based extensions.
    """

    # Class-level optimizer (shared across all patterns)
    _optimizer: PatternOptimizer = field(
        default_factory=PatternOptimizer, init=False, repr=False
    )

    # Class-level similarity calculator (shared across all patterns)
    _similarity_calculator: SimilarityCalculator = field(
        default_factory=SimilarityCalculator, init=False, repr=False
    )

    id: str
    name: str
    pixel_data: np.ndarray[Any, Any]  # Original image data (H x W x C)
    mask: np.ndarray[Any, Any]  # Mask array (H x W), values 0.0-1.0

    # Bounding box
    x: int = 0
    y: int = 0
    width: int | None = None
    height: int | None = None

    # Mask metadata
    mask_density: float = 1.0  # Percentage of active pixels
    mask_type: str = "full"  # Type of mask generation used

    # Pattern metadata
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # Matching parameters
    # Pattern-level similarity threshold
    #
    # Similarity Priority Cascade (highest to lowest):
    # 1. FindOptions.similarity (action-level, explicit) - HIGHEST
    # 2. Pattern.similarity (THIS LEVEL - image-level override)
    # 3. StateImage.threshold (state image-level from JSON config)
    # 4. QontinuiSettings.similarity_threshold (project config = 0.85)
    # 5. Library default (action_defaults = 0.7) - LOWEST
    #
    # If None: defers to lower priority levels (StateImage, project config, library default)
    # If set: overrides StateImage/project/library defaults, but can be overridden by FindOptions
    similarity: float | None = None
    use_color: bool = True
    scale_invariant: bool = False
    rotation_invariant: bool = False

    # Brobot Pattern properties
    fixed: bool = (
        False  # An image that should always appear in the same location has fixed==true
    )
    dynamic: bool = False  # Dynamic images cannot be found using pattern matching

    # Search regions - Following Brobot's model with SearchRegions object
    search_regions: "SearchRegions" = field(default_factory=lambda: None)  # type: ignore

    # Statistics
    match_count: int = 0
    success_rate: float = 0.0
    avg_match_time: float = 0.0

    # Optimization data
    optimization_history: list[dict[str, Any]] = field(default_factory=list)
    variations: list[np.ndarray[Any, Any]] = field(
        default_factory=list
    )  # Image variations for optimization

    # Additional attributes for compatibility
    path: str | None = None
    owner_state_name: str | None = None

    def __post_init__(self):
        """Initialize computed fields."""
        from ..search_regions import SearchRegions

        if self.width is None:
            self.width = self.pixel_data.shape[1]
        if self.height is None:
            self.height = self.pixel_data.shape[0]

        # Ensure mask has correct shape
        if self.mask.shape != self.pixel_data.shape[:2]:
            raise ValueError(
                f"Mask shape {self.mask.shape} doesn't match image shape {self.pixel_data.shape[:2]}"
            )

        # Initialize SearchRegions if not set
        if self.search_regions is None:
            self.search_regions = SearchRegions()

    @property
    def masked_pixel_hash(self) -> str:
        """Generate hash of masked pixel data."""
        masked_data = self.pixel_data * np.expand_dims(self.mask, axis=2)
        data_bytes = masked_data.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()

    @property
    def active_pixel_count(self) -> int:
        """Count of active (non-masked) pixels."""
        return int(np.sum(self.mask > 0.5))

    @property
    def total_pixel_count(self) -> int:
        """Total pixel count."""
        return int(self.mask.size)

    def calculate_similarity(
        self,
        other_image: np.ndarray[Any, Any],
        other_mask: np.ndarray[Any, Any] | None = None,
    ) -> float:
        """
        Calculate similarity between this pattern and another image.

        Delegates to SimilarityCalculator for the actual computation.

        Args:
            other_image: Image to compare against
            other_mask: Optional mask for the other image

        Returns:
            Similarity score (0.0-1.0)
        """
        return self._similarity_calculator.calculate_similarity(
            self.pixel_data, self.mask, other_image, other_mask
        )

    def optimize_mask(
        self,
        positive_samples: list[np.ndarray[Any, Any]],
        negative_samples: list[np.ndarray[Any, Any]] | None = None,
        method: str = "stability",
    ) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        """Optimize the mask based on positive and negative samples.

        Delegates to PatternOptimizer for the actual optimization logic.

        Args:
            positive_samples: Images that should match
            negative_samples: Images that should not match
            method: Optimization method ("stability" or "discriminative")

        Returns:
            Tuple of (optimized mask, optimization metrics)
        """
        return self._optimizer.optimize_mask(
            self.pixel_data,
            self.mask,
            positive_samples,
            negative_samples,
            method,
        )

    def update_mask(self, new_mask: np.ndarray[Any, Any], record_history: bool = True):
        """Update the pattern's mask.

        Delegates to PatternOptimizer for update tracking.

        Args:
            new_mask: New mask array
            record_history: Whether to record this update in optimization history
        """
        self._optimizer.update_mask(self, new_mask, record_history)

    def add_variation(self, image: np.ndarray[Any, Any]):
        """Add an image variation for optimization.

        Delegates to PatternOptimizer for variation management.

        Args:
            image: Image variation (same object, different screenshot)
        """
        self._optimizer.add_variation(self, image)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "mask_density": self.mask_density,
            "mask_type": self.mask_type,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "similarity": self.similarity,
            "use_color": self.use_color,
            "scale_invariant": self.scale_invariant,
            "rotation_invariant": self.rotation_invariant,
            "match_count": self.match_count,
            "success_rate": self.success_rate,
            "avg_match_time": self.avg_match_time,
            "active_pixels": self.active_pixel_count,
            "total_pixels": self.total_pixel_count,
            "pixel_hash": self.masked_pixel_hash,
            "variation_count": len(self.variations),
            "optimization_count": len(self.optimization_history),
        }

    @classmethod
    def from_image(
        cls, image: Any, name: str | None = None, pattern_id: str | None = None
    ) -> "Pattern":
        """Create Pattern from Image (delegates to factory).

        Args:
            image: Image object or numpy array
            name: Optional pattern name
            pattern_id: Optional pattern ID

        Returns:
            Pattern instance with full mask
        """
        return PatternFactory.from_image(image, name, pattern_id)

    @classmethod
    def from_file(cls, img_path: str, name: str | None = None) -> "Pattern":
        """Create Pattern from image file (delegates to factory).

        Args:
            img_path: Path to image file
            name: Optional pattern name

        Returns:
            Pattern instance
        """
        return PatternFactory.from_file(img_path, name)

    @property
    def image(self) -> Any:
        """Get the image data as an Image object.

        Returns:
            Image object created from pixel_data
        """
        from .image import Image

        return Image.from_numpy(self.pixel_data, name=self.name)

    def get_image(self) -> Any:
        """Get the image data as an Image object.

        Returns:
            Image object created from pixel_data
        """
        return self.image

    def has_image(self) -> bool:
        """Check if pattern has image data.

        Returns:
            True if pixel_data exists
        """
        return self.pixel_data is not None and self.pixel_data.size > 0

    def get_imgpath(self) -> str | None:
        """Get image path.

        Returns:
            Image path if available, None otherwise
        """
        return self.path

    def get_b_image(self) -> Any:
        """Get the secondary/background image if available.

        Returns:
            Secondary image or None
        """
        # This is a placeholder - actual implementation may vary
        # depending on pattern type (some patterns may have background images)
        return None

    def get_effective_similarity(
        self,
        find_options_similarity: float | None = None,
        application_default: float = 0.7,
    ) -> float:
        """Get the effective similarity threshold with proper precedence.

        Similarity precedence (highest to lowest):
        1. FindOptions similarity (passed as parameter) - HIGHEST PRECEDENCE
        2. Pattern-level similarity (self.similarity)
        3. Application-level default similarity - LOWEST PRECEDENCE

        This matches Brobot's precedence order where:
        - ActionOptions/FindOptions similarity takes precedence over everything
        - Pattern similarity is checked second
        - Application properties (Settings.MinSimilarity in Brobot) is the fallback

        Args:
            find_options_similarity: Similarity from FindOptions (highest precedence)
            application_default: Application-level default similarity (lowest precedence)

        Returns:
            Effective similarity threshold to use (0.0-1.0)

        Example:
            # Pattern with no explicit similarity uses application default
            pattern = Pattern(...)
            sim = pattern.get_effective_similarity()  # Returns 0.7

            # Pattern with explicit similarity uses that
            pattern.similarity = 0.85
            sim = pattern.get_effective_similarity()  # Returns 0.85

            # FindOptions overrides pattern similarity
            sim = pattern.get_effective_similarity(find_options_similarity=0.95)  # Returns 0.95
        """
        # 1. Highest precedence: FindOptions similarity
        if find_options_similarity is not None:
            return find_options_similarity

        # 2. Medium precedence: Pattern-level similarity
        if self.similarity is not None:
            return self.similarity

        # 3. Lowest precedence: Application-level default
        return application_default

    def with_similarity(self, threshold: float) -> "Pattern":
        """Set similarity threshold and return self for chaining.

        Args:
            threshold: New similarity threshold (0.0-1.0)

        Returns:
            Self for method chaining

        Note:
            This sets the Pattern-level similarity which has medium precedence.
            FindOptions similarity (if provided) will override this.
        """
        self.similarity = threshold
        return self

    def with_search_region(self, region: Any) -> "Pattern":
        """Set search region and return self for chaining.

        Args:
            region: Search region (Region object)

        Returns:
            Self for method chaining
        """
        self.search_regions.add_region(region)
        return self

    def add_search_region(self, region: Any) -> None:
        """Add a search region where this pattern should be looked for.

        Args:
            region: The region to add to the search areas
        """
        self.search_regions.add_search_regions(region)

    def reset_fixed_search_region(self) -> None:
        """Reset the fixed search region, allowing the pattern to be found anywhere."""
        self.search_regions.reset_fixed_region()

    def set_search_regions_to(self, *regions: Any) -> None:
        """Set the search regions to the specified regions.

        Args:
            *regions: Variable number of Region objects
        """
        self.search_regions.set_regions(list(regions))

    def get_regions(self) -> list[Any]:
        """Get all search regions for this pattern.

        If the image has a fixed location and has already been found, the region
        where it was found is returned. Otherwise, all regions are returned.

        Returns:
            All usable regions
        """
        return self.search_regions.get_regions(self.fixed)

    def get_regions_for_search(self) -> list[Any]:
        """Get regions for searching, with full-screen default if none are configured.

        This is the method to use when actually performing searches.

        Returns:
            Regions for searching (never empty)
        """
        return self.search_regions.get_regions_for_search()

    def get_region(self) -> Any:
        """Get a region for this pattern.

        If the image has a fixed location and has already been found, the region
        where it was found is returned. If there are multiple regions, returns
        a random selection.

        Returns:
            A region
        """
        return self.search_regions.get_fixed_if_defined_or_random_region(self.fixed)

    def is_defined(self) -> bool:
        """Check if this pattern has defined search regions.

        Returns:
            True if regions are defined, False otherwise
        """
        return self.search_regions.is_defined(self.fixed)

    @classmethod
    def from_match(cls, match: Any, pattern_id: str | None = None) -> "Pattern":
        """Create Pattern from a Match object (delegates to factory).

        Args:
            match: Match object to create pattern from
            pattern_id: Optional custom ID

        Returns:
            Pattern instance
        """
        return PatternFactory.from_match(match, pattern_id)

    @classmethod
    def from_state_image(
        cls, state_image: Any, pattern_id: str | None = None
    ) -> "Pattern":
        """Create a Pattern from a StateImage (delegates to factory).

        Args:
            state_image: StateImage object
            pattern_id: Optional custom ID

        Returns:
            Pattern instance
        """
        return PatternFactory.from_state_image(state_image, pattern_id)
