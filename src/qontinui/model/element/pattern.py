"""Pattern class with mask support for pattern optimization and matching.

This is the unified Pattern class that combines basic pattern matching
with advanced mask-based optimization capabilities.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class Pattern:
    """
    A pattern with an associated mask for matching.
    This is the core class for Pattern Optimization.
    """

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
    # Pattern-level similarity threshold (overrides application-level default)
    # This is the second level of precedence (below FindOptions, above application default)
    similarity: float | None = None  # If None, uses application default
    similarity_threshold: float = 0.95  # DEPRECATED: Use 'similarity' instead
    use_color: bool = True
    scale_invariant: bool = False
    rotation_invariant: bool = False

    # Search regions - Pattern-level search regions (precedence level 2: below Options, above StateImage)
    # These are regions from StateRegions that restrict where matches are valid
    search_regions: list[dict[str, Any]] = field(
        default_factory=list
    )  # List of {id, name, x, y, width, height, referenceImageId}

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
        if self.width is None:
            self.width = self.pixel_data.shape[1]
        if self.height is None:
            self.height = self.pixel_data.shape[0]

        # Ensure mask has correct shape
        if self.mask.shape != self.pixel_data.shape[:2]:
            raise ValueError(
                f"Mask shape {self.mask.shape} doesn't match image shape {self.pixel_data.shape[:2]}"
            )

        # Handle backward compatibility: if similarity is None but similarity_threshold is set
        # This allows migration from old code using similarity_threshold
        if self.similarity is None and self.similarity_threshold != 0.95:
            self.similarity = self.similarity_threshold

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
        self, other_image: np.ndarray[Any, Any], other_mask: np.ndarray[Any, Any] | None = None
    ) -> float:
        """
        Calculate similarity between this pattern and another image.

        Args:
            other_image: Image to compare against
            other_mask: Optional mask for the other image

        Returns:
            Similarity score (0.0-1.0)
        """
        # Check shape compatibility
        if other_image.shape != self.pixel_data.shape:
            return 0.0

        # Use full mask if other_mask not provided
        if other_mask is None:
            other_mask = np.ones(other_image.shape[:2], dtype=np.float32)

        # Convert to float for precise calculation
        img1 = self.pixel_data.astype(np.float32)
        img2 = other_image.astype(np.float32)

        # Apply masks
        if len(img1.shape) == 3:
            mask1_expanded = np.expand_dims(self.mask, axis=2)
            mask2_expanded = np.expand_dims(other_mask, axis=2)
        else:
            mask1_expanded = self.mask
            mask2_expanded = other_mask

        masked_img1 = img1 * mask1_expanded
        masked_img2 = img2 * mask2_expanded

        # Combined mask (both pixels must be active)
        combined_mask = mask1_expanded * mask2_expanded

        # Calculate difference
        diff = np.abs(masked_img1 - masked_img2)

        # Normalize difference
        normalized_diff = diff / 255.0

        # Calculate similarity for active pixels only
        active_pixels = np.sum(combined_mask > 0.5)
        if active_pixels == 0:
            return 0.0

        weighted_diff_sum = np.sum(normalized_diff * combined_mask)
        avg_diff = weighted_diff_sum / active_pixels

        similarity = 1.0 - avg_diff
        return float(similarity)

    def optimize_mask(
        self,
        positive_samples: list[np.ndarray[Any, Any]],
        negative_samples: list[np.ndarray[Any, Any]] | None = None,
        method: str = "stability",
    ) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        """
        Optimize the mask based on positive and negative samples.

        Args:
            positive_samples: Images that should match
            negative_samples: Images that should not match
            method: Optimization method

        Returns:
            Tuple of (optimized mask, optimization metrics)
        """
        if method == "stability":
            return self._optimize_by_stability(positive_samples)
        elif method == "discriminative":
            return self._optimize_discriminative(positive_samples, negative_samples)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def _optimize_by_stability(
        self, positive_samples: list[np.ndarray[Any, Any]]
    ) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        """
        Optimize mask by finding stable pixels across positive samples.
        """
        if not positive_samples:
            return self.mask, {"error": "No positive samples provided"}

        # Stack all samples including original
        all_samples = np.stack([self.pixel_data] + positive_samples, axis=0)

        # Calculate pixel variance
        if len(all_samples.shape) == 4:  # Color images
            # Convert to grayscale for stability calculation
            gray_samples = np.mean(all_samples, axis=3)
        else:
            gray_samples = all_samples

        # Calculate standard deviation for each pixel
        pixel_std = np.std(gray_samples, axis=0)

        # Create stability mask (low variance = stable = important)
        max_std = np.max(pixel_std)
        if max_std > 0:
            stability = 1.0 - (pixel_std / max_std)
        else:
            stability = np.ones_like(pixel_std)

        # Threshold to create binary mask
        threshold = 0.7
        optimized_mask = np.where(stability >= threshold, 1.0, 0.0).astype(np.float32)

        # Calculate metrics
        metrics = {
            "method": "stability",
            "samples_used": len(positive_samples),
            "avg_stability": float(np.mean(stability)),
            "mask_density": float(np.sum(optimized_mask > 0.5) / optimized_mask.size),
            "threshold": threshold,
        }

        return optimized_mask, metrics

    def _optimize_discriminative(
        self,
        positive_samples: list[np.ndarray[Any, Any]],
        negative_samples: list[np.ndarray[Any, Any]] | None = None,
    ) -> tuple[np.ndarray[Any, Any], dict[str, Any]]:
        """
        Optimize mask to maximize discrimination between positive and negative samples.
        """
        # Start with stability-based mask
        stability_mask, _ = self._optimize_by_stability(positive_samples)

        if not negative_samples:
            return stability_mask, {"method": "discriminative_stability_only"}

        # For each pixel, calculate discrimination power
        h, w = self.pixel_data.shape[:2]
        discrimination = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                # Skip if pixel is not stable
                if stability_mask[y, x] < 0.5:
                    continue

                # Get pixel values from positive samples
                pos_values = [img[y, x] for img in positive_samples]
                neg_values = [img[y, x] for img in negative_samples if img.shape[:2] == (h, w)]

                if not neg_values:
                    discrimination[y, x] = stability_mask[y, x]
                    continue

                # Calculate separation between positive and negative
                pos_mean = np.mean(pos_values, axis=0)
                neg_mean = np.mean(neg_values, axis=0)

                # Distance between means
                if len(pos_mean.shape) > 0:  # Color pixel
                    dist = np.linalg.norm(pos_mean - neg_mean)
                else:  # Grayscale pixel
                    dist = abs(pos_mean - neg_mean)

                discrimination[y, x] = min(dist / 255.0, 1.0)

        # Combine stability and discrimination
        optimized_mask = stability_mask * 0.5 + discrimination * 0.5

        # Threshold
        optimized_mask = np.where(optimized_mask >= 0.5, 1.0, 0.0).astype(np.float32)

        metrics = {
            "method": "discriminative",
            "positive_samples": len(positive_samples),
            "negative_samples": len(negative_samples) if negative_samples else 0,
            "mask_density": float(np.sum(optimized_mask > 0.5) / optimized_mask.size),
        }

        return optimized_mask, metrics

    def update_mask(self, new_mask: np.ndarray[Any, Any], record_history: bool = True):
        """
        Update the pattern's mask.

        Args:
            new_mask: New mask array
            record_history: Whether to record this update in optimization history
        """
        if new_mask.shape != self.mask.shape:
            raise ValueError(
                f"New mask shape {new_mask.shape} doesn't match current shape {self.mask.shape}"
            )

        old_density = self.mask_density

        self.mask = new_mask
        self.mask_density = float(np.sum(new_mask > 0.5) / new_mask.size)
        self.updated_at = datetime.now()

        if record_history:
            self.optimization_history.append(
                {
                    "timestamp": self.updated_at.isoformat(),
                    "old_density": old_density,
                    "new_density": self.mask_density,
                    "operation": "mask_update",
                }
            )

    def add_variation(self, image: np.ndarray[Any, Any]):
        """
        Add an image variation for optimization.

        Args:
            image: Image variation (same object, different screenshot)
        """
        if image.shape != self.pixel_data.shape:
            raise ValueError("Variation shape doesn't match pattern shape")

        self.variations.append(image)

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
            "similarity": self.similarity,  # New field with proper precedence support
            "similarity_threshold": self.similarity_threshold,  # Kept for compatibility
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
        """Create Pattern from Image with full mask.

        Args:
            image: Image object or numpy array
            name: Optional pattern name
            pattern_id: Optional pattern ID

        Returns:
            Pattern instance with full mask
        """
        from .image import Image

        # Convert to numpy array
        if isinstance(image, Image):
            pixel_data = image.get_mat_bgr()
            if name is None:
                name = image.name
        elif isinstance(image, np.ndarray):
            pixel_data = image
        else:
            raise ValueError(f"Invalid image type: {type(image)}")

        if pixel_data is None:
            raise ValueError("Could not extract pixel data from image")

        # Create full mask (all pixels active)
        mask = np.ones(pixel_data.shape[:2], dtype=np.float32)

        if pattern_id is None:
            pattern_id = f"pattern_{hashlib.md5(pixel_data.tobytes()).hexdigest()[:8]}"

        return cls(
            id=pattern_id,
            name=name or "pattern",
            pixel_data=pixel_data,
            mask=mask,
            width=pixel_data.shape[1],
            height=pixel_data.shape[0],
        )

    @classmethod
    def from_file(cls, img_path: str, name: str | None = None) -> "Pattern":
        """Create Pattern from image file.

        Args:
            img_path: Path to image file
            name: Optional pattern name

        Returns:
            Pattern instance
        """
        from .image import Image

        image = Image.from_file(img_path)
        if name is None:
            from pathlib import Path

            name = Path(img_path).stem

        return cls.from_image(image, name=name)

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
        self, find_options_similarity: float | None = None, application_default: float = 0.7
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
        # Also update similarity_threshold for backward compatibility
        self.similarity_threshold = threshold
        return self

    def with_search_region(self, region: Any) -> "Pattern":
        """Set search region and return self for chaining.

        Args:
            region: Search region (Region object)

        Returns:
            Self for method chaining

        Note:
            This method stores the search region for later use but doesn't
            directly affect Pattern matching. The Find class should use this
            information when configuring search operations.
        """
        # Pattern class doesn't store search_region directly, but we can
        # add it for compatibility with StateImage.get_pattern()
        # In practice, search region is handled by the Find class
        return self

    @classmethod
    def from_match(cls, match: Any, pattern_id: str | None = None) -> "Pattern":
        """Create Pattern from a Match object.

        Args:
            match: Match object to create pattern from
            pattern_id: Optional custom ID

        Returns:
            Pattern instance
        """
        if pattern_id is None:
            pattern_id = f"pattern_{hashlib.md5(str(match).encode()).hexdigest()[:8]}"

        # Extract pixel data from match
        pixel_data = (
            match.image.get_mat_bgr() if match.image else np.zeros((10, 10, 3), dtype=np.uint8)
        )

        # Create full mask
        mask = np.ones(pixel_data.shape[:2], dtype=np.float32)

        region = match.get_region()
        return cls(
            id=pattern_id,
            name=match.name or "pattern_from_match",
            pixel_data=pixel_data,
            mask=mask,
            x=region.x if region else 0,
            y=region.y if region else 0,
            width=region.width if region else pixel_data.shape[1],
            height=region.height if region else pixel_data.shape[0],
            similarity_threshold=match.score if hasattr(match, "score") else 0.95,
        )

    @classmethod
    def from_state_image(cls, state_image: Any, pattern_id: str | None = None) -> "Pattern":
        """
        Create a Pattern from a StateImage.

        Args:
            state_image: StateImage object
            pattern_id: Optional custom ID

        Returns:
            Pattern instance
        """
        if pattern_id is None:
            pattern_id = f"pattern_{state_image.id}"

        # Extract pixel data and mask
        pixel_data = state_image.pixel_data
        mask = state_image.mask if state_image.mask is not None else np.ones(pixel_data.shape[:2])

        return cls(
            id=pattern_id,
            name=state_image.name,
            pixel_data=pixel_data,
            mask=mask,
            x=state_image.x,
            y=state_image.y,
            width=state_image.x2 - state_image.x,
            height=state_image.y2 - state_image.y,
            mask_density=state_image.mask_density if hasattr(state_image, "mask_density") else 1.0,
            mask_type="imported",
            tags=state_image.tags if hasattr(state_image, "tags") else [],
            created_at=(
                state_image.created_at if hasattr(state_image, "created_at") else datetime.now()
            ),
        )
