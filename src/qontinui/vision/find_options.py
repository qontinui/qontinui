"""FindOptions dataclass - Configuration for vision find operations.

Comprehensive configuration for finding visual elements on screen,
including template matching, color detection, and text recognition.
Follows clean code principles with type hints and comprehensive docstrings.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from ..model.element import Region


class SearchType(Enum):
    """Type of search to perform.

    Defines how many matches to find and which to return.
    """

    FIRST = auto()  # Find first match encountered
    ALL = auto()  # Find all matches
    BEST = auto()  # Find best match by score
    EACH = auto()  # Find one of each pattern


class MatchMethod(Enum):
    """Image template matching methods.

    Different OpenCV matching algorithms available for template matching.
    """

    TM_CCOEFF = auto()  # Correlation coefficient
    TM_CCOEFF_NORMED = auto()  # Normalized correlation coefficient
    TM_CCORR = auto()  # Correlation
    TM_CCORR_NORMED = auto()  # Normalized correlation
    TM_SQDIFF = auto()  # Sum of squared differences
    TM_SQDIFF_NORMED = auto()  # Normalized SSD


class FindStrategy(Enum):
    """Vision search strategy.

    High-level strategy for finding elements on screen.
    """

    TEMPLATE = auto()  # Template matching (traditional image search)
    COLOR = auto()  # Color-based matching
    HISTOGRAM = auto()  # Histogram comparison
    TEXT = auto()  # OCR text finding
    MOTION = auto()  # Motion detection
    FEATURE = auto()  # Feature-based matching (SIFT, ORB, etc.)
    CUSTOM = auto()  # User-defined strategy


@dataclass
class FindOptions:
    """Configuration for vision find operations.

    Comprehensive options for searching visual elements on screen using
    various strategies including template matching, color detection, and OCR.

    Attributes:
        search_type: How many matches to find (FIRST, ALL, BEST, EACH)
        strategy: Vision strategy to use (TEMPLATE, COLOR, TEXT, etc.)
        match_method: Template matching method (for TEMPLATE strategy)
        similarity: Minimum similarity threshold (0.0-1.0), default 0.7
        search_regions: List of regions to restrict search
        max_matches: Maximum matches to return, default 100
        min_matches: Minimum matches required for success, default 0
        timeout: Maximum search time in seconds, default 10.0
        poll_interval: Time between search attempts in seconds, default 0.5
        return_best_match: Return only best match regardless of search_type
        search_multiple_regions: Search all regions or stop at first match
        use_multiscale: Scale invariant matching, default True
        scale_range: Tuple of (min_scale, max_scale) for multiscale search
        scale_steps: Number of scales to try for multiscale search
        cache_results: Cache results for repeated searches
        use_gpu: Use GPU acceleration if available
        parallel_search: Search regions in parallel
        save_debug_images: Save visualization of matches
        highlight_matches: Draw rectangles on matches
        metadata: Custom metadata dict for extensibility
    """

    # Search configuration
    search_type: SearchType = SearchType.FIRST
    strategy: FindStrategy = FindStrategy.TEMPLATE
    match_method: MatchMethod = MatchMethod.TM_CCOEFF_NORMED
    similarity: float = 0.7
    search_regions: list[Region] = field(default_factory=list)

    # Result limits
    max_matches: int = 100
    min_matches: int = 0

    # Timing
    timeout: float = 10.0
    poll_interval: float = 0.5

    # Search behavior
    return_best_match: bool = False
    search_multiple_regions: bool = False
    use_multiscale: bool = True
    scale_range: tuple[float, float] = (0.5, 2.0)
    scale_steps: int = 5
    cache_results: bool = False

    # Performance
    use_gpu: bool = False
    parallel_search: bool = False

    # Debug/visualization
    save_debug_images: bool = False
    highlight_matches: bool = False

    # Extensibility
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_search_region(self, region: Region) -> "FindOptions":
        """Add a search region.

        Args:
            region: Region to search in

        Returns:
            Self for fluent interface
        """
        self.search_regions.append(region)
        return self

    def add_search_regions(self, *regions: Region) -> "FindOptions":
        """Add multiple search regions.

        Args:
            *regions: Regions to search in

        Returns:
            Self for fluent interface
        """
        self.search_regions.extend(regions)
        return self

    def clear_search_regions(self) -> "FindOptions":
        """Clear all search regions.

        Returns:
            Self for fluent interface
        """
        self.search_regions.clear()
        return self

    def with_similarity(self, similarity: float) -> "FindOptions":
        """Set similarity threshold.

        Args:
            similarity: Minimum similarity (0.0-1.0)

        Returns:
            Self for fluent interface

        Raises:
            ValueError: If similarity is outside valid range
        """
        if not 0.0 <= similarity <= 1.0:
            raise ValueError(f"Similarity must be 0.0-1.0, got {similarity}")
        self.similarity = similarity
        return self

    def with_timeout(self, timeout: float) -> "FindOptions":
        """Set search timeout.

        Args:
            timeout: Maximum search time in seconds

        Returns:
            Self for fluent interface

        Raises:
            ValueError: If timeout is not positive
        """
        if timeout <= 0:
            raise ValueError(f"Timeout must be positive, got {timeout}")
        self.timeout = timeout
        return self

    def with_max_matches(self, max_matches: int) -> "FindOptions":
        """Set maximum matches to return.

        Args:
            max_matches: Maximum number of matches

        Returns:
            Self for fluent interface

        Raises:
            ValueError: If max_matches is negative
        """
        if max_matches < 0:
            raise ValueError(f"max_matches must be non-negative, got {max_matches}")
        self.max_matches = max_matches
        return self

    def with_min_matches(self, min_matches: int) -> "FindOptions":
        """Set minimum matches required.

        Args:
            min_matches: Minimum matches for success

        Returns:
            Self for fluent interface

        Raises:
            ValueError: If min_matches is negative
        """
        if min_matches < 0:
            raise ValueError(f"min_matches must be non-negative, got {min_matches}")
        self.min_matches = min_matches
        return self

    def find_all(self) -> "FindOptions":
        """Configure to find all matches.

        Returns:
            Self for fluent interface
        """
        self.search_type = SearchType.ALL
        return self

    def find_first(self) -> "FindOptions":
        """Configure to find first match.

        Returns:
            Self for fluent interface
        """
        self.search_type = SearchType.FIRST
        return self

    def find_best(self) -> "FindOptions":
        """Configure to find best match by similarity.

        Returns:
            Self for fluent interface
        """
        self.search_type = SearchType.BEST
        self.return_best_match = True
        return self

    def with_strategy(self, strategy: FindStrategy) -> "FindOptions":
        """Set vision strategy.

        Args:
            strategy: Strategy to use

        Returns:
            Self for fluent interface
        """
        self.strategy = strategy
        return self

    def with_match_method(self, method: MatchMethod) -> "FindOptions":
        """Set template matching method.

        Only relevant for TEMPLATE strategy.

        Args:
            method: Matching method to use

        Returns:
            Self for fluent interface
        """
        self.match_method = method
        return self

    def enable_multiscale(
        self, min_scale: float = 0.5, max_scale: float = 2.0, steps: int = 5
    ) -> "FindOptions":
        """Enable scale-invariant matching.

        Args:
            min_scale: Minimum scale (0.0-1.0, default 0.5)
            max_scale: Maximum scale (>1.0, default 2.0)
            steps: Number of scales to try (default 5)

        Returns:
            Self for fluent interface

        Raises:
            ValueError: If scale parameters are invalid
        """
        if min_scale <= 0 or max_scale <= 0:
            raise ValueError("Scale parameters must be positive")
        if min_scale > max_scale:
            raise ValueError(
                f"min_scale ({min_scale}) must be <= max_scale ({max_scale})"
            )
        if steps < 2:
            raise ValueError(f"steps must be >= 2, got {steps}")

        self.use_multiscale = True
        self.scale_range = (min_scale, max_scale)
        self.scale_steps = steps
        return self

    def disable_multiscale(self) -> "FindOptions":
        """Disable scale-invariant matching.

        Returns:
            Self for fluent interface
        """
        self.use_multiscale = False
        return self

    def enable_gpu(self) -> "FindOptions":
        """Enable GPU acceleration.

        Returns:
            Self for fluent interface
        """
        self.use_gpu = True
        return self

    def disable_gpu(self) -> "FindOptions":
        """Disable GPU acceleration.

        Returns:
            Self for fluent interface
        """
        self.use_gpu = False
        return self

    def enable_parallel(self) -> "FindOptions":
        """Enable parallel region search.

        Returns:
            Self for fluent interface
        """
        self.parallel_search = True
        return self

    def disable_parallel(self) -> "FindOptions":
        """Disable parallel region search.

        Returns:
            Self for fluent interface
        """
        self.parallel_search = False
        return self

    def enable_caching(self) -> "FindOptions":
        """Enable result caching.

        Returns:
            Self for fluent interface
        """
        self.cache_results = True
        return self

    def disable_caching(self) -> "FindOptions":
        """Disable result caching.

        Returns:
            Self for fluent interface
        """
        self.cache_results = False
        return self

    def enable_debug(
        self, save_images: bool = True, highlight: bool = True
    ) -> "FindOptions":
        """Enable debug visualization.

        Args:
            save_images: Save debug images (default True)
            highlight: Highlight matches (default True)

        Returns:
            Self for fluent interface
        """
        self.save_debug_images = save_images
        self.highlight_matches = highlight
        return self

    def disable_debug(self) -> "FindOptions":
        """Disable debug visualization.

        Returns:
            Self for fluent interface
        """
        self.save_debug_images = False
        self.highlight_matches = False
        return self

    def with_metadata(self, key: str, value: Any) -> "FindOptions":
        """Add custom metadata.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            Self for fluent interface
        """
        self.metadata[key] = value
        return self

    def validate(self) -> bool:
        """Validate configuration consistency.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if not 0.0 <= self.similarity <= 1.0:
            raise ValueError(f"similarity must be 0.0-1.0, got {self.similarity}")

        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")

        if self.poll_interval <= 0:
            raise ValueError(
                f"poll_interval must be positive, got {self.poll_interval}"
            )

        if self.max_matches < 0:
            raise ValueError(
                f"max_matches must be non-negative, got {self.max_matches}"
            )

        if self.min_matches < 0:
            raise ValueError(
                f"min_matches must be non-negative, got {self.min_matches}"
            )

        if self.min_matches > self.max_matches:
            raise ValueError(
                f"min_matches ({self.min_matches}) cannot exceed "
                f"max_matches ({self.max_matches})"
            )

        if self.use_multiscale:
            min_scale, max_scale = self.scale_range
            if min_scale <= 0 or max_scale <= 0:
                raise ValueError("Scale parameters must be positive")
            if min_scale > max_scale:
                raise ValueError(
                    f"min_scale ({min_scale}) must be <= max_scale ({max_scale})"
                )
            if self.scale_steps < 2:
                raise ValueError(f"scale_steps must be >= 2, got {self.scale_steps}")

        return True

    def __repr__(self) -> str:
        """String representation.

        Returns:
            Concise description of options
        """
        parts = [
            f"strategy={self.strategy.name}",
            f"similarity={self.similarity:.2f}",
            f"search_type={self.search_type.name}",
        ]
        if self.search_regions:
            parts.append(f"regions={len(self.search_regions)}")
        if self.max_matches != 100:
            parts.append(f"max={self.max_matches}")
        if self.use_multiscale:
            parts.append("multiscale")
        return f"FindOptions({', '.join(parts)})"
