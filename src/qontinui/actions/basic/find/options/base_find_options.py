"""Base find options - ported from Qontinui framework.

Abstract base class for all find operation configurations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

from .....model.element.region import Region
from ....action_config import ActionConfig


class SearchType(Enum):
    """Type of search to perform.

    Port of search from Qontinui framework type options.
    """

    FIRST = auto()  # Find first match
    ALL = auto()  # Find all matches
    BEST = auto()  # Find best match by score
    EACH = auto()  # Find one of each pattern


class FindStrategy(Enum):
    """Available find strategies.

    Port of FindStrategy from Qontinui framework enum.
    """

    TEMPLATE = auto()  # Traditional template matching
    COLOR = auto()  # Color-based matching
    HISTOGRAM = auto()  # Histogram comparison
    TEXT = auto()  # OCR text finding
    MOTION = auto()  # Motion detection
    FEATURE = auto()  # Feature-based matching
    CUSTOM = auto()  # User-defined strategy
    ML_OBJECT = auto()  # Machine learning object detection (enhancement)


@dataclass
class BaseFindOptions(ActionConfig, ABC):
    """Abstract base configuration for all find operations.

    Port of BaseFindOptions from Qontinui framework class.

    Provides common configuration that all find strategies share,
    including search regions, similarity thresholds, and result limits.
    """

    # Search configuration
    search_type: SearchType = SearchType.FIRST
    search_regions: list[Region] = field(default_factory=list)
    similarity: float = 0.7  # Minimum similarity threshold (0.0-1.0)

    # Result limits
    max_matches: int = 100  # Maximum matches to return
    min_matches: int = 0  # Minimum matches required for success

    # Timing
    timeout: float = 10.0  # Maximum search time in seconds
    poll_interval: float = 0.5  # Time between search attempts

    # Search behavior
    search_multiple_regions: bool = False  # Search all regions vs first with match
    return_best_match: bool = False  # Return only the best match
    cache_results: bool = False  # Cache results for repeated searches

    # Performance
    use_gpu: bool = False  # Use GPU acceleration if available
    parallel_search: bool = False  # Search regions in parallel

    # Debug/visualization
    save_debug_images: bool = False  # Save visualization of matches
    highlight_matches: bool = False  # Draw rectangles on matches

    @abstractmethod
    def get_strategy(self) -> FindStrategy:
        """Get the find strategy for this configuration.

        Returns:
            The find strategy to use
        """
        pass

    def add_search_region(self, region: Region) -> "BaseFindOptions":
        """Add a search region.

        Args:
            region: Region to search in

        Returns:
            Self for fluent interface
        """
        self.search_regions.append(region)
        return self

    def add_search_regions(self, *regions: Region) -> "BaseFindOptions":
        """Add multiple search regions.

        Args:
            *regions: Regions to search in

        Returns:
            Self for fluent interface
        """
        self.search_regions.extend(regions)
        return self

    def clear_search_regions(self) -> "BaseFindOptions":
        """Clear all search regions.

        Returns:
            Self for fluent interface
        """
        self.search_regions.clear()
        return self

    def with_similarity(self, similarity: float) -> "BaseFindOptions":
        """Set similarity threshold.

        Args:
            similarity: Minimum similarity (0.0-1.0)

        Returns:
            Self for fluent interface
        """
        self.similarity = max(0.0, min(1.0, similarity))
        return self

    def with_timeout(self, timeout: float) -> "BaseFindOptions":
        """Set search timeout.

        Args:
            timeout: Maximum search time in seconds

        Returns:
            Self for fluent interface
        """
        self.timeout = timeout
        return self

    def find_all(self) -> "BaseFindOptions":
        """Configure to find all matches.

        Returns:
            Self for fluent interface
        """
        self.search_type = SearchType.ALL
        return self

    def find_first(self) -> "BaseFindOptions":
        """Configure to find first match.

        Returns:
            Self for fluent interface
        """
        self.search_type = SearchType.FIRST
        return self

    def find_best(self) -> "BaseFindOptions":
        """Configure to find best match.

        Returns:
            Self for fluent interface
        """
        self.search_type = SearchType.BEST
        self.return_best_match = True
        return self

    def with_max_matches(self, max_matches: int) -> "BaseFindOptions":
        """Set maximum matches to return.

        Args:
            max_matches: Maximum number of matches

        Returns:
            Self for fluent interface
        """
        self.max_matches = max_matches
        return self

    def with_min_matches(self, min_matches: int) -> "BaseFindOptions":
        """Set minimum matches required.

        Args:
            min_matches: Minimum matches for success

        Returns:
            Self for fluent interface
        """
        self.min_matches = min_matches
        return self

    def enable_gpu(self) -> "BaseFindOptions":
        """Enable GPU acceleration.

        Returns:
            Self for fluent interface
        """
        self.use_gpu = True
        return self

    def enable_parallel(self) -> "BaseFindOptions":
        """Enable parallel search.

        Returns:
            Self for fluent interface
        """
        self.parallel_search = True
        return self

    def enable_caching(self) -> "BaseFindOptions":
        """Enable result caching.

        Returns:
            Self for fluent interface
        """
        self.cache_results = True
        return self

    def enable_debug(self) -> "BaseFindOptions":
        """Enable debug visualization.

        Returns:
            Self for fluent interface
        """
        self.save_debug_images = True
        self.highlight_matches = True
        return self

    def validate(self) -> bool:
        """Validate configuration.

        Returns:
            True if configuration is valid
        """
        if self.similarity < 0 or self.similarity > 1:
            return False
        if self.timeout <= 0:
            return False
        if self.max_matches < 0:
            return False
        if self.min_matches < 0:
            return False
        if self.min_matches > self.max_matches:
            return False
        return True

    def __repr__(self) -> str:
        """String representation.

        Returns:
            Description of options
        """
        parts = [
            f"strategy={self.get_strategy().name}",
            f"similarity={self.similarity:.2f}",
            f"search_type={self.search_type.name}",
        ]
        if self.search_regions:
            parts.append(f"regions={len(self.search_regions)}")
        if self.max_matches != 100:
            parts.append(f"max={self.max_matches}")
        return f"{self.__class__.__name__}({', '.join(parts)})"
