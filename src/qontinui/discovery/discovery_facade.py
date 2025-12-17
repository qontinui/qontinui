"""State Discovery Facade - Unified Interface for State Discovery Operations.

This module provides a high-level facade for discovering application states from
screenshot sequences. It orchestrates the various detection and analysis components
into a simple, unified API.

Architecture:
    The facade coordinates three main subsystems:
    1. Pixel Stability Analysis - Finds stable visual elements across screenshots
    2. Differential Consistency Detection - Finds regions that change consistently
    3. State Construction - Builds complete State objects from detected elements

Usage:
    >>> from qontinui.discovery import StateDiscoveryFacade, DiscoveryConfig
    >>>
    >>> # Simple usage with defaults
    >>> facade = StateDiscoveryFacade()
    >>> result = facade.discover_states(screenshots)
    >>>
    >>> print(f"Found {len(result.states)} states")
    >>> print(f"Found {len(result.state_images)} state images")
    >>>
    >>> # Advanced usage with transitions
    >>> result = facade.discover_states_with_transitions(
    ...     screenshots=screenshots,
    ...     transitions=transition_pairs,
    ... )

The facade is designed to be:
- Simple: One method call for common use cases
- Flexible: Configurable for advanced scenarios
- Stateless: Each call is independent (though caching can be enabled)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from .models import AnalysisConfig, AnalysisResult, DiscoveredState, StateImage

if TYPE_CHECKING:
    from .state_construction.state_builder import TransitionInfo


class DiscoveryAlgorithm(str, Enum):
    """Available state discovery algorithms."""

    PIXEL_STABILITY = "pixel_stability"
    """Fast analysis using pixel stability matrix. Best for screenshots of similar states."""

    DIFFERENTIAL_CONSISTENCY = "differential_consistency"
    """Analyzes before/after transition pairs. Best for finding state boundaries."""

    COMBINED = "combined"
    """Uses both algorithms and merges results. Most thorough but slower."""


@dataclass
class DiscoveryConfig:
    """Configuration for state discovery operations.

    Attributes:
        algorithm: Which discovery algorithm to use
        min_region_size: Minimum size (width, height) for detected regions
        max_region_size: Maximum size (width, height) for detected regions
        stability_threshold: Minimum stability score (0-1) for stable elements
        consistency_threshold: Minimum consistency score (0-1) for state boundaries
        min_screenshots: Minimum screenshots required for analysis
        similarity_threshold: Threshold for matching similar regions (0-1)
        enable_ocr_naming: Whether to use OCR for generating state names
        enable_state_grouping: Whether to group StateImages into States
    """

    algorithm: DiscoveryAlgorithm = DiscoveryAlgorithm.PIXEL_STABILITY
    min_region_size: tuple[int, int] = (20, 20)
    max_region_size: tuple[int, int] = (500, 500)
    stability_threshold: float = 0.98
    consistency_threshold: float = 0.7
    min_screenshots: int = 2
    similarity_threshold: float = 0.95
    enable_ocr_naming: bool = True
    enable_state_grouping: bool = True

    def to_analysis_config(self) -> AnalysisConfig:
        """Convert to internal AnalysisConfig."""
        return AnalysisConfig(
            min_region_size=self.min_region_size,
            max_region_size=self.max_region_size,
            stability_threshold=self.stability_threshold,
            min_screenshots_present=self.min_screenshots,
            similarity_threshold=self.similarity_threshold,
            enable_cooccurrence_analysis=self.enable_state_grouping,
        )


@dataclass
class DiscoveryResult:
    """Results from state discovery analysis.

    Attributes:
        states: Discovered application states
        state_images: Visual elements that identify states
        transitions: Detected state transitions
        statistics: Processing statistics and metadata
        config: Configuration used for this analysis
    """

    states: list[DiscoveredState] = field(default_factory=list)
    state_images: list[StateImage] = field(default_factory=list)
    transitions: list[dict[str, Any]] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)
    config: DiscoveryConfig | None = None

    @classmethod
    def from_analysis_result(
        cls, result: AnalysisResult, config: DiscoveryConfig | None = None
    ) -> DiscoveryResult:
        """Create DiscoveryResult from internal AnalysisResult."""
        return cls(
            states=result.states,
            state_images=result.state_images,
            transitions=[t.to_dict() for t in result.transitions],
            statistics=result.statistics,
            config=config,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "states": [s.to_dict() for s in self.states],
            "state_images": [si.to_dict() for si in self.state_images],
            "transitions": self.transitions,
            "statistics": self.statistics,
            "states_count": len(self.states),
            "state_images_count": len(self.state_images),
        }


class StateDiscoveryFacade:
    """Unified facade for state discovery operations.

    This facade provides a simple, high-level interface to the state discovery
    subsystem. It handles the complexity of coordinating multiple detection
    algorithms and presents a clean API for common use cases.

    Example:
        >>> facade = StateDiscoveryFacade()
        >>>
        >>> # Discover states from screenshots
        >>> result = facade.discover_states(screenshots)
        >>>
        >>> # Access discovered states
        >>> for state in result.states:
        ...     print(f"State: {state.name} with {len(state.state_image_ids)} images")
        >>>
        >>> # Access state images
        >>> for img in result.state_images:
        ...     print(f"Image: {img.name} at ({img.x}, {img.y})")

    The facade is stateless - each method call operates independently.
    For incremental discovery or caching, use the underlying components directly.
    """

    def __init__(self, config: DiscoveryConfig | None = None):
        """Initialize the discovery facade.

        Args:
            config: Default configuration for discovery operations.
                    Can be overridden per-method call.
        """
        self._default_config = config or DiscoveryConfig()
        self._pixel_analyzer = None
        self._diff_detector = None
        self._state_builder = None

    @property
    def pixel_analyzer(self):
        """Lazy-load PixelStabilityMatrixAnalyzer."""
        if self._pixel_analyzer is None:
            from .pixel_stability_matrix_analyzer import PixelStabilityMatrixAnalyzer

            self._pixel_analyzer = PixelStabilityMatrixAnalyzer(
                self._default_config.to_analysis_config()
            )
        return self._pixel_analyzer

    @property
    def diff_detector(self):
        """Lazy-load DifferentialConsistencyDetector."""
        if self._diff_detector is None:
            from .state_detection.differential_consistency_detector import (
                DifferentialConsistencyDetector,
            )

            self._diff_detector = DifferentialConsistencyDetector()
        return self._diff_detector

    @property
    def state_builder(self):
        """Lazy-load StateBuilder."""
        if self._state_builder is None:
            from .state_construction.state_builder import StateBuilder

            self._state_builder = StateBuilder(
                consistency_threshold=self._default_config.consistency_threshold,
            )
        return self._state_builder

    def discover_states(
        self,
        screenshots: list[np.ndarray[Any, Any]],
        config: DiscoveryConfig | None = None,
        region: tuple[int, int, int, int] | None = None,
    ) -> DiscoveryResult:
        """Discover states from a sequence of screenshots.

        This is the primary entry point for state discovery. Given a sequence of
        screenshots, it identifies stable visual elements (StateImages) and groups
        them into application states.

        Args:
            screenshots: List of screenshot images as numpy arrays (BGR format)
            config: Configuration overrides (uses default if not provided)
            region: Optional (x, y, x2, y2) to limit analysis to specific region

        Returns:
            DiscoveryResult containing discovered states, images, and statistics

        Raises:
            ValueError: If screenshots list is empty or too short

        Example:
            >>> screenshots = [cv2.imread(f) for f in glob.glob("captures/*.png")]
            >>> result = facade.discover_states(screenshots)
            >>> print(f"Found {len(result.states)} distinct states")
        """
        config = config or self._default_config

        if not screenshots:
            raise ValueError("Screenshots list cannot be empty")

        if len(screenshots) < config.min_screenshots:
            raise ValueError(
                f"Need at least {config.min_screenshots} screenshots, got {len(screenshots)}"
            )

        # Select algorithm
        if config.algorithm == DiscoveryAlgorithm.PIXEL_STABILITY:
            result = self._discover_with_pixel_stability(screenshots, config, region)
        elif config.algorithm == DiscoveryAlgorithm.DIFFERENTIAL_CONSISTENCY:
            # For differential, we need transition pairs - create from consecutive frames
            transitions = self._create_transitions_from_sequence(screenshots)
            result = self._discover_with_differential(transitions, config)
        else:  # COMBINED
            result = self._discover_combined(screenshots, config, region)

        return DiscoveryResult.from_analysis_result(result, config)

    def discover_states_with_transitions(
        self,
        screenshots: list[np.ndarray[Any, Any]],
        transitions: list[TransitionInfo],
        config: DiscoveryConfig | None = None,
    ) -> DiscoveryResult:
        """Discover states using transition data for enhanced boundary detection.

        When you have explicit before/after pairs (e.g., from user interactions),
        this method provides more accurate state boundary detection than analyzing
        screenshots alone.

        Args:
            screenshots: Screenshots showing the states
            transitions: List of TransitionInfo with before/after pairs
            config: Configuration overrides

        Returns:
            DiscoveryResult with states discovered using transition analysis

        Example:
            >>> # Capture transitions during recording
            >>> transitions = [
            ...     TransitionInfo(before_img, after_img, click_point=(100, 200))
            ...     for before_img, after_img in recorded_transitions
            ... ]
            >>> result = facade.discover_states_with_transitions(screenshots, transitions)
        """
        config = config or self._default_config

        if not screenshots:
            raise ValueError("Screenshots list cannot be empty")

        # Use differential consistency for transition-based discovery
        diff_result = self._discover_with_differential(transitions, config)

        # Optionally enhance with pixel stability
        if config.algorithm == DiscoveryAlgorithm.COMBINED:
            pixel_result = self._discover_with_pixel_stability(
                screenshots, config, None
            )
            merged = self._merge_results(pixel_result, diff_result)
            return DiscoveryResult.from_analysis_result(merged, config)

        return DiscoveryResult.from_analysis_result(diff_result, config)

    def analyze_region(
        self,
        screenshots: list[np.ndarray[Any, Any]],
        region: tuple[int, int, int, int],
        config: DiscoveryConfig | None = None,
    ) -> DiscoveryResult:
        """Analyze a specific region of the screen for state elements.

        Useful when you want to focus discovery on a particular area,
        such as a modal dialog or a specific UI panel.

        Args:
            screenshots: Screenshot sequence
            region: Region to analyze as (x, y, x2, y2)
            config: Configuration overrides

        Returns:
            DiscoveryResult for the specified region

        Example:
            >>> # Focus on a dialog area
            >>> dialog_region = (100, 100, 500, 400)
            >>> result = facade.analyze_region(screenshots, dialog_region)
        """
        return self.discover_states(screenshots, config, region)

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _discover_with_pixel_stability(
        self,
        screenshots: list[np.ndarray[Any, Any]],
        config: DiscoveryConfig,
        region: tuple[int, int, int, int] | None,
    ) -> AnalysisResult:
        """Run pixel stability analysis."""
        # Create analyzer with current config
        from .pixel_stability_matrix_analyzer import PixelStabilityMatrixAnalyzer

        analyzer = PixelStabilityMatrixAnalyzer(config.to_analysis_config())
        return analyzer.analyze_screenshots(screenshots, region)

    def _discover_with_differential(
        self,
        transitions: list[TransitionInfo],
        config: DiscoveryConfig,
    ) -> AnalysisResult:
        """Run differential consistency analysis on transitions."""
        if not transitions:
            return AnalysisResult(states=[], state_images=[], transitions=[])

        # Convert transitions to pairs format
        pairs = [(t.before_screenshot, t.after_screenshot) for t in transitions]

        # Detect state regions
        regions = self.diff_detector.detect_state_regions(
            pairs,
            consistency_threshold=config.consistency_threshold,
            min_region_area=config.min_region_size[0] * config.min_region_size[1],
        )

        # Convert regions to StateImages
        state_images = []
        for i, region in enumerate(regions):
            x, y, w, h = region.bbox
            state_image = StateImage(
                id=f"diff_img_{i}",
                name=f"Region_{i}",
                x=x,
                y=y,
                x2=x + w,
                y2=y + h,
                pixel_hash=f"diff_{i}",
                frequency=region.consistency_score,
                pixel_data=region.example_diff,
            )
            state_images.append(state_image)

        # Group into states (simple: each unique region combo is a state)
        states = []
        if state_images and config.enable_state_grouping:
            state = DiscoveredState(
                id="diff_state_0",
                name="Discovered State",
                state_image_ids=[si.id for si in state_images],
                screenshot_ids=[],
                confidence=(
                    sum(si.frequency for si in state_images) / len(state_images)
                    if state_images
                    else 0
                ),
            )
            states.append(state)

        return AnalysisResult(
            states=states,
            state_images=state_images,
            transitions=[],
            statistics={
                "algorithm": "differential_consistency",
                "transitions_analyzed": len(transitions),
                "regions_found": len(regions),
            },
        )

    def _discover_combined(
        self,
        screenshots: list[np.ndarray[Any, Any]],
        config: DiscoveryConfig,
        region: tuple[int, int, int, int] | None,
    ) -> AnalysisResult:
        """Run combined analysis using both algorithms."""
        # Run pixel stability
        pixel_result = self._discover_with_pixel_stability(screenshots, config, region)

        # Create transitions from consecutive frames and run differential
        transitions = self._create_transitions_from_sequence(screenshots)
        if transitions:
            diff_result = self._discover_with_differential(transitions, config)
            return self._merge_results(pixel_result, diff_result)

        return pixel_result

    def _create_transitions_from_sequence(
        self,
        screenshots: list[np.ndarray[Any, Any]],
    ) -> list[TransitionInfo]:
        """Create transition pairs from consecutive screenshots."""
        from .state_construction.state_builder import TransitionInfo

        transitions = []
        for i in range(len(screenshots) - 1):
            transitions.append(
                TransitionInfo(
                    before_screenshot=screenshots[i],
                    after_screenshot=screenshots[i + 1],
                )
            )
        return transitions

    def _merge_results(
        self,
        result1: AnalysisResult,
        result2: AnalysisResult,
    ) -> AnalysisResult:
        """Merge results from multiple algorithms."""
        # Combine state images (dedupe by overlap)
        all_images = list(result1.state_images)
        for img2 in result2.state_images:
            # Check if this image significantly overlaps with existing
            is_duplicate = False
            for img1 in all_images:
                if self._images_overlap(img1, img2, threshold=0.7):
                    is_duplicate = True
                    break
            if not is_duplicate:
                all_images.append(img2)

        # Combine states
        all_states = list(result1.states) + list(result2.states)

        # Merge statistics
        merged_stats = {
            **result1.statistics,
            **{f"diff_{k}": v for k, v in result2.statistics.items()},
            "merged": True,
        }

        return AnalysisResult(
            states=all_states,
            state_images=all_images,
            transitions=result1.transitions + result2.transitions,
            statistics=merged_stats,
        )

    def _images_overlap(
        self,
        img1: StateImage,
        img2: StateImage,
        threshold: float = 0.5,
    ) -> bool:
        """Check if two state images overlap significantly."""
        # Calculate intersection
        x1 = max(img1.x, img2.x)
        y1 = max(img1.y, img2.y)
        x2 = min(img1.x2, img2.x2)
        y2 = min(img1.y2, img2.y2)

        if x1 >= x2 or y1 >= y2:
            return False

        intersection = (x2 - x1) * (y2 - y1)
        area1 = img1.width * img1.height
        area2 = img2.width * img2.height
        min_area = min(area1, area2)

        return intersection / min_area >= threshold if min_area > 0 else False


# Convenience function for simple usage
def discover_states(
    screenshots: list[np.ndarray[Any, Any]],
    algorithm: DiscoveryAlgorithm = DiscoveryAlgorithm.PIXEL_STABILITY,
    **config_kwargs: Any,
) -> DiscoveryResult:
    """Convenience function for quick state discovery.

    Args:
        screenshots: Screenshot sequence to analyze
        algorithm: Which algorithm to use
        **config_kwargs: Additional configuration options

    Returns:
        DiscoveryResult with discovered states

    Example:
        >>> from qontinui.discovery import discover_states
        >>> result = discover_states(screenshots, stability_threshold=0.95)
    """
    config = DiscoveryConfig(algorithm=algorithm, **config_kwargs)
    facade = StateDiscoveryFacade(config)
    return facade.discover_states(screenshots)
