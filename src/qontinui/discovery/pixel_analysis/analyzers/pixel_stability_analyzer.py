"""Pipeline coordinator for pixel stability analysis."""

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np

from ...models import AnalysisConfig, AnalysisResult
from .cooccurrence_analyzer import CooccurrenceAnalyzer
from .stability_map_creator import StabilityMapCreator
from .stable_region_extractor import StableRegionExtractor
from .state_image_factory import StateImageFactory
from .transition_detector import TransitionDetector

logger = logging.getLogger(__name__)


class PixelStabilityAnalyzer:
    """Coordinates the pixel stability analysis pipeline."""

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        """Initialize analyzer with configuration."""
        self.config = config or AnalysisConfig()
        self.progress_callback: Callable[..., Any] | None = None
        self._current_progress = 0

        # Initialize analysis components
        self.stability_map_creator = StabilityMapCreator(self.config)
        self.region_extractor = StableRegionExtractor(self.config)
        self.state_image_factory = StateImageFactory(self.config)
        self.cooccurrence_analyzer = CooccurrenceAnalyzer()
        self.transition_detector = TransitionDetector()

    def analyze_screenshots(
        self,
        screenshots: list[np.ndarray[Any, Any]],
        progress_callback: Callable[..., Any] | None = None,
    ) -> AnalysisResult:
        """
        Analyze screenshots to discover states and StateImages.

        Args:
            screenshots: List of screenshot arrays
            progress_callback: Optional callback for progress updates

        Returns:
            AnalysisResult containing discovered states and StateImages
        """
        self.progress_callback = progress_callback

        if len(screenshots) < 2:
            raise ValueError("At least 2 screenshots required for analysis")

        # Ensure all screenshots have same dimensions
        self._validate_dimensions(screenshots)

        # Step 1: Create stability map (30% progress)
        self._update_progress(0, "Creating pixel stability map...")
        stability_map = self.stability_map_creator.create(screenshots)
        self._update_progress(30, "Stability map created")

        # Step 2: Extract stable regions (60% progress)
        self._update_progress(30, "Extracting stable regions...")
        stable_regions = self.region_extractor.extract(stability_map, screenshots[0])
        self._update_progress(60, f"Found {len(stable_regions)} stable regions")

        # Step 3: Create StateImages from regions (80% progress)
        self._update_progress(60, "Creating StateImages...")
        state_images = self.state_image_factory.create(stable_regions, screenshots)
        self._update_progress(80, f"Created {len(state_images)} StateImages")

        # Step 4: Group into states (90% progress)
        self._update_progress(80, "Grouping into states...")
        states = []
        if self.config.enable_cooccurrence_analysis:
            states = self.cooccurrence_analyzer.analyze(state_images, screenshots)
        self._update_progress(90, f"Discovered {len(states)} states")

        # Step 5: Find transitions (100% progress)
        self._update_progress(90, "Analyzing transitions...")
        transitions = self.transition_detector.detect(states, screenshots)
        self._update_progress(100, "Analysis complete")

        # Calculate statistics
        statistics = self._calculate_statistics(screenshots, state_images, states, stability_map)

        return AnalysisResult(
            states=states,
            state_images=state_images,
            transitions=transitions,
            stability_map=stability_map,
            statistics=statistics,
        )

    def _validate_dimensions(self, screenshots: list[np.ndarray[Any, Any]]):
        """Ensure all screenshots have the same dimensions."""
        if not screenshots:
            return

        ref_shape = screenshots[0].shape
        for i, img in enumerate(screenshots[1:], 1):
            if img.shape != ref_shape:
                raise ValueError(
                    f"Screenshot {i} has different dimensions: {img.shape} vs {ref_shape}"
                )

    def _calculate_statistics(
        self,
        screenshots: list[np.ndarray[Any, Any]],
        state_images: list[Any],
        states: list[Any],
        stability_map: np.ndarray[Any, Any],
    ) -> dict[str, Any]:
        """Calculate analysis statistics."""
        total_pixels = stability_map.size
        stable_pixels = np.sum(stability_map)

        return {
            "total_screenshots": len(screenshots),
            "states_found": len(states),
            "state_images_found": len(state_images),
            "average_state_images_per_state": (len(state_images) / len(states) if states else 0),
            "pixel_stability_score": (stable_pixels / total_pixels if total_pixels else 0),
            "stable_pixel_count": int(stable_pixels),
            "total_pixel_count": int(total_pixels),
        }

    def _update_progress(self, percentage: int, message: str):
        """Update progress via callback if provided."""
        self._current_progress = percentage
        if self.progress_callback:
            self.progress_callback(
                {
                    "percentage": percentage,
                    "message": message,
                    "timestamp": datetime.now().isoformat(),
                }
            )
