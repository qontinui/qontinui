"""
Pixel Stability Analyzer for State Discovery
Analyzes screenshots to find stable regions and calculate pixel statistics
"""

import hashlib
import logging
from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from .models import AnalysisConfig, AnalysisResult, DiscoveredState, StateImage

logger = logging.getLogger(__name__)


class PixelStabilityAnalyzer:
    """Analyzes pixel stability across screenshots to identify UI elements"""

    def __init__(self, config: AnalysisConfig):
        self.config = config

    def analyze_screenshots(self, screenshots: list[np.ndarray]) -> AnalysisResult:
        """
        Main analysis entry point

        Args:
            screenshots: List of screenshot images as numpy arrays

        Returns:
            AnalysisResult with discovered states and state images
        """
        logger.info(f"Starting analysis of {len(screenshots)} screenshots")

        # Find stable regions
        stable_regions = self._find_stable_regions(screenshots)

        # Convert regions to StateImage objects with pixel analysis
        state_images = self._create_state_images(stable_regions, screenshots)

        # Group into states
        states = self._discover_states(state_images, screenshots)

        # Calculate statistics
        statistics = self._calculate_statistics(screenshots, state_images, states)

        return AnalysisResult(
            states=states,
            state_images=state_images,
            transitions=[],  # TODO: Implement transition detection
            statistics=statistics,
        )

    def _find_stable_regions(self, screenshots: list[np.ndarray]) -> list[dict[str, Any]]:
        """Find regions that are stable across screenshots"""
        if len(screenshots) < 2:
            return []

        height, width = screenshots[0].shape[:2]
        logger.info(f"Analyzing {len(screenshots)} screenshots of size {width}x{height}")
        stable_regions = []

        # Simple grid-based approach for demo
        # In production, use more sophisticated region detection
        grid_size = 50
        step_size = max(grid_size // 2, 25)  # Use at least 25 pixel steps

        # Ensure we have valid ranges
        y_max = max(1, height - grid_size + 1)
        x_max = max(1, width - grid_size + 1)

        # Calculate total regions to process
        y_steps = len(range(0, y_max, step_size))
        x_steps = len(range(0, x_max, step_size))
        total_regions = y_steps * x_steps
        logger.info(f"Will process {total_regions} regions ({y_steps}x{x_steps} grid)")

        regions_processed = 0
        for y in range(0, y_max, step_size):
            for x in range(0, x_max, step_size):
                # Ensure we don't exceed image bounds
                y_end = min(y + grid_size, height)
                x_end = min(x + grid_size, width)

                region_hash = None
                present_in = []
                reference_region = None

                for idx, screenshot in enumerate(screenshots):
                    region = screenshot[y:y_end, x:x_end]
                    current_hash = hashlib.md5(region.tobytes()).hexdigest()

                    if region_hash is None:
                        region_hash = current_hash
                        reference_region = region.copy()
                        present_in.append(f"screenshot_{idx:03d}")
                    elif region_hash == current_hash:
                        # Exact match
                        present_in.append(f"screenshot_{idx:03d}")
                    elif reference_region is not None and self._regions_similar(
                        region, reference_region
                    ):
                        # Similar enough
                        present_in.append(f"screenshot_{idx:03d}")

                if len(present_in) >= self.config.min_screenshots_present:
                    stable_regions.append(
                        {
                            "x": x,
                            "y": y,
                            "x2": x_end,
                            "y2": y_end,
                            "hash": region_hash,
                            "screenshots": present_in,
                        }
                    )

                regions_processed += 1
                if regions_processed % 100 == 0:
                    logger.info(
                        f"Processed {regions_processed}/{total_regions} regions, found {len(stable_regions)} stable"
                    )

        logger.info(
            f"Found {len(stable_regions)} stable regions out of {regions_processed} processed"
        )
        return stable_regions

    def _regions_similar(self, region1: np.ndarray, region2: np.ndarray) -> bool:
        """Check if two regions are similar enough"""
        if region1.shape != region2.shape:
            return False

        # Simple pixel difference check
        diff = cv2.absdiff(region1, region2)
        mean_diff = np.mean(diff)

        return mean_diff < self.config.color_tolerance

    def _calculate_pixel_percentages(self, region: np.ndarray) -> tuple[float, float, float]:
        """
        Calculate dark and light pixel percentages for a region

        Returns:
            Tuple of (dark_percentage, light_percentage, avg_brightness)
        """
        # Convert to grayscale if needed
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region

        # Calculate brightness statistics
        total_pixels = gray.size
        dark_pixels = np.sum(gray < self.config.dark_pixel_threshold)
        light_pixels = np.sum(gray > self.config.light_pixel_threshold)
        avg_brightness = np.mean(gray)

        dark_percentage = (dark_pixels / total_pixels) * 100
        light_percentage = (light_pixels / total_pixels) * 100

        return dark_percentage, light_percentage, float(avg_brightness)

    def _create_state_images(
        self, stable_regions: list[dict[str, Any]], screenshots: list[np.ndarray]
    ) -> list[StateImage]:
        """Convert stable regions to StateImage objects with pixel analysis"""
        state_images = []

        for idx, region in enumerate(stable_regions):
            # Get the region from the first screenshot for pixel analysis
            x, y, x2, y2 = region["x"], region["y"], region["x2"], region["y2"]
            # Ensure coordinates are within bounds
            height, width = screenshots[0].shape[:2]
            x2 = min(x2, width)
            y2 = min(y2, height)
            region_pixels = screenshots[0][y:y2, x:x2]

            # Calculate pixel percentages
            dark_pct, light_pct, avg_brightness = self._calculate_pixel_percentages(region_pixels)

            state_image = StateImage(
                id=f"si_{idx:04d}",
                name=f"StateImage_{idx:04d}",
                x=x,
                y=y,
                x2=x2,
                y2=y2,
                width=x2 - x,
                height=y2 - y,
                pixel_hash=region["hash"],
                stability_score=1.0,  # Perfect match when present
                screenshots=region["screenshots"],
                dark_pixel_percentage=dark_pct,
                light_pixel_percentage=light_pct,
                avg_brightness=avg_brightness,
            )
            state_images.append(state_image)

        logger.info(f"Created {len(state_images)} state images with pixel analysis")
        return state_images

    def _discover_states(
        self, state_images: list[StateImage], screenshots: list[np.ndarray]
    ) -> list[DiscoveredState]:
        """Group state images into states based on co-occurrence"""
        states = []

        # Simple grouping by screenshot presence
        # In production, use more sophisticated clustering
        screenshot_groups = defaultdict(list)

        for si in state_images:
            key = tuple(sorted(si.screenshots))
            screenshot_groups[key].append(si)

        for idx, (screenshot_ids, group) in enumerate(screenshot_groups.items()):
            if len(group) >= 1:  # At least 1 state image to form a state
                state = DiscoveredState(
                    id=f"state_{idx:03d}",
                    name=f"State_{idx:03d}",
                    state_image_ids=[si.id for si in group],
                    screenshot_ids=list(screenshot_ids),
                    confidence=0.9,  # Simplified confidence
                )
                states.append(state)

        logger.info(f"Discovered {len(states)} states")
        return states

    def _calculate_statistics(
        self,
        screenshots: list[np.ndarray],
        state_images: list[StateImage],
        states: list[DiscoveredState],
    ) -> dict[str, Any]:
        """Calculate analysis statistics"""
        total_dark_images = sum(1 for si in state_images if si.dark_pixel_percentage > 50)
        total_light_images = sum(1 for si in state_images if si.light_pixel_percentage > 50)

        return {
            "total_screenshots": len(screenshots),
            "total_state_images": len(state_images),
            "total_states": len(states),
            "avg_state_images_per_state": len(state_images) / len(states) if states else 0,
            "pixel_stability_score": 0.85,  # Simplified score
            "dark_dominant_images": total_dark_images,
            "light_dominant_images": total_light_images,
            "avg_brightness": (
                np.mean([si.avg_brightness for si in state_images]) if state_images else 0
            ),
        }
