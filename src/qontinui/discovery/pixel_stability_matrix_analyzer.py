"""
Pixel Stability Matrix Analyzer for Non-Rectangular Shape Discovery

This analyzer uses a pixel-level stability matrix to discover UI elements
of any shape, not limited to rectangles.
"""

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, cast

import cv2
import numpy as np

from .models import AnalysisConfig, AnalysisResult, DiscoveredState, StateImage

# scipy not needed - using cv2 for connected components


logger = logging.getLogger(__name__)


@dataclass
class PixelPattern:
    """Represents a pixel's presence pattern across screenshots"""

    screenshots: set[int] = field(default_factory=set)
    reference_value: np.ndarray[Any, Any] | None = None
    pixel_coords: list[tuple[int, int]] = field(default_factory=list)


class PixelStabilityMatrixAnalyzer:
    """Discovers non-rectangular UI elements using pixel stability analysis"""

    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config
        self.min_component_size = config.min_region_size[0] * config.min_region_size[1]
        self.max_component_size = config.max_region_size[0] * config.max_region_size[1]

    def analyze_screenshots(
        self,
        screenshots: list[np.ndarray[Any, Any]],
        region: tuple[int, int, int, int] | None = None,
    ) -> AnalysisResult:
        """
        Main analysis method using Pixel Stability Matrix approach

        Args:
            screenshots: List of screenshot images (numpy arrays)
            region: Optional (x, y, x2, y2) to limit analysis to specific region

        Returns:
            AnalysisResult with discovered StateImages and States
        """
        if not screenshots or len(screenshots) < self.config.min_screenshots_present:
            logger.warning(f"Need at least {self.config.min_screenshots_present} screenshots")
            return AnalysisResult(states=[], state_images=[], transitions=[])

        logger.info(f"Starting Pixel Stability Matrix analysis with {len(screenshots)} screenshots")

        # Step 1: Build stability matrix
        stability_matrix = self._build_stability_matrix(screenshots, region)

        # Step 2: Group pixels by presence pattern
        pattern_groups = self._group_by_pattern(stability_matrix)

        # Step 3: Find connected components for each pattern
        state_images: list[StateImage] = []
        for pattern_key, pixel_coords in pattern_groups.items():
            if len(pixel_coords) < self.min_component_size:
                continue

            h, w = screenshots[0].shape[:2]
            components = self._find_connected_components(pixel_coords, (h, w))

            for component in components:
                if self.min_component_size <= len(component) <= self.max_component_size:
                    state_image = self._create_state_image(
                        component, pattern_key, screenshots, len(state_images)
                    )
                    if state_image:
                        state_images.append(state_image)

        # Step 4: Group StateImages into States
        states = self._discover_states(state_images, screenshots)

        logger.info(f"Discovered {len(state_images)} StateImages and {len(states)} States")

        return AnalysisResult(
            states=states,
            state_images=state_images,
            transitions=[],
            statistics={
                "total_screenshots": len(screenshots),
                "patterns_found": len(pattern_groups),
                "state_images": len(state_images),
                "states": len(states),
            },
        )

    def _build_stability_matrix(
        self,
        screenshots: list[np.ndarray[Any, Any]],
        region: tuple[int, int, int, int] | None = None,
    ) -> dict[tuple[int, int], tuple[set[int], np.ndarray[Any, Any]]]:
        """
        Build a matrix tracking which screenshots each pixel appears in

        Returns:
            Dict mapping (x,y) to (set of screenshot indices, reference pixel value)
        """
        height, width = screenshots[0].shape[:2]

        # Define region bounds
        if region:
            x_start, y_start, x_end, y_end = region
            x_end = min(x_end, width)
            y_end = min(y_end, height)
        else:
            x_start, y_start, x_end, y_end = 0, 0, width, height

        stability_matrix = {}
        total_pixels = (x_end - x_start) * (y_end - y_start)
        pixels_processed = 0

        logger.info(f"Building stability matrix for {total_pixels} pixels")

        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                # Find which screenshots have stable pixel at this location
                presence_set = set()
                reference_value = None

                for idx, screenshot in enumerate(screenshots):
                    pixel_value = screenshot[y, x]

                    if reference_value is None:
                        # First screenshot - set as reference
                        reference_value = pixel_value.copy()
                        presence_set.add(idx)
                    else:
                        # Compare with reference
                        if self._pixels_match(pixel_value, reference_value):
                            presence_set.add(idx)

                # Only store if pixel appears in minimum screenshots
                if (
                    len(presence_set) >= self.config.min_screenshots_present
                    and reference_value is not None
                ):
                    stability_matrix[(x, y)] = (presence_set, reference_value)

                pixels_processed += 1
                if pixels_processed % 10000 == 0:
                    logger.debug(f"Processed {pixels_processed}/{total_pixels} pixels")

        logger.info(f"Stability matrix built: {len(stability_matrix)} stable pixels found")
        return stability_matrix

    def _pixels_match(self, pixel1: np.ndarray[Any, Any], pixel2: np.ndarray[Any, Any]) -> bool:
        """Check if two pixels match within tolerance"""
        if pixel1.shape != pixel2.shape:
            return False

        # Calculate color difference
        diff = np.abs(pixel1.astype(float) - pixel2.astype(float))
        mean_diff = np.mean(diff)

        return cast(bool, mean_diff <= self.config.color_tolerance)

    def _group_by_pattern(
        self, stability_matrix: dict[tuple[int, int], tuple[set[int], np.ndarray[Any, Any]]]
    ) -> dict[str, list[tuple[int, int]]]:
        """
        Group pixels by their presence pattern (which screenshots they appear in)

        Returns:
            Dict mapping pattern string to list of pixel coordinates
        """
        pattern_groups = defaultdict(list)

        for coords, (presence_set, _) in stability_matrix.items():
            # Create a string key from the presence pattern
            pattern_key = ",".join(map(str, sorted(presence_set)))
            pattern_groups[pattern_key].append(coords)

        logger.info(f"Found {len(pattern_groups)} unique presence patterns")
        for pattern, pixels in pattern_groups.items():
            screenshots = pattern.split(",")
            logger.debug(f"Pattern {pattern}: {len(pixels)} pixels in screenshots {screenshots}")

        return pattern_groups

    def _find_connected_components(
        self, pixel_coords: list[tuple[int, int]], image_shape: tuple[int, int]
    ) -> list[list[tuple[int, int]]]:
        """
        Find connected components from a list of pixel coordinates

        Returns:
            List of connected components, each as a list of pixel coordinates
        """
        if not pixel_coords:
            return []

        height, width = image_shape

        # Create binary mask
        mask = np.zeros((height, width), dtype=np.uint8)
        for x, y in pixel_coords:
            mask[y, x] = 255

        # Find connected components
        num_labels, labels = cv2.connectedComponents(mask, connectivity=8)

        components = []
        for label in range(1, num_labels):  # Skip background (0)
            component_mask = labels == label
            component_pixels = []

            # Get pixel coordinates for this component
            y_coords, x_coords = np.where(component_mask)
            for y, x in zip(y_coords, x_coords, strict=False):
                component_pixels.append((x, y))

            if component_pixels:
                components.append(component_pixels)

        logger.debug(
            f"Found {len(components)} connected components from {len(pixel_coords)} pixels"
        )
        return components

    def _create_state_image(
        self,
        component: list[tuple[int, int]],
        pattern_key: str,
        screenshots: list[np.ndarray[Any, Any]],
        idx: int,
    ) -> StateImage | None:
        """
        Create a StateImage from a connected component

        Args:
            component: List of pixel coordinates in the component
            pattern_key: String representing which screenshots this appears in
            screenshots: Original screenshot images
            idx: Index for naming

        Returns:
            StateImage object or None if creation fails
        """
        if not component:
            return None

        # Get bounding box
        xs = [x for x, y in component]
        ys = [y for x, y in component]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Create mask for non-rectangular shape
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        mask = np.zeros((height, width), dtype=np.float32)

        for x, y in component:
            mask[y - y_min, x - x_min] = 1.0

        # Get screenshot indices
        screenshot_indices = [int(i) for i in pattern_key.split(",")]
        screenshot_ids = [f"screenshot_{i:03d}" for i in screenshot_indices]

        # Calculate pixel hash from first screenshot where present
        first_screenshot = screenshots[screenshot_indices[0]]
        region = first_screenshot[y_min : y_max + 1, x_min : x_max + 1]

        # Apply mask to region for hash calculation
        masked_region = region.copy()
        mask_3channel = np.stack([mask] * 3, axis=2) if len(region.shape) == 3 else mask
        masked_region = (masked_region * mask_3channel).astype(np.uint8)

        pixel_hash = hashlib.md5(masked_region.tobytes()).hexdigest()

        # Calculate pixel statistics
        dark_pct, light_pct, avg_brightness = self._calculate_pixel_percentages(masked_region, mask)

        # Calculate mask density (how much of the bounding box is filled)
        mask_density = float(np.sum(mask) / (width * height))

        state_image = StateImage(
            id=f"si_{idx:04d}",
            name=f"StateImage_{idx:04d}",
            x=x_min,
            y=y_min,
            x2=x_max,
            y2=y_max,
            pixel_hash=pixel_hash,
            frequency=len(screenshot_ids)
            / len(screenshots),  # Fraction of screenshots where present
            screenshot_ids=screenshot_ids,
            mask=mask,  # Include the actual mask
            dark_pixel_percentage=dark_pct,
            light_pixel_percentage=light_pct,
            mask_density=mask_density,
        )

        # Add tag if it's a non-rectangular shape
        if mask_density < 0.95:  # Not a full rectangle
            state_image.tags.append("non_rectangular")
            state_image.tags.append(f"mask_density_{mask_density:.2f}")

        logger.debug(
            f"Created StateImage {state_image.id}: {width}x{height}, "
            f"mask_density={mask_density:.2f}, in {len(screenshot_ids)} screenshots"
        )

        return state_image

    def _calculate_pixel_percentages(
        self, region: np.ndarray[Any, Any], mask: np.ndarray[Any, Any] | None = None
    ) -> tuple[float, float, float]:
        """
        Calculate dark and light pixel percentages for a region

        Args:
            region: Image region
            mask: Optional mask for non-rectangular regions

        Returns:
            Tuple of (dark_percentage, light_percentage, avg_brightness)
        """
        # Convert to grayscale if needed
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region.copy()

        # Apply mask if provided
        if mask is not None:
            # Only consider pixels where mask > 0
            masked_pixels = gray[mask > 0]
        else:
            masked_pixels = gray.flatten()

        if len(masked_pixels) == 0:
            return 0.0, 0.0, 0.0

        # Calculate percentages
        dark_threshold = 50  # Pixels below this are "dark"
        light_threshold = 200  # Pixels above this are "light"

        dark_count = np.sum(masked_pixels < dark_threshold)
        light_count = np.sum(masked_pixels > light_threshold)
        total_pixels = len(masked_pixels)

        dark_pct = float((dark_count / total_pixels) * 100)
        light_pct = float((light_count / total_pixels) * 100)
        avg_brightness = float(np.mean(masked_pixels))

        return dark_pct, light_pct, avg_brightness

    def _discover_states(
        self, state_images: list[StateImage], screenshots: list[np.ndarray[Any, Any]]
    ) -> list[DiscoveredState]:
        """
        Group StateImages into States based on co-occurrence patterns

        Returns:
            List of DiscoveredState objects
        """
        states = []

        # Group StateImages by their screenshot presence
        screenshot_groups = defaultdict(list)

        for si in state_images:
            # Use screenshot list as grouping key
            key = tuple(sorted(si.screenshot_ids))
            screenshot_groups[key].append(si)

        # Create states from groups
        for idx, (screenshot_ids, group) in enumerate(screenshot_groups.items()):
            if len(group) >= 1:  # At least 1 StateImage to form a state
                state = DiscoveredState(
                    id=f"state_{idx:03d}",
                    name=f"State_{idx:03d}",
                    state_image_ids=[si.id for si in group],
                    screenshot_ids=list(screenshot_ids),
                    confidence=0.95,  # High confidence for pixel-perfect matches
                )
                states.append(state)

                logger.debug(
                    f"Created State {state.id} with {len(group)} StateImages "
                    f"in screenshots {screenshot_ids[:3]}..."
                )

        return states
