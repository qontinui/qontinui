"""Pixel stability analysis for discovering StateImages."""

import hashlib
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

import cv2
import numpy as np

from ..models import (
    AnalysisConfig,
    AnalysisResult,
    DiscoveredState,
    StateImage,
    StateTransition,
)
from .merge_components import merge_nearby_components

logger = logging.getLogger(__name__)


class PixelStabilityAnalyzer:
    """Analyzes pixel stability across screenshots to discover StateImages."""

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        """Initialize analyzer with configuration."""
        self.config = config or AnalysisConfig()
        self.progress_callback: Callable[..., Any] | None = None
        self._current_progress = 0

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
        stability_map = self.create_stability_map(screenshots)
        self._update_progress(30, "Stability map created")

        # Step 2: Extract stable regions (60% progress)
        self._update_progress(30, "Extracting stable regions...")
        stable_regions = self.extract_stable_regions(stability_map, screenshots[0])
        self._update_progress(60, f"Found {len(stable_regions)} stable regions")

        # Step 3: Create StateImages from regions (80% progress)
        self._update_progress(60, "Creating StateImages...")
        state_images = self.create_state_images(stable_regions, screenshots)
        self._update_progress(80, f"Created {len(state_images)} StateImages")

        # Step 4: Group into states (90% progress)
        self._update_progress(80, "Grouping into states...")
        states = []
        if self.config.enable_cooccurrence_analysis:
            states = self.group_by_cooccurrence(state_images, screenshots)
        self._update_progress(90, f"Discovered {len(states)} states")

        # Step 5: Find transitions (100% progress)
        self._update_progress(90, "Analyzing transitions...")
        transitions = self.find_transitions(states, screenshots)
        self._update_progress(100, "Analysis complete")

        # Calculate statistics
        statistics = self._calculate_statistics(
            screenshots, state_images, states, stability_map
        )

        return AnalysisResult(
            states=states,
            state_images=state_images,
            transitions=transitions,
            stability_map=stability_map,
            statistics=statistics,
        )

    def create_stability_map(
        self, screenshots: list[np.ndarray[Any, Any]]
    ) -> np.ndarray[Any, Any]:
        """
        Create a map showing pixel stability across screenshots.

        Args:
            screenshots: List of screenshot arrays

        Returns:
            Binary stability map where 1 indicates stable pixels
        """
        if not screenshots:
            return np.array([])

        height, width = screenshots[0].shape[:2]

        # Stack screenshots for variance calculation
        stack = np.stack(screenshots, axis=0)

        # Calculate pixel-wise variance across screenshots
        pixel_variance = np.var(stack, axis=0)

        # For RGB images, check if all channels are stable
        if len(pixel_variance.shape) == 3:
            # All channels must be below threshold
            stability_map = np.all(
                pixel_variance < self.config.variance_threshold, axis=2
            ).astype(np.uint8)
        else:
            stability_map = (pixel_variance < self.config.variance_threshold).astype(
                np.uint8
            )

        return cast(np.ndarray[Any, Any], stability_map)

    def extract_stable_regions(
        self, stability_map: np.ndarray[Any, Any], reference_image: np.ndarray[Any, Any]
    ) -> list[dict[str, Any]]:
        """
        Extract stable regions from stability map using component merging.

        Args:
            stability_map: Binary stability map
            reference_image: Reference screenshot for pixel data

        Returns:
            List of stable regions with coordinates and pixel data
        """
        # Use component merging to reduce region count
        use_merging = getattr(self.config, "use_component_merging", True)
        merge_gap = getattr(self.config, "merge_gap", 8)
        min_pixels = getattr(self.config, "min_component_pixels", 50)

        if use_merging:
            logger.info("Using component merging to extract regions")

            # Get merged regions
            merged_regions = merge_nearby_components(
                stability_map,
                max_gap=merge_gap,
                min_pixels=min_pixels,
                min_region_size=self.config.min_region_size,
                max_region_size=self.config.max_region_size,
            )

            # Convert merged regions to standard format
            regions = []
            for merged in merged_regions:
                # Extract pixel data from reference image
                x, y, x2, y2 = merged["x"], merged["y"], merged["x2"], merged["y2"]
                pixel_data = reference_image[y : y2 + 1, x : x2 + 1].copy()

                # Calculate pixel hash
                pixel_hash = self._calculate_pixel_hash(pixel_data)

                regions.append(
                    {
                        "x": x,
                        "y": y,
                        "x2": x2,
                        "y2": y2,
                        "width": merged["width"],
                        "height": merged["height"],
                        "pixel_data": pixel_data,
                        "pixel_hash": pixel_hash,
                        "mask": merged["mask"],
                    }
                )

            logger.info(f"Extracted {len(regions)} merged regions")

        else:
            # Original method - find all connected components
            logger.info("Using original connected components method")
            regions = []

            # Find connected components
            num_labels, labels = cv2.connectedComponents(stability_map)
            logger.info(f"Found {num_labels - 1} connected components")

            for label_id in range(1, min(num_labels, 1000)):  # Limit to 1000 for safety
                # Create mask for this component
                mask = (labels == label_id).astype(np.uint8)

                # Find bounding box
                coords = np.column_stack(np.where(mask))
                if len(coords) == 0:
                    continue

                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)

                width = x_max - x_min + 1
                height = y_max - y_min + 1

                # Check size constraints
                if (
                    width < self.config.min_region_size[0]
                    or height < self.config.min_region_size[1]
                    or width > self.config.max_region_size[0]
                    or height > self.config.max_region_size[1]
                ):
                    continue

                # Extract pixel data
                pixel_data = reference_image[
                    y_min : y_max + 1, x_min : x_max + 1
                ].copy()

                # Calculate pixel hash
                pixel_hash = self._calculate_pixel_hash(pixel_data)

                regions.append(
                    {
                        "x": int(x_min),
                        "y": int(y_min),
                        "x2": int(x_max),
                        "y2": int(y_max),
                        "width": width,
                        "height": height,
                        "pixel_data": pixel_data,
                        "pixel_hash": pixel_hash,
                        "mask": mask[y_min : y_max + 1, x_min : x_max + 1],
                    }
                )

        # Optionally decompose complex regions
        if self.config.enable_rectangle_decomposition:
            regions = self._decompose_complex_regions(regions)

        return regions

    def create_state_images(
        self, regions: list[dict[str, Any]], screenshots: list[np.ndarray[Any, Any]]
    ) -> list[StateImage]:
        """
        Create StateImage objects from stable regions.

        Args:
            regions: List of stable regions
            screenshots: All screenshots for frequency calculation

        Returns:
            List of StateImage objects
        """
        state_images = []

        for i, region in enumerate(regions):
            # Check presence in each screenshot
            present_in = []
            for j, screenshot in enumerate(screenshots):
                if self._is_region_present(region, screenshot):
                    present_in.append(f"screenshot_{j:03d}")

            frequency = len(present_in) / len(screenshots)

            # Skip if not present in enough screenshots
            if len(present_in) < self.config.min_screenshots_present:
                continue

            # Get or create mask efficiently
            mask = region.get("mask")
            if mask is not None:
                # Ensure mask is 2D and float32
                if len(mask.shape) > 2:
                    mask = mask[:, :, 0]
                mask = mask.astype(np.float32)
                mask_density = np.mean(mask)
            else:
                # Create full mask for backward compatibility
                h, w = region["pixel_data"].shape[:2]
                mask = np.ones((h, w), dtype=np.float32)
                mask_density = 1.0

            # Calculate pixel percentages using the mask
            dark_percentage, light_percentage = self._calculate_pixel_percentages(
                region["pixel_data"], mask
            )

            state_image = StateImage(
                id=f"si_{i:04d}_{region['pixel_hash'][:8]}",
                name=f"StateImage_{i:04d}",
                x=region["x"],
                y=region["y"],
                x2=region["x2"],
                y2=region["y2"],
                pixel_hash=region["pixel_hash"],
                frequency=frequency,
                screenshot_ids=present_in,
                pixel_data=region["pixel_data"],
                mask=mask,
                mask_density=mask_density,
                dark_pixel_percentage=dark_percentage,
                light_pixel_percentage=light_percentage,
            )

            state_images.append(state_image)

        return state_images

    def group_by_cooccurrence(
        self, state_images: list[StateImage], screenshots: list[np.ndarray[Any, Any]]
    ) -> list[DiscoveredState]:
        """
        Group StateImages into states based on co-occurrence patterns.

        Args:
            state_images: List of discovered StateImages
            screenshots: Original screenshots

        Returns:
            List of discovered states
        """
        states = []

        # Build co-occurrence matrix
        n = len(state_images)
        cooccurrence: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.zeros(
            (n, n)
        )

        for i in range(n):
            for j in range(i, n):
                # Count screenshots where both appear
                common = set(state_images[i].screenshot_ids) & set(
                    state_images[j].screenshot_ids
                )
                cooccurrence[i, j] = len(common)
                cooccurrence[j, i] = len(common)

        # Normalize by total screenshots
        cooccurrence = (
            (cooccurrence / len(screenshots)).astype(np.float64).reshape((n, n))
        )

        # Group StateImages that ALWAYS appear together (when one appears, all appear)
        grouped = []
        used = set()

        for i in range(n):
            if i in used:
                continue

            group = [i]
            used.add(i)
            group_screenshots = set(state_images[i].screenshot_ids)

            for j in range(i + 1, n):
                if j in used:
                    continue

                j_screenshots = set(state_images[j].screenshot_ids)

                # Check if j always appears with the group
                # This means: wherever j appears, all group members appear too
                # AND wherever any group member appears, j appears too
                # In other words: they appear in exactly the same screenshots
                if j_screenshots == group_screenshots:
                    group.append(j)
                    used.add(j)

            if group:
                grouped.append(group)

        # Create states from groups
        for i, group_indices in enumerate(grouped):
            state_image_ids = [state_images[idx].id for idx in group_indices]

            # Find screenshots where ALL state images appear together (intersection)
            screenshot_sets = [
                set(state_images[idx].screenshot_ids) for idx in group_indices
            ]
            # Since we grouped images that appear in exactly the same screenshots,
            # all sets should be identical, so intersection equals any individual set
            if screenshot_sets:
                common_screenshots = sorted(screenshot_sets[0])
            else:
                common_screenshots = []

            state = DiscoveredState(
                id=f"state_{i:03d}",
                name=f"State_{i:03d}",
                state_image_ids=state_image_ids,
                screenshot_ids=common_screenshots,
                confidence=0.9,
            )
            states.append(state)

        return states

    def find_transitions(
        self, states: list[DiscoveredState], screenshots: list[np.ndarray[Any, Any]]
    ) -> list[StateTransition]:
        """
        Find potential transitions between states.

        Args:
            states: List of discovered states
            screenshots: Original screenshots

        Returns:
            List of state transitions
        """
        transitions: list[StateTransition] = []

        # FUTURE ENHANCEMENT: Implement transition detection based on screenshot sequence.
        # This would analyze temporal patterns in the screenshot sequence to identify
        # state transitions (e.g., when State A disappears and State B appears).
        # Requires: Sequential analysis of state appearances/disappearances over time.

        return transitions

    def _validate_dimensions(self, screenshots: list[np.ndarray[Any, Any]]):
        """Ensure all screenshots have the same dimensions."""
        if not screenshots:
            return

        ref_shape = screenshots[0].shape
        for i, img in enumerate(screenshots[1:], 1):
            if img.shape != ref_shape:
                raise ValueError(
                    f"Screenshot {i} has different dimensions: "
                    f"{img.shape} vs {ref_shape}"
                )

    def _calculate_pixel_hash(self, pixel_data: np.ndarray[Any, Any]) -> str:
        """Calculate hash of pixel data for comparison."""
        # Convert to bytes and hash
        data_bytes = pixel_data.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()

    def _calculate_pixel_percentages(
        self, pixel_data: np.ndarray[Any, Any], mask: np.ndarray[Any, Any] | None = None
    ) -> tuple[float, float]:
        """
        Calculate dark and light pixel percentages for a masked region.

        Args:
            pixel_data: RGB pixel data array
            mask: Optional mask array (0.0-1.0). If None, uses full rectangle.

        Returns:
            Tuple of (dark_percentage, light_percentage) for active mask pixels only
        """
        # Define thresholds
        dark_threshold = 60  # Pixels with brightness < 60 are considered dark
        light_threshold = 200  # Pixels with brightness > 200 are considered light

        # Convert to grayscale for brightness calculation
        if len(pixel_data.shape) == 3:
            # RGB image - calculate brightness as average
            brightness = np.mean(pixel_data, axis=2)
        else:
            # Already grayscale
            brightness = pixel_data

        # Apply mask if provided
        if mask is not None:
            # Ensure mask is 2D
            if len(mask.shape) > 2:
                mask = mask[:, :, 0]  # Take first channel if 3D

            # Quick shape check - masks should be created with correct dimensions
            if mask.shape != brightness.shape:
                # This shouldn't happen if masks are created properly
                # Fall back to no mask to avoid expensive resize
                logger.warning(
                    f"Mask shape {mask.shape} doesn't match brightness {brightness.shape}, skipping mask"
                )
                mask = None

        if mask is not None:
            # Use vectorized operations without flattening
            active_pixels = mask > 0.5
            total_pixels = np.count_nonzero(active_pixels)

            if total_pixels == 0:
                return 0.0, 0.0

            # Apply mask and count in one operation
            masked_brightness = brightness * active_pixels
            dark_pixels = np.count_nonzero(
                (masked_brightness < dark_threshold) & active_pixels
            )
            light_pixels = np.count_nonzero(
                (masked_brightness > light_threshold) & active_pixels
            )
        else:
            # No mask - use all pixels
            total_pixels = brightness.size
            if total_pixels == 0:
                return 0.0, 0.0

            dark_pixels = np.count_nonzero(brightness < dark_threshold)
            light_pixels = np.count_nonzero(brightness > light_threshold)

        # Calculate percentages
        dark_percentage = (dark_pixels / total_pixels) * 100
        light_percentage = (light_pixels / total_pixels) * 100

        return dark_percentage, light_percentage

    def _is_region_present(
        self, region: dict[str, Any], screenshot: np.ndarray[Any, Any]
    ) -> bool:
        """Check if a region is present in a screenshot using masked similarity."""
        x, y, x2, y2 = region["x"], region["y"], region["x2"], region["y2"]

        # Extract region from screenshot
        roi = screenshot[y : y2 + 1, x : x2 + 1]

        # Compare with original
        if roi.shape != region["pixel_data"].shape:
            return False

        # Get cached expanded mask if available
        mask_expanded = region.get("mask_expanded")

        if mask_expanded is None:
            # Get or create mask
            mask = region.get("mask")
            if mask is None:
                # No mask - do simple comparison
                # Calculate mean absolute difference
                diff = np.mean(
                    np.abs(
                        roi.astype(np.float32) - region["pixel_data"].astype(np.float32)
                    )
                )
                similarity = 1.0 - (diff / 255.0)
                return cast(bool, similarity >= self.config.similarity_threshold)

            # Expand mask once and cache it in the region
            mask_expanded = (
                np.expand_dims(mask, axis=2) if len(roi.shape) == 3 else mask
            )
            region["mask_expanded"] = mask_expanded  # Cache for future use

            # Also cache active pixel count
            region["active_pixels"] = np.count_nonzero(mask > 0.5)

        active_pixels = region.get(
            "active_pixels",
            (
                np.count_nonzero(mask_expanded[:, :, 0] > 0.5)
                if len(mask_expanded.shape) == 3
                else np.count_nonzero(mask_expanded > 0.5)
            ),
        )

        if active_pixels == 0:
            return False  # No active pixels to compare

        # Fast vectorized similarity calculation
        # Use integer arithmetic when possible
        diff = np.abs(roi.astype(np.int16) - region["pixel_data"].astype(np.int16))

        # Apply mask and calculate mean difference for active pixels only
        masked_diff = diff * mask_expanded
        total_diff = np.sum(masked_diff)

        num_channels = 3 if len(roi.shape) == 3 else 1
        avg_diff = total_diff / (active_pixels * num_channels * 255.0)
        similarity = 1.0 - avg_diff

        # Use similarity threshold from config
        return cast(bool, similarity >= self.config.similarity_threshold)

    def _decompose_complex_regions(
        self, regions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Decompose complex regions into rectangles.

        FUTURE ENHANCEMENT: For complex shapes (e.g., L-shaped window frames),
        decompose into multiple simple rectangles (e.g., 4 separate borders).
        This would improve matching performance and reduce false positives.

        Current behavior: Returns regions unchanged (no decomposition).
        """
        return regions

    def _calculate_statistics(
        self,
        screenshots: list[np.ndarray[Any, Any]],
        state_images: list[StateImage],
        states: list[DiscoveredState],
        stability_map: np.ndarray[Any, Any],
    ) -> dict[str, Any]:
        """Calculate analysis statistics."""
        total_pixels = stability_map.size
        stable_pixels = np.sum(stability_map)

        return {
            "total_screenshots": len(screenshots),
            "states_found": len(states),
            "state_images_found": len(state_images),
            "average_state_images_per_state": (
                len(state_images) / len(states) if states else 0
            ),
            "pixel_stability_score": (
                stable_pixels / total_pixels if total_pixels else 0
            ),
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
