"""Create StateImage objects from stable regions."""

import logging
from typing import Any, cast

import numpy as np

from ...models import AnalysisConfig, StateImage

logger = logging.getLogger(__name__)


class StateImageFactory:
    """Creates StateImage objects from extracted stable regions."""

    def __init__(self, config: AnalysisConfig) -> None:
        """Initialize with analysis configuration."""
        self.config = config

    def create(
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
