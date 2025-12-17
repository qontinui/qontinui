"""Create stability maps from screenshots."""

from typing import Any, cast

import numpy as np

from ...models import AnalysisConfig


class StabilityMapCreator:
    """Creates pixel stability maps by analyzing variance across screenshots."""

    def __init__(self, config: AnalysisConfig) -> None:
        """Initialize with analysis configuration."""
        self.config = config

    def create(self, screenshots: list[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
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
            stability_map = np.all(pixel_variance < self.config.variance_threshold, axis=2).astype(
                np.uint8
            )
        else:
            stability_map = (pixel_variance < self.config.variance_threshold).astype(np.uint8)

        return cast(np.ndarray[Any, Any], stability_map)
