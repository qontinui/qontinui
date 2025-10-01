"""Base interface for description generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class DescriptionGenerator(ABC):
    """Abstract base class for generating semantic descriptions of image regions.

    This interface allows different description generation strategies to be
    plugged into segmentation processors, enabling modular combination of
    segmentation (e.g., SAM2) and description (e.g., CLIP, BLIP, GPT-4V).
    """

    def __init__(self):
        """Initialize the description generator."""
        self._config: dict[str, Any] = {}

    @abstractmethod
    def generate(
        self,
        image: np.ndarray[Any, Any],
        mask: np.ndarray[Any, Any] | None = None,
        bbox: tuple[Any, ...] | None = None,
    ) -> str:
        """Generate a semantic description for an image region.

        Args:
            image: Full image or cropped region (BGR format)
            mask: Optional binary mask to focus on specific pixels
            bbox: Optional bounding box (x, y, w, h) if working with full image

        Returns:
            Semantic description of the region
        """
        pass

    @abstractmethod
    def batch_generate(self, image: np.ndarray[Any, Any], regions: list[Any]) -> list[str]:
        """Generate descriptions for multiple regions in batch.

        More efficient than calling generate() multiple times.

        Args:
            image: Full image (BGR format)
            regions: List of dicts with 'mask' and/or 'bbox' keys

        Returns:
            List of descriptions corresponding to each region
        """
        pass

    def configure(self, **kwargs) -> None:
        """Configure the generator with custom parameters.

        Args:
            **kwargs: Configuration parameters specific to the generator
        """
        self._config.update(kwargs)

    def get_config(self) -> dict[str, Any]:
        """Get current configuration.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this generator is available for use.

        Returns:
            True if all dependencies are met and models are loaded
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the underlying model.

        Returns:
            Model identifier string
        """
        pass

    def preprocess_image(
        self,
        image: np.ndarray[Any, Any],
        mask: np.ndarray[Any, Any] | None = None,
        bbox: tuple[Any, ...] | None = None,
    ) -> np.ndarray[Any, Any]:
        """Preprocess image region for description generation.

        Default implementation that can be overridden.

        Args:
            image: Input image
            mask: Optional mask
            bbox: Optional bounding box

        Returns:
            Preprocessed image region
        """
        # If bbox provided, crop to region
        if bbox is not None and len(image.shape) >= 2:
            x, y, w, h = bbox
            y_end = min(y + h, image.shape[0])
            x_end = min(x + w, image.shape[1])
            region = image[y:y_end, x:x_end]
        else:
            region = image

        # Apply mask if provided
        if mask is not None:
            if len(region.shape) == 3:
                # Apply mask to each channel
                masked = region.copy()
                mask_region = mask
                if bbox is not None:
                    x, y, w, h = bbox
                    y_end = min(y + h, mask.shape[0])
                    x_end = min(x + w, mask.shape[1])
                    mask_region = mask[y:y_end, x:x_end]

                for c in range(region.shape[2]):
                    masked[:, :, c] = region[:, :, c] * mask_region
                region = masked
            else:
                region = region * mask

        return region
