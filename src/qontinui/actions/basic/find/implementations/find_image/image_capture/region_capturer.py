"""Region-specific capture implementation."""

import logging
from typing import Any

import cv2
import numpy as np

from .......model.element.region import Region

logger = logging.getLogger(__name__)


class RegionCapturer:
    """Handles region-specific capture operations.

    Provides clean interface for capturing specific screen regions,
    routing through the controller abstraction layer.
    """

    def capture(self, region: Region) -> np.ndarray[Any, Any] | None:
        """Capture specific screen region.

        Args:
            region: Region to capture

        Returns:
            Region image as BGR numpy array or None on error
        """
        try:
            from ......wrappers import get_controller

            controller = get_controller()
            screenshot = controller.capture.capture_region(region)

            # Convert PIL Image to numpy array (BGR for OpenCV)
            img_array = np.array(screenshot)

            # Convert RGB to BGR for OpenCV
            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            logger.debug(f"Captured region {region}: {img_array.shape}")
            return img_array

        except Exception as e:
            logger.error(f"Error capturing region {region}: {e}", exc_info=True)
            return None

    def capture_multiple(self, regions: list[Region]) -> list[np.ndarray[Any, Any]]:
        """Capture multiple regions.

        Args:
            regions: List of regions to capture

        Returns:
            List of captured images (excludes failed captures)
        """
        images = []
        for region in regions:
            img = self.capture(region)
            if img is not None:
                images.append(img)
        return images
