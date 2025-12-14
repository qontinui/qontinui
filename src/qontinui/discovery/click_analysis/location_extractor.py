"""Click-location-based image region extraction.

This module provides functionality for extracting image regions around user click
points. This is useful for identifying UI elements that were interacted with during
recording sessions.

Key Features:
- Extract regions around click coordinates with configurable padding
- Validate extracted regions against size constraints
- Extract context images for better matching
- Integration with StateImage model

Example:
    >>> from qontinui.discovery.click_analysis import ClickLocationExtractor
    >>> extractor = ClickLocationExtractor()
    >>> state_image = extractor.extract_at_location(
    ...     screenshot=screenshot_array,
    ...     click_x=350,
    ...     click_y=250,
    ...     padding=20
    ... )
"""

import logging
from typing import Any

import numpy as np

from ..models import StateImage

logger = logging.getLogger(__name__)


class ClickLocationExtractor:
    """Extracts image regions around click locations.

    This extractor creates StateImage objects from regions around user click
    points. It applies size validation and extracts both the target region
    and a larger context area for improved matching.
    """

    def __init__(
        self,
        min_size: tuple[int, int] = (20, 20),
        max_size: tuple[int, int] = (500, 500),
        similarity_threshold: float = 0.85,
    ) -> None:
        """Initialize the click location extractor.

        Args:
            min_size: Minimum (width, height) for extracted regions.
            max_size: Maximum (width, height) for extracted regions.
            similarity_threshold: Default similarity threshold for matching.
        """
        self.min_size = min_size
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold

    def extract_at_location(
        self,
        screenshot: np.ndarray[Any, Any],
        click_x: int,
        click_y: int,
        padding: int = 20,
        name: str | None = None,
    ) -> StateImage | None:
        """Extract image region around a specific click location.

        Creates a StateImage from the region around the click point. The region
        is defined as a square centered on the click with the specified padding.
        A larger context region (2x padding) is also extracted for improved
        matching accuracy.

        Args:
            screenshot: Screenshot image (BGR or RGB format).
            click_x: X coordinate of the click point.
            click_y: Y coordinate of the click point.
            padding: Padding around the click point in pixels.
            name: Optional name for the StateImage. If None, uses coordinates.

        Returns:
            StateImage object if extraction succeeds, None if region is invalid.
        """
        try:
            height, width = screenshot.shape[:2]

            # Calculate bounding box with padding
            x1 = max(0, click_x - padding)
            y1 = max(0, click_y - padding)
            x2 = min(width, click_x + padding)
            y2 = min(height, click_y + padding)

            # Validate size
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            if bbox_w < self.min_size[0] or bbox_h < self.min_size[1]:
                logger.debug(
                    f"Region too small at ({click_x}, {click_y}): {bbox_w}x{bbox_h}"
                )
                return None

            if bbox_w > self.max_size[0] or bbox_h > self.max_size[1]:
                logger.debug(
                    f"Region too large at ({click_x}, {click_y}): {bbox_w}x{bbox_h}"
                )
                return None

            # Extract image
            extracted = screenshot[y1:y2, x1:x2].copy()

            # Extract larger context (2x padding)
            context_padding = padding * 2
            cx1 = max(0, click_x - context_padding)
            cy1 = max(0, click_y - context_padding)
            cx2 = min(width, click_x + context_padding)
            cy2 = min(height, click_y + context_padding)
            context = screenshot[cy1:cy2, cx1:cx2].copy()

            # Generate name if not provided
            if name is None:
                name = f"click_{click_x}_{click_y}"

            # Create pixel hash from image data
            from hashlib import md5

            pixel_hash = md5(extracted.tobytes()).hexdigest()

            # Create StateImage
            state_image = StateImage(
                id=f"img_{pixel_hash[:12]}",
                name=name,
                x=x1,
                y=y1,
                x2=x2,
                y2=y2,
                pixel_hash=pixel_hash,
                frequency=0.0,  # Will be set later during analysis
                pixel_data=extracted,
                tags=["click_extraction"],
            )

            return state_image

        except Exception as e:
            logger.error(f"Error extracting at location ({click_x}, {click_y}): {e}")
            return None

    def extract_multiple(
        self,
        screenshot: np.ndarray[Any, Any],
        click_locations: list[tuple[int, int]],
        padding: int = 20,
    ) -> list[StateImage]:
        """Extract regions around multiple click locations.

        Args:
            screenshot: Screenshot image.
            click_locations: List of (x, y) click coordinates.
            padding: Padding around each click point.

        Returns:
            List of successfully extracted StateImage objects.
        """
        extracted_images = []

        for click_x, click_y in click_locations:
            state_image = self.extract_at_location(
                screenshot=screenshot,
                click_x=click_x,
                click_y=click_y,
                padding=padding,
            )
            if state_image:
                extracted_images.append(state_image)

        return extracted_images
