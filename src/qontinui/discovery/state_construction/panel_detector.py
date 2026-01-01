"""Panel and container detection for UI analysis.

This module provides specialized panel detection capabilities using contour
detection and border analysis to identify rectangular UI panels, dialogs,
and containers.
"""

from typing import Any

import cv2
import numpy as np

from qontinui.discovery.state_construction.element_identifier import (
    IdentifiedRegion,
    RegionType,
)


class PanelDetector:
    """Detects bordered panels and containers in screenshots.

    Identifies rectangular regions with borders using contour detection,
    color analysis, and border density calculations. Common in UI panels,
    dialogs, and containers.
    """

    def __init__(
        self,
        min_region_size: tuple[int, int] = (20, 20),
        max_region_size: tuple[int, int] = (2000, 2000),
        min_rectangularity: float = 0.7,
        min_border_density: float = 0.3,
    ):
        """Initialize the panel detector.

        Args:
            min_region_size: Minimum region size (width, height)
            max_region_size: Maximum region size (width, height)
            min_rectangularity: Minimum area ratio for rectangular shapes
            min_border_density: Minimum perimeter edge density for panels
        """
        self.min_region_size = min_region_size
        self.max_region_size = max_region_size
        self.min_rectangularity = min_rectangularity
        self.min_border_density = min_border_density

    def detect_panel_regions(self, screenshot: np.ndarray) -> list[IdentifiedRegion]:
        """Detect bordered panels or containers.

        Identifies rectangular regions with borders using contour detection
        and color analysis. Common in UI panels, dialogs, and containers.

        Args:
            screenshot: Screenshot image as numpy array (BGR format)

        Returns:
            List of detected panel regions
        """
        panel_regions = []
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to find edges
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size
            if not self._is_valid_size((w, h)):
                continue

            # Check if it looks like a panel (rectangular, has area)
            area = cv2.contourArea(contour)
            bbox_area = w * h

            if bbox_area == 0:
                continue

            # Panel should be fairly rectangular
            rectangularity = area / bbox_area
            if rectangularity < self.min_rectangularity:
                continue

            # Extract region for analysis
            region_img = screenshot[y : y + h, x : x + w]

            # Analyze if it has panel characteristics
            properties = self._analyze_panel_properties(region_img)

            if properties["is_panel"]:
                panel = IdentifiedRegion(
                    region_type=RegionType.PANEL,
                    bounds=(x, y, w, h),
                    confidence=properties["confidence"],
                    properties=properties,
                    sub_elements=[],
                )
                panel_regions.append(panel)

        return panel_regions

    def _analyze_panel_properties(self, region_img: np.ndarray) -> dict[str, Any]:
        """Analyze if a region has panel characteristics.

        Checks for border presence by analyzing edge density around the
        perimeter of the region.

        Args:
            region_img: Region image to analyze

        Returns:
            Dictionary with is_panel flag, confidence, and border properties
        """
        properties: dict[str, Any] = {"is_panel": False, "confidence": 0.0}

        if region_img.size == 0:
            return properties

        # Check for border (edges around perimeter)
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Check perimeter edge density
        h, w = edges.shape
        if h < 4 or w < 4:
            return properties

        border_thickness = 2
        top_border = edges[0:border_thickness, :]
        bottom_border = edges[h - border_thickness : h, :]
        left_border = edges[:, 0:border_thickness]
        right_border = edges[:, w - border_thickness : w]

        border_density = (
            np.sum(top_border > 0)
            + np.sum(bottom_border > 0)
            + np.sum(left_border > 0)
            + np.sum(right_border > 0)
        ) / (2 * border_thickness * (h + w))

        # Panels typically have visible borders
        if border_density > self.min_border_density:
            properties["is_panel"] = True
            properties["confidence"] = min(0.9, border_density)
            properties["border_density"] = float(border_density)

        return properties

    def _is_valid_size(self, size: tuple[int, int]) -> bool:
        """Check if size is within valid range.

        Args:
            size: (width, height) tuple

        Returns:
            True if size is valid
        """
        w, h = size
        min_w, min_h = self.min_region_size
        max_w, max_h = self.max_region_size

        return min_w <= w <= max_w and min_h <= h <= max_h  # type: ignore[no-any-return]
