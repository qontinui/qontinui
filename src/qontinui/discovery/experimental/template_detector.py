"""
EXPERIMENTAL: Template-based detection using corner detection and pattern matching.

WARNING: This module is experimental and unstable. APIs may change without notice.
Do not use in production code.

This detector uses corner detection (good features to track) and morphological
operations to identify rectangular UI elements.
"""

from typing import Any

import cv2
import numpy as np

from .base_detector import BaseDetector
from .types import BBox


class TemplateDetector(BaseDetector):
    """Detect GUI elements using corner detection and rectangular patterns.

    EXPERIMENTAL: This detector is experimental and uses corner detection
    to identify potential UI element boundaries.

    Works well for:
    - Button detection
    - Panel and card detection
    - Rectangular UI elements with corners
    """

    def __init__(self):
        super().__init__("Template/Corner Detector")

    def detect(self, image_path: str, **params) -> list[BBox]:
        """
        Detect elements using corner detection

        Parameters:
            quality_level: Quality level for corner detection (default: 0.01)
            min_distance: Minimum distance between corners (default: 10)
            block_size: Block size for corner detection (default: 3)
            expansion: Pixels to expand around corners to form boxes (default: 20)
        """
        img = cv2.imread(image_path)
        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Parameters
        quality_level = params.get("quality_level", 0.01)
        min_distance = params.get("min_distance", 10)
        block_size = params.get("block_size", 3)
        expansion = params.get("expansion", 20)
        min_area = params.get("min_area", 100)
        max_area = params.get("max_area", img.shape[0] * img.shape[1] * 0.9)

        # Detect corners
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=1000,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
        )

        if corners is None or len(corners) == 0:
            return []

        corners = corners.reshape(-1, 2).astype(int)

        # Create boxes around corner clusters
        boxes = []

        # Method 1: Expand around each corner
        for x, y in corners:
            x1 = max(0, x - expansion)
            y1 = max(0, y - expansion)
            x2 = min(img.shape[1], x + expansion)
            y2 = min(img.shape[0], y + expansion)

            area = (x2 - x1) * (y2 - y1)
            if area >= min_area and area <= max_area:
                boxes.append(BBox(x1, y1, x2, y2))

        # Method 2: Find rectangular regions using morphological operations
        # Create a mask with corners
        corner_mask = np.zeros_like(gray)
        for x, y in corners:
            cv2.circle(corner_mask, (x, y), 3, 255, -1)  # type: ignore[call-overload]

        # Dilate to connect nearby corners
        kernel = np.ones((15, 15), np.uint8)
        dilated = cv2.dilate(corner_mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area >= min_area and area <= max_area:
                boxes.append(BBox(x, y, x + w, y + h))

        # Post-process
        boxes = self.remove_contained_boxes(boxes)
        boxes = self.merge_overlapping_boxes(boxes)

        return boxes

    def get_param_grid(self) -> list[dict[str, Any]]:
        """Parameter grid for hyperparameter search"""
        return [
            # High quality - fewer corners
            {"quality_level": 0.05, "min_distance": 20, "expansion": 15},
            {"quality_level": 0.05, "min_distance": 15, "expansion": 20},
            # Moderate quality
            {"quality_level": 0.01, "min_distance": 10, "expansion": 20},
            {"quality_level": 0.01, "min_distance": 10, "expansion": 25},
            {"quality_level": 0.01, "min_distance": 15, "expansion": 20},
            # Lower quality - more corners
            {"quality_level": 0.005, "min_distance": 10, "expansion": 20},
            {"quality_level": 0.001, "min_distance": 10, "expansion": 25},
            # Very aggressive
            {"quality_level": 0.001, "min_distance": 5, "expansion": 30},
        ]
