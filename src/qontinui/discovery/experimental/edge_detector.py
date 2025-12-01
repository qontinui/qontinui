"""
EXPERIMENTAL: Edge-based GUI element detection using Canny edge detection.

WARNING: This module is experimental and unstable. APIs may change without notice.
Do not use in production code.

This detector uses edge detection to identify UI elements by finding regions
with strong edges. It's a traditional computer vision approach that may work
well for high-contrast UIs.
"""

from typing import Any

import cv2
import numpy as np

from .base_detector import BaseDetector
from .types import BBox


class EdgeBasedDetector(BaseDetector):
    """Detect GUI elements using Canny edge detection and contour finding.

    EXPERIMENTAL: This detector is experimental and uses traditional edge
    detection techniques to identify UI boundaries.

    Works best with:
    - High-contrast interfaces
    - Well-defined element boundaries
    - Simple, rectangular UI elements
    """

    def __init__(self):
        super().__init__("Edge-Based Detector")

    def detect(self, image_path: str, **params) -> list[BBox]:
        """
        Detect elements using Canny edge detection

        Parameters:
            canny_low: Low threshold for Canny (default: 50)
            canny_high: High threshold for Canny (default: 150)
            min_area: Minimum contour area (default: 100)
            max_area: Maximum contour area (default: None)
            dilation_kernel: Size of dilation kernel (default: 3)
            dilation_iterations: Number of dilation iterations (default: 1)
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Parameters
        canny_low = params.get("canny_low", 50)
        canny_high = params.get("canny_high", 150)
        min_area = params.get("min_area", 100)
        max_area = params.get("max_area", img.shape[0] * img.shape[1] * 0.9)
        kernel_size = params.get("dilation_kernel", 3)
        iterations = params.get("dilation_iterations", 1)

        # Edge detection
        edges = cv2.Canny(gray, canny_low, canny_high)

        # Dilate to connect nearby edges
        if iterations > 0:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=iterations)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Convert contours to bounding boxes
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area >= min_area and area <= max_area:
                boxes.append(BBox(x, y, x + w, y + h))

        # Post-process
        boxes = self.remove_contained_boxes(boxes)
        boxes = self.merge_overlapping_boxes(boxes, iou_threshold=0.7)

        return boxes

    def get_param_grid(self) -> list[dict[str, Any]]:
        """Parameter grid for hyperparameter search"""
        return [
            # Conservative - high thresholds
            {"canny_low": 100, "canny_high": 200, "dilation_iterations": 0},
            {"canny_low": 100, "canny_high": 200, "dilation_iterations": 1},
            # Moderate
            {"canny_low": 50, "canny_high": 150, "dilation_iterations": 0},
            {"canny_low": 50, "canny_high": 150, "dilation_iterations": 1},
            {"canny_low": 50, "canny_high": 150, "dilation_iterations": 2},
            # Aggressive - low thresholds
            {"canny_low": 30, "canny_high": 100, "dilation_iterations": 1},
            {"canny_low": 30, "canny_high": 100, "dilation_iterations": 2},
            # Very aggressive
            {"canny_low": 20, "canny_high": 60, "dilation_iterations": 2},
            {"canny_low": 20, "canny_high": 60, "dilation_iterations": 3},
        ]
