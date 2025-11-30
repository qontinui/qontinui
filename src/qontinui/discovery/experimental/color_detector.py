"""
EXPERIMENTAL: Color-based detection using k-means clustering and color segmentation.

WARNING: This module is experimental and unstable. APIs may change without notice.
Do not use in production code.

This detector uses k-means clustering to segment the image by color, then
identifies connected regions within each color cluster.
"""

from typing import Any

import cv2
import numpy as np

from .base_detector import BaseDetector
from .types import BBox


class ColorClusterDetector(BaseDetector):
    """Detect GUI elements using color clustering.

    EXPERIMENTAL: This detector is experimental and uses k-means clustering
    to segment the image by color before detecting elements.

    Useful for:
    - Color-coded UI elements
    - Distinct visual regions
    - Interfaces with clear color separation
    """

    def __init__(self):
        super().__init__("Color Cluster Detector")

    def detect(self, image_path: str, **params) -> list[BBox]:
        """
        Detect elements using color clustering

        Parameters:
            n_clusters: Number of color clusters (default: 8)
            min_area: Minimum area (default: 100)
            use_hsv: Use HSV color space instead of RGB (default: False)
        """
        img = cv2.imread(image_path)
        if img is None:
            return []

        # Parameters
        n_clusters = params.get("n_clusters", 8)
        min_area = params.get("min_area", 100)
        max_area = params.get("max_area", img.shape[0] * img.shape[1] * 0.9)
        use_hsv = params.get("use_hsv", False)

        # Convert color space
        if use_hsv:
            color_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        else:
            color_img = img.copy()

        # Reshape for k-means
        pixels = color_img.reshape(-1, 3).astype(np.float32)

        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )  # type: ignore[call-overload]

        # Reshape labels back to image
        labels = labels.reshape(img.shape[:2])

        # Find connected components for each cluster
        boxes = []
        for cluster_id in range(n_clusters):
            mask = (labels == cluster_id).astype(np.uint8) * 255

            # Find contours in this cluster
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
            {"n_clusters": 4, "use_hsv": False},
            {"n_clusters": 6, "use_hsv": False},
            {"n_clusters": 8, "use_hsv": False},
            {"n_clusters": 10, "use_hsv": False},
            {"n_clusters": 12, "use_hsv": False},
            {"n_clusters": 4, "use_hsv": True},
            {"n_clusters": 6, "use_hsv": True},
            {"n_clusters": 8, "use_hsv": True},
            {"n_clusters": 10, "use_hsv": True},
        ]
