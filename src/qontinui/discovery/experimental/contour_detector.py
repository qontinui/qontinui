"""
EXPERIMENTAL: Contour-based detection using adaptive thresholding.

WARNING: This module is experimental and unstable. APIs may change without notice.
Do not use in production code.

This detector uses adaptive thresholding to identify regions and then finds
contours around those regions to detect UI elements.
"""

from typing import Any, Dict, List

import cv2
import numpy as np

from .base_detector import BaseDetector
from .types import BBox


class ContourDetector(BaseDetector):
    """Detect GUI elements using adaptive thresholding and contours.

    EXPERIMENTAL: This detector is experimental and uses adaptive thresholding
    to segment the image before finding contours.

    Effective for:
    - Text-heavy interfaces
    - Varying lighting conditions
    - Complex backgrounds
    """

    def __init__(self):
        super().__init__("Contour Detector")

    def detect(self, image_path: str, **params) -> List[BBox]:
        """
        Detect elements using adaptive thresholding

        Parameters:
            block_size: Block size for adaptive threshold (default: 11)
            c_value: Constant subtracted from mean (default: 2)
            min_area: Minimum area (default: 100)
            use_otsu: Use Otsu thresholding instead (default: False)
            invert: Invert threshold (default: False)
        """
        img = cv2.imread(image_path)
        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Parameters
        block_size = params.get("block_size", 11)
        c_value = params.get("c_value", 2)
        min_area = params.get("min_area", 100)
        max_area = params.get("max_area", img.shape[0] * img.shape[1] * 0.9)
        use_otsu = params.get("use_otsu", False)
        invert = params.get("invert", False)

        # Thresholding
        if use_otsu:
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Ensure block_size is odd
            if block_size % 2 == 0:
                block_size += 1

            thresh = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                c_value,
            )

        if invert:
            thresh = cv2.bitwise_not(thresh)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Convert to bounding boxes
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area >= min_area and area <= max_area:
                boxes.append(BBox(x, y, x + w, y + h))

        # Post-process
        boxes = self.remove_contained_boxes(boxes)
        boxes = self.merge_overlapping_boxes(boxes)

        return boxes

    def get_param_grid(self) -> List[Dict[str, Any]]:
        """Parameter grid for hyperparameter search"""
        return [
            # Adaptive threshold with different parameters
            {"block_size": 11, "c_value": 2, "invert": False},
            {"block_size": 11, "c_value": 2, "invert": True},
            {"block_size": 15, "c_value": 3, "invert": False},
            {"block_size": 15, "c_value": 3, "invert": True},
            {"block_size": 21, "c_value": 5, "invert": False},
            {"block_size": 21, "c_value": 5, "invert": True},
            # Otsu thresholding
            {"use_otsu": True, "invert": False},
            {"use_otsu": True, "invert": True},
        ]
