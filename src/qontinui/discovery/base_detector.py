"""Base class for all detection algorithms in qontinui.

This module provides the abstract base class for single-image detectors,
establishing a common interface and utility methods for all detection algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np


class BaseDetector(ABC):
    """Base class for single-image detection algorithms.

    This abstract class defines the interface for all detection algorithms
    that operate on a single image. Detectors analyze images to find specific
    patterns, regions, or UI elements.

    Attributes:
        name: Human-readable name for this detector.

    Example:
        >>> class MyDetector(BaseDetector):
        ...     def detect(self, image, **params):
        ...         # Implementation here
        ...         return [{"bbox": (x, y, w, h), "confidence": 0.9}]
        ...
        >>> detector = MyDetector("my_detector")
        >>> results = detector.detect(image, threshold=0.5)
    """

    def __init__(self, name: str) -> None:
        """Initialize the detector.

        Args:
            name: Human-readable name for this detector.
        """
        self.name = name

    @abstractmethod
    def detect(self, image: np.ndarray[Any, Any], **params: Any) -> list[dict[str, Any]]:
        """Detect elements or regions in a single image.

        This is the main detection method that must be implemented by all
        concrete detector classes. It should analyze the input image and
        return a list of detected regions/elements.

        Args:
            image: Input image as numpy array (BGR or grayscale).
            **params: Algorithm-specific parameters. Common parameters include:
                - threshold: Detection confidence threshold (float)
                - min_size: Minimum region size as (width, height) tuple
                - max_size: Maximum region size as (width, height) tuple

        Returns:
            List of detections, where each detection is a dictionary containing:
                - bbox: Bounding box as (x, y, w, h) tuple
                - confidence: Detection confidence score (0.0-1.0)
                - type: Optional element type string
                - Additional algorithm-specific metadata

        Example:
            >>> results = detector.detect(image, threshold=0.8)
            >>> for detection in results:
            ...     x, y, w, h = detection["bbox"]
            ...     confidence = detection["confidence"]
            ...     print(f"Found at ({x}, {y}) with confidence {confidence}")
        """
        pass

    def get_param_grid(self) -> list[dict[str, Any]]:
        """Return parameter grid for hyperparameter search.

        This method provides a list of parameter configurations that can be
        used for hyperparameter tuning or grid search optimization. Override
        this method to specify parameter ranges for your detector.

        Returns:
            List of parameter dictionaries to try during hyperparameter search.
            Empty list indicates no hyperparameter search is configured.

        Example:
            >>> def get_param_grid(self):
            ...     return [
            ...         {"threshold": 0.5, "min_size": (10, 10)},
            ...         {"threshold": 0.7, "min_size": (20, 20)},
            ...         {"threshold": 0.9, "min_size": (30, 30)},
            ...     ]
        """
        return []

    @staticmethod
    def merge_overlapping_boxes(
        boxes: list[tuple[int, int, int, int]], overlap_threshold: float = 0.5
    ) -> list[tuple[int, int, int, int]]:
        """Merge overlapping bounding boxes using Non-Maximum Suppression.

        This utility method helps reduce duplicate detections by merging
        boxes that overlap significantly. Uses a greedy algorithm based on
        Intersection over Union (IoU).

        Args:
            boxes: List of bounding boxes as (x, y, w, h) tuples.
            overlap_threshold: IoU threshold for merging (0.0-1.0).
                Higher values require more overlap to merge.

        Returns:
            List of merged bounding boxes as (x, y, w, h) tuples.

        Example:
            >>> boxes = [(10, 10, 50, 50), (15, 15, 50, 50), (100, 100, 30, 30)]
            >>> merged = BaseDetector.merge_overlapping_boxes(boxes, 0.5)
            >>> print(len(merged))  # Should be 2 (first two merged)
            2
        """
        if not boxes:
            return []

        # Convert to (x1, y1, x2, y2) format for easier IoU calculation
        boxes_xyxy = [(x, y, x + w, y + h) for x, y, w, h in boxes]

        # Sort by area (largest first) to prioritize larger boxes
        boxes_with_area = [(box, (box[2] - box[0]) * (box[3] - box[1])) for box in boxes_xyxy]
        boxes_with_area.sort(key=lambda x: x[1], reverse=True)
        sorted_boxes = [box for box, _ in boxes_with_area]

        merged = []
        used = [False] * len(sorted_boxes)

        for i, box1 in enumerate(sorted_boxes):
            if used[i]:
                continue

            # Start with current box
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box1

            # Find all boxes that overlap significantly
            for j, box2 in enumerate(sorted_boxes[i + 1 :], start=i + 1):
                if used[j]:
                    continue

                iou = BaseDetector._calculate_iou(box1, box2)
                if iou >= overlap_threshold:
                    # Merge by expanding bounding box
                    x2_min = min(x2_min, box2[0])
                    y2_min = min(y2_min, box2[1])
                    x2_max = max(x2_max, box2[2])
                    y2_max = max(y2_max, box2[3])
                    used[j] = True

            # Add merged box (convert back to xywh format)
            merged.append((x2_min, y2_min, x2_max - x2_min, y2_max - y2_min))
            used[i] = True

        return merged

    @staticmethod
    def remove_contained_boxes(
        boxes: list[tuple[int, int, int, int]], containment_threshold: float = 0.9
    ) -> list[tuple[int, int, int, int]]:
        """Remove boxes that are contained within other boxes.

        This utility method removes smaller boxes that are largely or entirely
        contained within larger boxes, helping to eliminate redundant detections.

        Args:
            boxes: List of bounding boxes as (x, y, w, h) tuples.
            containment_threshold: Fraction of smaller box that must be
                contained (0.0-1.0). Higher values require more containment.

        Returns:
            List of boxes with contained boxes removed.

        Example:
            >>> boxes = [(10, 10, 100, 100), (20, 20, 30, 30), (200, 200, 50, 50)]
            >>> filtered = BaseDetector.remove_contained_boxes(boxes, 0.9)
            >>> print(len(filtered))  # Should be 2 (middle box removed)
            2
        """
        if not boxes:
            return []

        # Sort by area (largest first)
        boxes_with_area = [(box, box[2] * box[3]) for box in boxes]
        boxes_with_area.sort(key=lambda x: x[1], reverse=True)
        sorted_boxes = [box for box, _ in boxes_with_area]

        keep: list[Any] = []

        for _i, box1 in enumerate(sorted_boxes):
            x1, y1, w1, h1 = box1
            is_contained = False

            # Check if this box is contained in any larger box already kept
            for box2 in keep:
                x2, y2, w2, h2 = box2

                # Calculate intersection
                x_left = max(x1, x2)
                y_top = max(y1, y2)
                x_right = min(x1 + w1, x2 + w2)
                y_bottom = min(y1 + h1, y2 + h2)

                if x_right > x_left and y_bottom > y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    box1_area = w1 * h1

                    # Check if box1 is mostly contained in box2
                    if intersection_area >= containment_threshold * box1_area:
                        is_contained = True
                        break

            if not is_contained:
                keep.append(box1)

        return keep

    @staticmethod
    def filter_by_size(
        boxes: list[tuple[int, int, int, int]],
        min_size: tuple[int, int] | None = None,
        max_size: tuple[int, int] | None = None,
    ) -> list[tuple[int, int, int, int]]:
        """Filter bounding boxes by size constraints.

        Args:
            boxes: List of bounding boxes as (x, y, w, h) tuples.
            min_size: Minimum size as (width, height) tuple. None = no minimum.
            max_size: Maximum size as (width, height) tuple. None = no maximum.

        Returns:
            List of boxes that meet size constraints.

        Example:
            >>> boxes = [(0, 0, 10, 10), (0, 0, 50, 50), (0, 0, 200, 200)]
            >>> filtered = BaseDetector.filter_by_size(
            ...     boxes, min_size=(20, 20), max_size=(100, 100)
            ... )
            >>> print(len(filtered))  # Should be 1 (only middle box)
            1
        """
        filtered = []

        for box in boxes:
            x, y, w, h = box

            # Check minimum size
            if min_size is not None:
                min_w, min_h = min_size
                if w < min_w or h < min_h:
                    continue

            # Check maximum size
            if max_size is not None:
                max_w, max_h = max_size
                if w > max_w or h > max_h:
                    continue

            filtered.append(box)

        return filtered

    @staticmethod
    def _calculate_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IoU) for two boxes.

        Args:
            box1: First box as (x1, y1, x2, y2) tuple.
            box2: Second box as (x1, y1, x2, y2) tuple.

        Returns:
            IoU value between 0.0 and 1.0.
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # Calculate intersection
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersection_area

        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    @staticmethod
    def convert_to_grayscale(image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Convert image to grayscale if needed.

        Args:
            image: Input image (BGR, RGB, or grayscale).

        Returns:
            Grayscale image as 2D numpy array.
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def resize_image(
        image: np.ndarray[Any, Any],
        target_size: tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray[Any, Any]:
        """Resize image to target size.

        Args:
            image: Input image.
            target_size: Target size as (width, height) tuple.
            interpolation: OpenCV interpolation method.

        Returns:
            Resized image.
        """
        return cv2.resize(image, target_size, interpolation=interpolation)

    def __repr__(self) -> str:
        """Return string representation of detector."""
        return f"{self.__class__.__name__}(name='{self.name}')"
