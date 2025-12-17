"""
EXPERIMENTAL: Base detector classes for experimental detection algorithms.

WARNING: This module is experimental and unstable. APIs may change without notice.
Do not use in production code.

This module provides abstract base classes for implementing custom detection
algorithms. These are research implementations and may not have the same
performance or stability guarantees as the main qontinui detection pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .types import BBox

if TYPE_CHECKING:
    from .types import MultiScreenshotDataset


class BaseDetector(ABC):
    """Abstract base class for all experimental detectors.

    EXPERIMENTAL: This is a base class for experimental detection algorithms.
    Implementations may be unstable and are subject to change.

    Usage:
        Inherit from this class to implement custom detection algorithms.
        Override the `detect` method to provide detection logic.
    """

    def __init__(self, name: str):
        self.name = name
        self.params: dict[str, Any] = {}

    @abstractmethod
    def detect(self, image_path: str, **params) -> list[BBox]:
        """
        Detect GUI elements in an image

        Args:
            image_path: Path to the image file
            **params: Detector-specific parameters

        Returns:
            List of detected bounding boxes
        """
        pass

    def set_params(self, **params):
        """Update detector parameters"""
        self.params.update(params)

    def get_param_grid(self) -> list[dict[str, Any]]:
        """
        Return parameter grid for hyperparameter search

        Returns:
            List of parameter dictionaries to try
        """
        return [{}]  # Default: no parameters

    @staticmethod
    def merge_overlapping_boxes(
        boxes: list[BBox], iou_threshold: float = 0.5
    ) -> list[BBox]:
        """Merge highly overlapping boxes"""
        if not boxes:
            return []

        # Sort by area (largest first)
        boxes = sorted(boxes, key=lambda b: b.area, reverse=True)

        merged = []
        used = set()

        for i, box1 in enumerate(boxes):
            if i in used:
                continue

            # Find all boxes that overlap significantly
            group = [box1]
            for j, box2 in enumerate(boxes[i + 1 :], start=i + 1):
                if j in used:
                    continue

                # Compute IoU
                x1 = max(box1.x1, box2.x1)
                y1 = max(box1.y1, box2.y1)
                x2 = min(box1.x2, box2.x2)
                y2 = min(box1.y2, box2.y2)

                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    union = box1.area + box2.area - intersection
                    iou = intersection / union if union > 0 else 0

                    if iou >= iou_threshold:
                        group.append(box2)
                        used.add(j)

            # Merge group by taking bounding box of all boxes
            if len(group) == 1:
                merged.append(group[0])
            else:
                min_x = min(b.x1 for b in group)
                min_y = min(b.y1 for b in group)
                max_x = max(b.x2 for b in group)
                max_y = max(b.y2 for b in group)
                merged.append(BBox(min_x, min_y, max_x, max_y))

        return merged

    @staticmethod
    def filter_boxes_by_size(
        boxes: list[BBox],
        min_area: int = 100,
        max_area: int | None = None,
        min_width: int = 5,
        min_height: int = 5,
    ) -> list[BBox]:
        """Filter boxes by size constraints"""
        filtered = []
        for box in boxes:
            width = box.x2 - box.x1
            height = box.y2 - box.y1
            area = box.area

            if width < min_width or height < min_height:
                continue
            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue

            filtered.append(box)

        return filtered

    @staticmethod
    def remove_contained_boxes(boxes: list[BBox], threshold: float = 0.9) -> list[BBox]:
        """Remove boxes that are almost entirely contained in other boxes"""
        if not boxes:
            return []

        # Sort by area (largest first)
        boxes = sorted(boxes, key=lambda b: b.area, reverse=True)

        keep = []

        for i, box1 in enumerate(boxes):
            is_contained = False

            for box2 in boxes[:i]:  # Check against larger boxes
                # Calculate overlap
                x1 = max(box1.x1, box2.x1)
                y1 = max(box1.y1, box2.y1)
                x2 = min(box1.x2, box2.x2)
                y2 = min(box1.y2, box2.y2)

                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    # Check if box1 is mostly contained in box2
                    if intersection / box1.area >= threshold:
                        is_contained = True
                        break

            if not is_contained:
                keep.append(box1)

        return keep


class MultiScreenshotDetector(ABC):
    """Abstract base class for detectors that work across multiple screenshots.

    EXPERIMENTAL: This is a base class for experimental multi-screenshot
    detection algorithms. Implementations may be unstable and are subject to change.

    Usage:
        Inherit from this class to implement detection algorithms that analyze
        multiple screenshots to find consistent or varying elements.
    """

    def __init__(self, name: str):
        self.name = name
        self.params: dict[str, Any] = {}

    @abstractmethod
    def detect_multi(
        self, dataset: MultiScreenshotDataset, **params
    ) -> dict[int, list[BBox]]:
        """
        Detect GUI elements across multiple screenshots

        Args:
            dataset: MultiScreenshotDataset containing screenshots and annotations
            **params: Detector-specific parameters

        Returns:
            Dict mapping screenshot_id to list of detected bounding boxes
        """
        pass

    def set_params(self, **params):
        """Update detector parameters"""
        self.params.update(params)

    def get_param_grid(self) -> list[dict[str, Any]]:
        """
        Return parameter grid for hyperparameter search

        Returns:
            List of parameter dictionaries to try
        """
        return [{}]  # Default: no parameters
