"""Element detector base implementation.

This module provides the core element detection interface and base implementations
that will be used by specific detection methods (template matching, feature detection,
OCR, ML-based detection).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DetectedElement:
    """Represents a detected UI element.

    Attributes:
        element_type: Type of element (button, text_field, icon, etc.)
        bounds: Bounding box (x, y, width, height)
        confidence: Detection confidence score (0.0 to 1.0)
        features: Additional features or metadata
        image: Optional cropped image of the element
    """

    element_type: str
    bounds: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    features: Optional[dict] = None
    image: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        """String representation of detected element."""
        return (
            f"DetectedElement(type={self.element_type}, "
            f"bounds={self.bounds}, confidence={self.confidence:.3f})"
        )


class ElementDetector(ABC):
    """Abstract base class for element detection implementations.

    Subclasses should implement specific detection strategies such as:
    - Template matching
    - Feature-based detection
    - OCR-based text detection
    - Machine learning classification
    """

    @abstractmethod
    def detect(self, screenshot: np.ndarray) -> List[DetectedElement]:
        """Detect elements in a screenshot.

        Args:
            screenshot: Screenshot image as numpy array

        Returns:
            List of detected elements
        """
        pass

    @abstractmethod
    def configure(self, **kwargs) -> None:
        """Configure detector parameters.

        Args:
            **kwargs: Configuration parameters specific to the detector
        """
        pass


class CompositeElementDetector(ElementDetector):
    """Combines multiple detection methods.

    This detector runs multiple detection strategies and combines their results,
    handling overlapping detections and selecting the best results.
    """

    def __init__(self, detectors: Optional[List[ElementDetector]] = None):
        """Initialize composite detector.

        Args:
            detectors: List of element detectors to combine
        """
        self.detectors = detectors or []
        self.confidence_threshold = 0.7
        self.iou_threshold = 0.5  # For non-maximum suppression

    def detect(self, screenshot: np.ndarray) -> List[DetectedElement]:
        """Detect elements using all configured detectors.

        Args:
            screenshot: Screenshot image

        Returns:
            Combined list of detected elements with duplicates removed
        """
        all_elements: List[DetectedElement] = []

        # Run all detectors
        for detector in self.detectors:
            elements = detector.detect(screenshot)
            all_elements.extend(elements)

        # Filter by confidence
        filtered = [e for e in all_elements if e.confidence >= self.confidence_threshold]

        # Apply non-maximum suppression to remove overlaps
        final_elements = self._non_maximum_suppression(filtered)

        return final_elements

    def configure(self, **kwargs) -> None:
        """Configure composite detector.

        Args:
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for non-maximum suppression
        """
        if "confidence_threshold" in kwargs:
            self.confidence_threshold = kwargs["confidence_threshold"]
        if "iou_threshold" in kwargs:
            self.iou_threshold = kwargs["iou_threshold"]

    def add_detector(self, detector: ElementDetector) -> None:
        """Add a detector to the composite.

        Args:
            detector: Element detector to add
        """
        self.detectors.append(detector)

    def _non_maximum_suppression(
        self, elements: List[DetectedElement]
    ) -> List[DetectedElement]:
        """Apply non-maximum suppression to remove overlapping detections.

        Args:
            elements: List of detected elements

        Returns:
            Filtered list with overlaps removed
        """
        if not elements:
            return []

        # Sort by confidence (descending)
        sorted_elements = sorted(elements, key=lambda e: e.confidence, reverse=True)

        keep = []
        while sorted_elements:
            # Keep the highest confidence element
            current = sorted_elements.pop(0)
            keep.append(current)

            # Remove overlapping elements
            sorted_elements = [
                e
                for e in sorted_elements
                if self._calculate_iou(current.bounds, e.bounds) < self.iou_threshold
            ]

        return keep

    def _calculate_iou(
        self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate Intersection over Union of two bounding boxes.

        Args:
            box1: First bounding box (x, y, width, height)
            box2: Second bounding box (x, y, width, height)

        Returns:
            IoU score between 0 and 1
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0


# Placeholder for future implementations
class TemplateDetector(ElementDetector):
    """Template matching based element detector.

    TODO: Implement template matching detection
    """

    def detect(self, screenshot: np.ndarray) -> List[DetectedElement]:
        """Detect elements using template matching."""
        raise NotImplementedError("Template detection not yet implemented")

    def configure(self, **kwargs) -> None:
        """Configure template detector."""
        pass


class FeatureDetector(ElementDetector):
    """Feature-based element detector.

    TODO: Implement feature-based detection (SIFT, ORB, etc.)
    """

    def detect(self, screenshot: np.ndarray) -> List[DetectedElement]:
        """Detect elements using feature detection."""
        raise NotImplementedError("Feature detection not yet implemented")

    def configure(self, **kwargs) -> None:
        """Configure feature detector."""
        pass


class OCRDetector(ElementDetector):
    """OCR-based text element detector.

    TODO: Implement OCR integration
    """

    def detect(self, screenshot: np.ndarray) -> List[DetectedElement]:
        """Detect text elements using OCR."""
        raise NotImplementedError("OCR detection not yet implemented")

    def configure(self, **kwargs) -> None:
        """Configure OCR detector."""
        pass
