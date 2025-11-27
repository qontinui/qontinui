"""
Base test cases for element detection components.

This module provides base test classes and common test utilities for testing
element detection functionality across different detector implementations.

Example usage:
    >>> import pytest
    >>> from tests.discovery.element_detection.test_base import BaseDetectorTest
    >>>
    >>> class TestMyDetector(BaseDetectorTest):
    ...     @pytest.fixture
    ...     def detector(self):
    ...         return MyDetector()
    ...
    ...     def test_detection_accuracy(self, detector, screenshot_with_elements):
    ...         # Test implementation
    ...         pass
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pytest

from tests.fixtures.screenshot_fixtures import ElementSpec, SyntheticScreenshotGenerator


class BaseDetectorTest(ABC):
    """
    Abstract base class for detector test cases.

    Provides common test methods and utilities for testing element detectors.
    Subclass this to test specific detector implementations.

    Example:
        >>> class TestButtonDetector(BaseDetectorTest):
        ...     @pytest.fixture
        ...     def detector(self):
        ...         return ButtonDetector()
        ...
        ...     def test_button_detection(self, detector, synthetic_screenshot):
        ...         screenshot = synthetic_screenshot(
        ...             elements=[ElementSpec("button", x=100, y=100)]
        ...         )
        ...         results = detector.detect(screenshot)
        ...         assert len(results) == 1
    """

    @pytest.fixture
    @abstractmethod
    def detector(self):
        """
        Fixture that provides the detector instance to test.

        Must be implemented by subclasses.

        Returns:
            Detector instance to test
        """
        pass

    def test_detector_initialization(self, detector):
        """
        Test that detector initializes correctly.

        Args:
            detector: Detector instance from fixture
        """
        assert detector is not None
        assert hasattr(detector, "detect") or hasattr(detector, "detect_elements")

    def test_empty_screenshot(self, detector):
        """
        Test detector behavior on empty screenshot.

        Args:
            detector: Detector instance from fixture
        """
        generator = SyntheticScreenshotGenerator()
        empty_screen = generator.generate(width=800, height=600, elements=[])

        if hasattr(detector, "detect"):
            results = detector.detect(empty_screen)
        else:
            results = detector.detect_elements(empty_screen)

        # Should return empty list or list with no high-confidence detections
        assert isinstance(results, list)
        high_confidence = [r for r in results if getattr(r, "confidence", 1.0) > 0.8]
        assert len(high_confidence) == 0

    def test_single_element_detection(self, detector):
        """
        Test detection of a single element.

        Args:
            detector: Detector instance from fixture
        """
        generator = SyntheticScreenshotGenerator()
        screenshot = generator.generate(
            width=800,
            height=600,
            elements=[ElementSpec("button", x=100, y=100, width=120, height=40, text="Submit")],
        )

        if hasattr(detector, "detect"):
            results = detector.detect(screenshot)
        else:
            results = detector.detect_elements(screenshot)

        assert isinstance(results, list)
        # Should detect at least one element
        assert len(results) >= 1

    def test_multiple_elements_detection(self, detector):
        """
        Test detection of multiple elements.

        Args:
            detector: Detector instance from fixture
        """
        generator = SyntheticScreenshotGenerator()
        screenshot, elements = generator.generate_with_known_elements(
            width=1024, height=768, num_buttons=3, num_inputs=2, num_icons=2
        )

        if hasattr(detector, "detect"):
            results = detector.detect(screenshot)
        else:
            results = detector.detect_elements(screenshot)

        assert isinstance(results, list)
        # Should detect at least some elements (not necessarily all)
        assert len(results) >= 1

    def test_detection_result_format(self, detector):
        """
        Test that detection results have expected format.

        Args:
            detector: Detector instance from fixture
        """
        generator = SyntheticScreenshotGenerator()
        screenshot = generator.generate(
            width=800, height=600, elements=[ElementSpec("button", x=100, y=100, text="Test")]
        )

        if hasattr(detector, "detect"):
            results = detector.detect(screenshot)
        else:
            results = detector.detect_elements(screenshot)

        if len(results) > 0:
            result = results[0]
            # Check that result has required attributes
            assert hasattr(result, "bbox") or (
                hasattr(result, "x")
                and hasattr(result, "y")
                and hasattr(result, "width")
                and hasattr(result, "height")
            )


class BaseElementTypeTest:
    """
    Base test class for testing detection of specific element types.

    Provides test methods focused on detecting and validating specific
    UI element types (buttons, inputs, icons, etc.).

    Example:
        >>> class TestButtonDetection(BaseElementTypeTest):
        ...     element_type = "button"
        ...
        ...     @pytest.fixture
        ...     def detector(self):
        ...         return ButtonDetector()
    """

    element_type: str = "unknown"

    @pytest.fixture
    @abstractmethod
    def detector(self):
        """Fixture providing the detector instance."""
        pass

    def test_detect_single_element(self, detector):
        """Test detection of single element of specified type."""
        generator = SyntheticScreenshotGenerator()
        screenshot = generator.generate(
            width=800,
            height=600,
            elements=[
                ElementSpec(
                    self.element_type,
                    x=100,
                    y=100,
                    width=120,
                    height=40,
                    text=f"Test {self.element_type}",
                )
            ],
        )

        if hasattr(detector, "detect"):
            results = detector.detect(screenshot)
        else:
            results = detector.detect_elements(screenshot)

        # Should detect the element
        assert len(results) >= 1

    def test_detect_multiple_same_type(self, detector):
        """Test detection of multiple elements of same type."""
        generator = SyntheticScreenshotGenerator()
        elements = [
            ElementSpec(self.element_type, x=100 + i * 150, y=100, width=120, height=40)
            for i in range(3)
        ]

        screenshot = generator.generate(width=800, height=600, elements=elements)

        if hasattr(detector, "detect"):
            results = detector.detect(screenshot)
        else:
            results = detector.detect_elements(screenshot)

        # Should detect multiple elements
        assert len(results) >= 2

    def test_detect_with_varying_sizes(self, detector):
        """Test detection with elements of varying sizes."""
        generator = SyntheticScreenshotGenerator()
        elements = [
            ElementSpec(self.element_type, x=50, y=50, width=80, height=30),
            ElementSpec(self.element_type, x=200, y=50, width=150, height=60),
            ElementSpec(self.element_type, x=400, y=50, width=100, height=40),
        ]

        screenshot = generator.generate(width=800, height=600, elements=elements)

        if hasattr(detector, "detect"):
            results = detector.detect(screenshot)
        else:
            results = detector.detect_elements(screenshot)

        # Should detect elements of different sizes
        assert len(results) >= 1


class DetectionPerformanceTest:
    """
    Base class for performance testing of detectors.

    Provides utilities for measuring detection speed, accuracy, and efficiency.

    Example:
        >>> class TestDetectorPerformance(DetectionPerformanceTest):
        ...     @pytest.fixture
        ...     def detector(self):
        ...         return MyDetector()
        ...
        ...     def test_detection_speed(self, detector):
        ...         self.measure_detection_time(detector, num_runs=10)
    """

    @pytest.fixture
    @abstractmethod
    def detector(self):
        """Fixture providing the detector instance."""
        pass

    def measure_detection_time(
        self, detector, screenshot: np.ndarray | None = None, num_runs: int = 10
    ) -> float:
        """
        Measure average detection time.

        Args:
            detector: Detector instance
            screenshot: Screenshot to test with (generates one if None)
            num_runs: Number of detection runs to average

        Returns:
            Average detection time in seconds
        """
        import time

        if screenshot is None:
            generator = SyntheticScreenshotGenerator()
            screenshot, _ = generator.generate_with_known_elements()

        times = []
        for _ in range(num_runs):
            start = time.time()
            if hasattr(detector, "detect"):
                detector.detect(screenshot)
            else:
                detector.detect_elements(screenshot)
            times.append(time.time() - start)

        return sum(times) / len(times)

    def test_detection_speed_reasonable(self, detector):
        """
        Test that detection completes in reasonable time.

        Args:
            detector: Detector instance from fixture
        """
        avg_time = self.measure_detection_time(detector, num_runs=5)
        # Should complete in less than 5 seconds per detection
        assert avg_time < 5.0, f"Detection too slow: {avg_time:.2f}s"

    def test_consistent_results(self, detector):
        """
        Test that detector produces consistent results on same input.

        Args:
            detector: Detector instance from fixture
        """
        generator = SyntheticScreenshotGenerator()
        screenshot = generator.generate(
            width=800,
            height=600,
            elements=[ElementSpec("button", x=100, y=100, width=120, height=40)],
        )

        # Run detection multiple times
        results_list = []
        for _ in range(3):
            if hasattr(detector, "detect"):
                results = detector.detect(screenshot)
            else:
                results = detector.detect_elements(screenshot)
            results_list.append(results)

        # Results should be consistent (same number of detections)
        result_counts = [len(r) for r in results_list]
        assert len(set(result_counts)) <= 2, "Results vary significantly across runs"


# Utility functions for testing


def assert_bbox_valid(bbox: tuple[int, int, int, int], max_width: int, max_height: int):
    """
    Assert that a bounding box is valid.

    Args:
        bbox: Bounding box as (x, y, width, height)
        max_width: Maximum valid width
        max_height: Maximum valid height

    Raises:
        AssertionError: If bbox is invalid
    """
    x, y, w, h = bbox
    assert x >= 0, f"Negative x coordinate: {x}"
    assert y >= 0, f"Negative y coordinate: {y}"
    assert w > 0, f"Non-positive width: {w}"
    assert h > 0, f"Non-positive height: {h}"
    assert x + w <= max_width, f"Bbox extends beyond width: {x + w} > {max_width}"
    assert y + h <= max_height, f"Bbox extends beyond height: {y + h} > {max_height}"


def assert_detection_confidence_valid(confidence: float):
    """
    Assert that detection confidence is valid.

    Args:
        confidence: Confidence score

    Raises:
        AssertionError: If confidence is invalid
    """
    assert 0.0 <= confidence <= 1.0, f"Invalid confidence: {confidence}"


def calculate_iou(bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: First bbox as (x, y, width, height)
        bbox2: Second bbox as (x, y, width, height)

    Returns:
        IoU score between 0 and 1

    Example:
        >>> bbox1 = (0, 0, 100, 100)
        >>> bbox2 = (50, 50, 100, 100)
        >>> iou = calculate_iou(bbox1, bbox2)
        >>> assert 0 < iou < 1
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Convert to corner coordinates
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    # Calculate intersection
    x_overlap = max(0, min(x1_max, x2_max) - max(x1, x2))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1, y2))
    intersection = x_overlap * y_overlap

    # Calculate union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def find_matching_detection(
    ground_truth_bbox: tuple[int, int, int, int], detections: list[Any], iou_threshold: float = 0.5
) -> Any | None:
    """
    Find detection that matches ground truth bbox.

    Args:
        ground_truth_bbox: Expected bbox as (x, y, width, height)
        detections: List of detection results
        iou_threshold: Minimum IoU to consider a match

    Returns:
        Matching detection or None

    Example:
        >>> detections = [
        ...     MockDetectionResult("button", 0.9, (100, 100, 80, 40)),
        ...     MockDetectionResult("button", 0.8, (300, 100, 80, 40)),
        ... ]
        >>> match = find_matching_detection((95, 95, 85, 45), detections)
        >>> assert match is not None
    """
    best_iou = 0.0
    best_match = None

    for detection in detections:
        if hasattr(detection, "bbox"):
            det_bbox = detection.bbox
        else:
            det_bbox = (detection.x, detection.y, detection.width, detection.height)

        iou = calculate_iou(ground_truth_bbox, det_bbox)

        if iou > best_iou and iou >= iou_threshold:
            best_iou = iou
            best_match = detection

    return best_match
