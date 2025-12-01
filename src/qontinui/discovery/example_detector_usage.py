"""Example usage of BaseDetector and MultiScreenshotDetector base classes.

This file demonstrates how to create concrete detector implementations
by inheriting from the base classes.
"""

from typing import Any

import cv2
import numpy as np

from .base_detector import BaseDetector
from .multi_screenshot_detector import MultiScreenshotDetector


class ExampleButtonDetector(BaseDetector):
    """Example detector that finds button-like rectangular regions.

    This is a simple example showing how to implement BaseDetector.
    """

    def __init__(self) -> None:
        """Initialize the button detector."""
        super().__init__("example_button_detector")

    def detect(
        self, image: np.ndarray[Any, Any], **params: Any
    ) -> list[dict[str, Any]]:
        """Detect button-like regions in the image.

        Args:
            image: Input image as numpy array.
            **params: Optional parameters:
                - min_size: Minimum button size as (width, height)
                - max_size: Maximum button size as (width, height)
                - threshold: Edge detection threshold

        Returns:
            List of detected button regions.
        """
        # Extract parameters with defaults
        min_size = params.get("min_size", (50, 20))
        max_size = params.get("max_size", (400, 80))
        threshold = params.get("threshold", 50)

        # Convert to grayscale
        gray = self.convert_to_grayscale(image)

        # Detect edges
        edges = cv2.Canny(gray, threshold, threshold * 2)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []

        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Check if it looks like a button (rectangular, reasonable size)
            aspect_ratio = w / h if h > 0 else 0
            if 1.5 < aspect_ratio < 8.0:  # Buttons are typically wider than tall
                # Check size constraints
                if min_size[0] <= w <= max_size[0] and min_size[1] <= h <= max_size[1]:

                    detections.append(
                        {
                            "bbox": (x, y, w, h),
                            "confidence": 0.8,  # Placeholder confidence
                            "type": "button",
                            "aspect_ratio": aspect_ratio,
                        }
                    )

        # Apply post-processing utilities
        boxes = [d["bbox"] for d in detections]
        merged_boxes = self.merge_overlapping_boxes(boxes, overlap_threshold=0.5)  # type: ignore[arg-type]
        filtered_boxes = self.remove_contained_boxes(merged_boxes)

        # Rebuild detections with filtered boxes
        return [
            {"bbox": box, "confidence": 0.8, "type": "button"} for box in filtered_boxes
        ]

    def get_param_grid(self) -> list[dict[str, Any]]:
        """Return parameter configurations for hyperparameter tuning."""
        return [
            {"min_size": (50, 20), "threshold": 30},
            {"min_size": (50, 20), "threshold": 50},
            {"min_size": (50, 20), "threshold": 70},
            {"min_size": (30, 15), "threshold": 50},
        ]


class ExampleConsistencyDetector(MultiScreenshotDetector):
    """Example detector that finds regions consistent across screenshots.

    This is a simple example showing how to implement MultiScreenshotDetector.
    """

    def __init__(self) -> None:
        """Initialize the consistency detector."""
        super().__init__("example_consistency_detector")

    def detect_multi(
        self, screenshots: list[np.ndarray[Any, Any]], **params: Any
    ) -> dict[int, list[dict[str, Any]]]:
        """Detect consistent regions across multiple screenshots.

        Args:
            screenshots: List of screenshot images.
            **params: Optional parameters:
                - consistency_threshold: Threshold for pixel consistency
                - min_frequency: Minimum appearance frequency
                - min_size: Minimum region size

        Returns:
            Dictionary mapping screenshot index to detected regions.
        """
        # Validate input
        self.validate_screenshots(screenshots)

        # Extract parameters with defaults
        consistency_threshold = params.get("consistency_threshold", 0.95)
        min_frequency = params.get("min_frequency", 0.7)
        min_size = params.get("min_size", (20, 20))

        # Find persistent regions across all screenshots
        persistent_regions = self.find_persistent_regions(
            screenshots,
            min_frequency=min_frequency,
            similarity_threshold=consistency_threshold,
        )

        # Filter by size
        filtered_regions = [
            region
            for region in persistent_regions
            if (region["bbox"][2] >= min_size[0] and region["bbox"][3] >= min_size[1])
        ]

        # Build result dictionary: map each screenshot to its detections
        result: dict[int, list[dict[str, Any]]] = {}

        for region in filtered_regions:
            bbox = region["bbox"]
            frequency = region["frequency"]
            screenshot_indices = region["screenshot_indices"]

            # Add this detection to each screenshot where it appears
            for idx in screenshot_indices:
                if idx not in result:
                    result[idx] = []

                result[idx].append(
                    {
                        "bbox": bbox,
                        "confidence": frequency,
                        "frequency": frequency,
                        "type": "persistent_region",
                        "screenshot_count": len(screenshot_indices),
                    }
                )

        return result

    def get_param_grid(self) -> list[dict[str, Any]]:
        """Return parameter configurations for hyperparameter tuning."""
        return [
            {"consistency_threshold": 0.90, "min_frequency": 0.6},
            {"consistency_threshold": 0.95, "min_frequency": 0.7},
            {"consistency_threshold": 0.98, "min_frequency": 0.8},
        ]


def example_single_image_detection() -> None:
    """Example of using a single-image detector."""
    # Create a sample image (black background with white rectangle)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (300, 150), (255, 255, 255), -1)

    # Create detector and run detection
    detector = ExampleButtonDetector()
    detections = detector.detect(image, min_size=(50, 20), threshold=50)

    print(f"Single-image detection with {detector.name}:")
    print(f"  Found {len(detections)} button(s)")
    for detection in detections:
        bbox = detection["bbox"]
        print(f"    - Button at {bbox} with confidence {detection['confidence']}")


def example_multi_screenshot_detection() -> None:
    """Example of using a multi-screenshot detector."""
    # Create sample screenshots with a persistent region
    screenshots = []
    for _i in range(5):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Add a persistent white rectangle in all screenshots
        cv2.rectangle(img, (50, 50), (150, 100), (255, 255, 255), -1)
        screenshots.append(img)

    # Create detector and run detection
    detector = ExampleConsistencyDetector()
    results = detector.detect_multi(
        screenshots, consistency_threshold=0.95, min_frequency=0.7
    )

    print(f"\nMulti-screenshot detection with {detector.name}:")
    print(f"  Analyzed {len(screenshots)} screenshots")
    print(f"  Found detections in {len(results)} screenshots")
    for idx, detections in results.items():
        print(f"    Screenshot {idx}: {len(detections)} detection(s)")
        for detection in detections:
            bbox = detection["bbox"]
            freq = detection["frequency"]
            print(f"      - Region at {bbox}, frequency={freq:.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print("BaseDetector and MultiScreenshotDetector Usage Examples")
    print("=" * 60)

    example_single_image_detection()
    example_multi_screenshot_detection()

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
