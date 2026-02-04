"""Base class for detectors that analyze multiple screenshots.

This module provides the abstract base class for multi-screenshot detectors,
which analyze sequences of screenshots to find patterns, transitions, or
persistent UI elements across multiple frames.
"""

from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np


class MultiScreenshotDetector(ABC):
    """Base class for multi-screenshot detection algorithms.

    This abstract class defines the interface for detection algorithms that
    analyze multiple screenshots simultaneously. These detectors are useful for:
    - Finding persistent UI elements across states
    - Detecting state transitions
    - Identifying dynamic vs static regions
    - Analyzing temporal patterns

    Attributes:
        name: Human-readable name for this detector.

    Example:
        >>> class MyMultiDetector(MultiScreenshotDetector):
        ...     def detect_multi(self, screenshots, **params):
        ...         # Implementation here
        ...         return {0: [{"bbox": (x, y, w, h)}], 1: [...]}
        ...
        >>> detector = MyMultiDetector("my_multi_detector")
        >>> results = detector.detect_multi([img1, img2, img3], threshold=0.5)
    """

    def __init__(self, name: str) -> None:
        """Initialize the multi-screenshot detector.

        Args:
            name: Human-readable name for this detector.
        """
        self.name = name

    @abstractmethod
    def detect_multi(
        self, screenshots: list[np.ndarray[Any, Any]], **params: Any
    ) -> dict[int, list[dict[str, Any]]]:
        """Detect patterns across multiple screenshots.

        This is the main detection method that must be implemented by all
        concrete multi-screenshot detector classes. It analyzes a sequence
        of screenshots and returns detections organized by screenshot index.

        Args:
            screenshots: List of screenshots as numpy arrays (BGR or grayscale).
                All screenshots should have the same dimensions.
            **params: Algorithm-specific parameters. Common parameters include:
                - consistency_threshold: Required consistency across screenshots
                - min_frequency: Minimum frequency for detection
                - similarity_threshold: Threshold for matching regions

        Returns:
            Dictionary mapping screenshot index to list of detections.
            Each detection is a dictionary containing:
                - bbox: Bounding box as (x, y, w, h) tuple
                - confidence: Detection confidence score (0.0-1.0)
                - frequency: How often this appears (0.0-1.0)
                - screenshot_ids: List of screenshot indices where detected
                - Additional algorithm-specific metadata

        Example:
            >>> results = detector.detect_multi([img1, img2, img3])
            >>> for idx, detections in results.items():
            ...     print(f"Screenshot {idx}: {len(detections)} detections")
            ...     for detection in detections:
            ...         bbox = detection["bbox"]
            ...         freq = detection["frequency"]
            ...         print(f"  Region {bbox} appears in {freq*100:.1f}% of frames")
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
            ...         {"consistency_threshold": 0.8, "min_frequency": 0.5},
            ...         {"consistency_threshold": 0.9, "min_frequency": 0.7},
            ...     ]
        """
        return []

    @staticmethod
    def compute_consistency_map(
        screenshots: list[np.ndarray[Any, Any]], threshold: float = 0.95
    ) -> np.ndarray[Any, Any]:
        """Compute pixel-wise consistency map across screenshots.

        Calculates how consistent each pixel is across all screenshots.
        Useful for identifying static vs dynamic regions.

        Args:
            screenshots: List of screenshots (same dimensions).
            threshold: Similarity threshold for considering pixels consistent.

        Returns:
            Consistency map as float array (0.0-1.0) where 1.0 = always consistent.

        Example:
            >>> consistency = MultiScreenshotDetector.compute_consistency_map(
            ...     [img1, img2, img3], threshold=0.95
            ... )
            >>> static_mask = consistency > 0.9
            >>> print(f"Static pixels: {np.sum(static_mask)}")
        """
        if not screenshots:
            raise ValueError("Empty screenshot list provided")

        if len(screenshots) == 1:
            # Single screenshot: everything is consistent
            return np.ones(screenshots[0].shape[:2], dtype=np.float32)

        # Convert all to grayscale for comparison
        gray_screenshots = [MultiScreenshotDetector._to_grayscale(img) for img in screenshots]

        # Initialize consistency map
        height, width = gray_screenshots[0].shape
        consistency_map = np.zeros((height, width), dtype=np.float32)

        # Compare each screenshot with reference (first screenshot)
        reference = gray_screenshots[0].astype(np.float32)

        for screenshot in gray_screenshots[1:]:
            screenshot_float = screenshot.astype(np.float32)
            diff = np.abs(reference - screenshot_float)

            # Pixels are consistent if difference is below threshold
            consistent = diff < (255 * (1 - threshold))
            consistency_map += consistent.astype(np.float32)

        # Normalize by number of comparisons
        consistency_map /= len(screenshots) - 1

        return consistency_map

    @staticmethod
    def compute_stability_matrix(
        screenshots: list[np.ndarray[Any, Any]], variance_threshold: float = 10.0
    ) -> np.ndarray[Any, Any]:
        """Compute pixel stability matrix using variance analysis.

        Calculates variance for each pixel across all screenshots.
        Low variance indicates stable/static regions.

        Args:
            screenshots: List of screenshots (same dimensions).
            variance_threshold: Maximum variance for considering pixel stable.

        Returns:
            Binary stability matrix where 1 = stable, 0 = unstable.

        Example:
            >>> stability = MultiScreenshotDetector.compute_stability_matrix(
            ...     [img1, img2, img3], variance_threshold=10.0
            ... )
            >>> print(f"Stable pixels: {np.sum(stability)}")
        """
        if not screenshots:
            raise ValueError("Empty screenshot list provided")

        # Convert all to grayscale
        gray_screenshots = [MultiScreenshotDetector._to_grayscale(img) for img in screenshots]

        # Stack into 3D array: (num_screenshots, height, width)
        screenshot_stack = np.stack(gray_screenshots, axis=0).astype(np.float32)

        # Compute variance across screenshot axis
        variance = np.var(screenshot_stack, axis=0)

        # Create binary stability matrix
        stability = (variance <= variance_threshold).astype(np.uint8)

        return stability  # type: ignore[no-any-return]

    @staticmethod
    def find_persistent_regions(
        screenshots: list[np.ndarray[Any, Any]],
        min_frequency: float = 0.7,
        similarity_threshold: float = 0.95,
    ) -> list[dict[str, Any]]:
        """Find regions that appear persistently across screenshots.

        Identifies bounding boxes of regions that appear in a minimum
        fraction of the provided screenshots.

        Args:
            screenshots: List of screenshots (same dimensions).
            min_frequency: Minimum frequency for a region to be considered
                persistent (0.0-1.0).
            similarity_threshold: Threshold for matching regions across frames.

        Returns:
            List of persistent regions, each as a dictionary containing:
                - bbox: Bounding box as (x, y, w, h) tuple
                - frequency: Appearance frequency (0.0-1.0)
                - screenshot_indices: List of indices where region appears
                - representative_image: Representative image crop of the region

        Example:
            >>> regions = MultiScreenshotDetector.find_persistent_regions(
            ...     screenshots, min_frequency=0.8
            ... )
            >>> for region in regions:
            ...     bbox = region["bbox"]
            ...     freq = region["frequency"]
            ...     print(f"Region {bbox} appears {freq*100:.1f}% of the time")
        """
        if not screenshots:
            return []

        # Compute consistency map
        consistency = MultiScreenshotDetector.compute_consistency_map(
            screenshots, threshold=similarity_threshold
        )

        # Threshold to get stable regions
        stable_mask = (consistency >= min_frequency).astype(np.uint8)

        # Find connected components
        num_labels, labels = cv2.connectedComponents(stable_mask)

        persistent_regions = []

        for label_id in range(1, num_labels):  # Skip background (0)
            # Create mask for this component
            component_mask = (labels == label_id).astype(np.uint8)

            # Get bounding box
            coords = np.column_stack(np.where(component_mask > 0))
            if len(coords) == 0:
                continue

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            # Calculate actual frequency across screenshots
            region_frequency = consistency[component_mask > 0].mean()

            # Get representative image from first screenshot
            representative = screenshots[0][y_min : y_max + 1, x_min : x_max + 1].copy()

            # Find which screenshots contain this region
            screenshot_indices = []
            for idx, screenshot in enumerate(screenshots):
                roi = screenshot[y_min : y_max + 1, x_min : x_max + 1]
                if roi.shape == representative.shape:
                    diff = cv2.absdiff(roi, representative)
                    mean_diff = np.mean(diff)
                    if mean_diff < (255 * (1 - similarity_threshold)):
                        screenshot_indices.append(idx)

            persistent_regions.append(
                {
                    "bbox": (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1),
                    "frequency": float(region_frequency),
                    "screenshot_indices": screenshot_indices,
                    "representative_image": representative,
                    "pixel_count": int(np.sum(component_mask)),
                }
            )

        return persistent_regions

    @staticmethod
    def compute_transition_diff(
        before: np.ndarray[Any, Any],
        after: np.ndarray[Any, Any],
        threshold: float = 30.0,
    ) -> np.ndarray[Any, Any]:
        """Compute difference mask between two screenshots.

        Identifies regions that changed between two screenshots,
        useful for detecting state transitions.

        Args:
            before: Screenshot before transition.
            after: Screenshot after transition.
            threshold: Pixel difference threshold for considering a change.

        Returns:
            Binary mask where 1 = changed, 0 = unchanged.

        Example:
            >>> diff_mask = MultiScreenshotDetector.compute_transition_diff(
            ...     before_img, after_img, threshold=30.0
            ... )
            >>> changed_pixels = np.sum(diff_mask)
            >>> print(f"Changed pixels: {changed_pixels}")
        """
        # Convert to grayscale
        before_gray = MultiScreenshotDetector._to_grayscale(before)
        after_gray = MultiScreenshotDetector._to_grayscale(after)

        # Compute absolute difference
        diff = cv2.absdiff(before_gray, after_gray)

        # Threshold to get binary mask
        _, binary_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        return binary_mask

    @staticmethod
    def group_by_similarity(
        screenshots: list[np.ndarray[Any, Any]], similarity_threshold: float = 0.95
    ) -> dict[int, list[int]]:
        """Group screenshots by visual similarity.

        Clusters screenshots into groups where members are visually similar.
        Useful for identifying distinct application states.

        Args:
            screenshots: List of screenshots to group.
            similarity_threshold: Threshold for considering screenshots similar.

        Returns:
            Dictionary mapping group ID to list of screenshot indices.

        Example:
            >>> groups = MultiScreenshotDetector.group_by_similarity(
            ...     screenshots, similarity_threshold=0.9
            ... )
            >>> print(f"Found {len(groups)} distinct groups")
            >>> for group_id, indices in groups.items():
            ...     print(f"Group {group_id}: {len(indices)} screenshots")
        """
        if not screenshots:
            return {}

        # Convert all to grayscale for comparison
        gray_screenshots = [MultiScreenshotDetector._to_grayscale(img) for img in screenshots]

        groups: dict[int, list[int]] = {}
        assigned = [False] * len(screenshots)
        next_group_id = 0

        for i, img1 in enumerate(gray_screenshots):
            if assigned[i]:
                continue

            # Start new group
            group_id = next_group_id
            groups[group_id] = [i]
            assigned[i] = True
            next_group_id += 1

            # Find similar screenshots
            for j, img2 in enumerate(gray_screenshots[i + 1 :], start=i + 1):
                if assigned[j]:
                    continue

                # Compute similarity
                diff = cv2.absdiff(img1, img2)
                mean_diff = np.mean(diff)
                similarity = 1.0 - (mean_diff / 255.0)

                if similarity >= similarity_threshold:
                    groups[group_id].append(j)
                    assigned[j] = True

        return groups

    @staticmethod
    def _to_grayscale(image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
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
    def validate_screenshots(screenshots: list[np.ndarray[Any, Any]]) -> None:
        """Validate that all screenshots have the same dimensions.

        Args:
            screenshots: List of screenshots to validate.

        Raises:
            ValueError: If screenshots have inconsistent dimensions or list is empty.

        Example:
            >>> MultiScreenshotDetector.validate_screenshots([img1, img2, img3])
        """
        if not screenshots:
            raise ValueError("Empty screenshot list provided")

        reference_shape = screenshots[0].shape[:2]  # (height, width)

        for idx, screenshot in enumerate(screenshots[1:], start=1):
            if screenshot.shape[:2] != reference_shape:
                raise ValueError(
                    f"Screenshot {idx} has shape {screenshot.shape[:2]}, expected {reference_shape}"
                )

    def __repr__(self) -> str:
        """Return string representation of detector."""
        return f"{self.__class__.__name__}(name='{self.name}')"
