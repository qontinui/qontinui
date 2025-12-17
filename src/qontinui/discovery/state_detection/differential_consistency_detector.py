"""Differential Consistency Detector for state region identification.

This module implements the core Differential Consistency Detection algorithm,
which identifies state regions by finding pixels that change consistently together
across multiple transitions to the same state.

Key Insight:
    State boundaries have consistent change patterns across many examples, while
    dynamic backgrounds (like game animations) have inconsistent changes.

Example:
    Given 1000 transitions to a menu state:
    - Menu pixels: Always change from background → menu (high consistency)
    - Background pixels: Random animation changes (low consistency)
    - Result: Menu region is clearly identified with high confidence

Usage:
    >>> from qontinui.discovery.state_detection import DifferentialConsistencyDetector
    >>>
    >>> detector = DifferentialConsistencyDetector()
    >>>
    >>> # Provide before/after screenshot pairs (100-1000s recommended)
    >>> transitions = [
    ...     (before_img1, after_img1),
    ...     (before_img2, after_img2),
    ...     # ... more pairs
    ... ]
    >>>
    >>> regions = detector.detect_state_regions(
    ...     transitions,
    ...     consistency_threshold=0.7,
    ...     min_region_area=500
    ... )
    >>>
    >>> for region in regions:
    ...     print(f"Found region at {region.bbox}")
    ...     print(f"Consistency score: {region.consistency_score:.2f}")
"""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ..multi_screenshot_detector import MultiScreenshotDetector


@dataclass
class StateRegion:
    """A detected state region with consistency metrics.

    Attributes:
        bbox: Bounding box as (x, y, w, h) tuple
        consistency_score: Consistency score between 0.0 and 1.0
            Higher scores indicate more consistent changes
        example_diff: Representative difference image for this region
        pixel_count: Number of pixels in this region
    """

    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    consistency_score: float
    example_diff: np.ndarray[Any, Any]  # Representative difference image
    pixel_count: int = 0

    def __repr__(self) -> str:
        """String representation of state region."""
        return (
            f"StateRegion(bbox={self.bbox}, "
            f"score={self.consistency_score:.3f}, "
            f"pixels={self.pixel_count})"
        )


class DifferentialConsistencyDetector(MultiScreenshotDetector):
    """Detects state regions using differential consistency analysis.

    This detector analyzes before/after screenshot pairs to find regions
    that change consistently together. It's particularly effective for:
    - Modal dialogs and popup windows
    - Menu systems
    - UI overlays
    - State boundaries in dynamic environments

    The algorithm works by:
    1. Computing pixel-wise differences for all transition pairs
    2. Analyzing consistency (mean/std ratio) at each pixel location
    3. Extracting connected regions from the consistency map
    4. Ranking regions by their consistency scores

    Attributes:
        name: Name identifier for this detector
    """

    def __init__(self) -> None:
        """Initialize the differential consistency detector."""
        super().__init__("DifferentialConsistencyDetector")

    def detect_state_regions(
        self,
        transition_pairs: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]],
        consistency_threshold: float = 0.7,
        min_region_area: int = 500,
        **params: Any,
    ) -> list[StateRegion]:
        """Detect state regions from before/after screenshot pairs.

        Args:
            transition_pairs: List of (before, after) screenshot pairs.
                All pairs should represent transitions to the SAME state.
                More examples (100-1000) give better results.
            consistency_threshold: Minimum consistency score (0.0-1.0).
                Higher values are more strict. Recommended: 0.6-0.8
            min_region_area: Minimum pixel area for detected regions.
                Filters out noise. Recommended: 500-1000 for desktop apps.
            **params: Additional algorithm-specific parameters:
                - morphology_kernel_size: Size of morphology kernel (default: 5)
                - normalize_method: Normalization method ('minmax' or 'zscore')

        Returns:
            List of detected state regions sorted by consistency score (descending).

        Raises:
            ValueError: If fewer than 10 transition examples are provided.
            ValueError: If screenshots have inconsistent dimensions.

        Example:
            >>> detector = DifferentialConsistencyDetector()
            >>> regions = detector.detect_state_regions(
            ...     transition_pairs=[(before1, after1), (before2, after2), ...],
            ...     consistency_threshold=0.7,
            ...     min_region_area=500
            ... )
            >>> print(f"Found {len(regions)} state regions")
            >>> best_region = regions[0]
            >>> print(f"Best region: {best_region.bbox}, score: {best_region.consistency_score}")
        """
        if len(transition_pairs) < 10:
            raise ValueError(
                f"Need at least 10 transition examples, got {len(transition_pairs)}. "
                f"More examples (100-1000) give better results."
            )

        # Step 1: Compute difference images for all transitions
        diff_images = self._compute_differences(transition_pairs)

        # Step 2: Analyze consistency across all differences
        consistency_map = self._compute_consistency(
            diff_images, method=params.get("normalize_method", "minmax")
        )

        # Step 3: Extract regions from consistency map
        regions = self._extract_regions(
            consistency_map,
            consistency_threshold,
            min_region_area,
            kernel_size=params.get("morphology_kernel_size", 5),
        )

        # Step 4: Score and rank regions
        scored_regions = self._score_regions(regions, consistency_map, diff_images)

        return scored_regions

    def detect_multi(
        self, screenshots: list[np.ndarray[Any, Any]], **params: Any
    ) -> dict[int, list[dict[str, Any]]]:
        """Detect patterns across multiple screenshots.

        Note: This detector is designed for transition pairs, not sequential
        screenshots. This method creates consecutive pairs from the screenshot
        list and analyzes them.

        Args:
            screenshots: List of screenshots
            **params: Parameters passed to detect_state_regions

        Returns:
            Dictionary mapping screenshot index to detected regions
        """
        # Create consecutive pairs
        pairs = [
            (screenshots[i], screenshots[i + 1]) for i in range(len(screenshots) - 1)
        ]

        if len(pairs) < 10:
            raise ValueError(
                f"Need at least 10 screenshot pairs (11 screenshots), "
                f"got {len(pairs)} pairs"
            )

        # Detect regions
        regions = self.detect_state_regions(pairs, **params)

        # Map regions to screenshot indices
        # For each pair, the region appears in the 'after' screenshot
        result: dict[int, list[dict[str, Any]]] = {}
        for i in range(len(pairs)):
            screenshot_idx = i + 1  # The 'after' screenshot
            result[screenshot_idx] = [
                {
                    "bbox": region.bbox,
                    "confidence": region.consistency_score,
                    "type": "state_region",
                    "pixel_count": region.pixel_count,
                }
                for region in regions
            ]

        return result

    def _compute_differences(
        self, pairs: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]
    ) -> np.ndarray[Any, Any]:
        """Compute difference images for all transition pairs.

        Args:
            pairs: List of (before, after) screenshot pairs

        Returns:
            Array of shape (N, H, W) where N = number of pairs,
            containing absolute difference values as float32

        Raises:
            ValueError: If screenshot dimensions are inconsistent
        """
        diffs: list[np.ndarray[Any, Any]] = []

        reference_shape: tuple[int, int] | None = None

        for idx, (before, after) in enumerate(pairs):
            # Ensure same size
            if before.shape[:2] != after.shape[:2]:
                # Resize 'after' to match 'before'
                after = cv2.resize(after, (before.shape[1], before.shape[0]))

            # Check consistency across all pairs
            if reference_shape is None:
                reference_shape = before.shape[:2]  # type: ignore[assignment]
            elif before.shape[:2] != reference_shape:
                raise ValueError(
                    f"Inconsistent dimensions in pair {idx}: "
                    f"expected {reference_shape}, got {before.shape[:2]}"
                )

            # Convert to grayscale if needed
            if len(before.shape) == 3:
                before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
            else:
                before_gray = before

            if len(after.shape) == 3:
                after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
            else:
                after_gray = after

            # Compute absolute difference
            diff = cv2.absdiff(before_gray, after_gray).astype(np.float32)
            diffs.append(diff)

        return np.stack(diffs, axis=0)

    def _compute_consistency(
        self, diff_images: np.ndarray[Any, Any], method: str = "minmax"
    ) -> np.ndarray[Any, Any]:
        """Compute consistency score for each pixel across all differences.

        The consistency score measures how predictably a pixel changes.
        High consistency = pixel changes similarly across all examples
        Low consistency = pixel changes randomly

        Formula: consistency = mean_diff / (std_diff + epsilon)
        - High mean + low std = consistent change
        - Low mean or high std = inconsistent change

        Args:
            diff_images: Array of shape (N, H, W) with difference values
            method: Normalization method:
                - 'minmax': Normalize to [0, 1] range (default)
                - 'zscore': Z-score normalization then clip

        Returns:
            Consistency map (H, W) with scores 0.0-1.0
        """
        # Compute mean and std of differences at each pixel
        mean_diff = np.mean(diff_images, axis=0)
        std_diff = np.std(diff_images, axis=0)

        # Consistency score: high mean + low std = consistent change
        # Add small epsilon to avoid division by zero
        epsilon = 1.0
        consistency = mean_diff / (std_diff + epsilon)

        # Normalize to 0-1 range
        if method == "minmax":
            consistency = cv2.normalize(consistency, None, 0.0, 1.0, cv2.NORM_MINMAX)  # type: ignore[call-overload]
        elif method == "zscore":
            # Z-score normalization
            mean_val = np.mean(consistency)
            std_val = np.std(consistency)
            if std_val > 0:
                consistency = (consistency - mean_val) / std_val
                # Map to 0-1 using sigmoid-like function
                consistency = 1.0 / (1.0 + np.exp(-consistency))
            else:
                consistency = np.ones_like(consistency) * 0.5
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return consistency.astype(np.float32)  # type: ignore[no-any-return]

    def _extract_regions(
        self,
        consistency_map: np.ndarray[Any, Any],
        threshold: float,
        min_area: int,
        kernel_size: int = 5,
    ) -> list[tuple[int, int, int, int]]:
        """Extract connected regions from consistency map.

        Args:
            consistency_map: Consistency scores (H, W) in range [0, 1]
            threshold: Minimum consistency threshold
            min_area: Minimum pixel area for regions
            kernel_size: Size of morphology kernel for cleanup

        Returns:
            List of bounding boxes as (x, y, w, h) tuples
        """
        # Threshold consistency map
        mask = (consistency_map >= threshold).astype(np.uint8) * 255

        # Clean up with morphological operations
        # Close: Fill small holes
        # Open: Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # type: ignore[assignment]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # type: ignore[assignment]

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract bounding boxes
        bboxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area >= min_area:
                bboxes.append((x, y, w, h))

        return bboxes

    def _score_regions(
        self,
        bboxes: list[tuple[int, int, int, int]],
        consistency_map: np.ndarray[Any, Any],
        diff_images: np.ndarray[Any, Any],
    ) -> list[StateRegion]:
        """Score regions by average consistency within bounding box.

        Args:
            bboxes: List of bounding boxes (x, y, w, h)
            consistency_map: Pixel-wise consistency scores
            diff_images: Original difference images (N, H, W)

        Returns:
            List of StateRegion objects sorted by score (descending)
        """
        regions: list[StateRegion] = []

        for bbox in bboxes:
            x, y, w, h = bbox

            # Extract consistency values in this region
            region_consistency = consistency_map[y : y + h, x : x + w]
            avg_consistency = float(np.mean(region_consistency))

            # Get representative difference image (median across all transitions)
            region_diffs = diff_images[:, y : y + h, x : x + w]
            median_diff = np.median(region_diffs, axis=0).astype(np.uint8)

            # Calculate pixel count
            pixel_count = w * h

            regions.append(
                StateRegion(
                    bbox=bbox,
                    consistency_score=avg_consistency,
                    example_diff=median_diff,
                    pixel_count=pixel_count,
                )
            )

        # Sort by consistency score (highest first)
        regions.sort(key=lambda r: r.consistency_score, reverse=True)

        return regions

    def visualize_consistency(
        self,
        consistency_map: np.ndarray[Any, Any],
        regions: list[StateRegion],
        screenshot: np.ndarray[Any, Any],
        show_scores: bool = True,
    ) -> np.ndarray[Any, Any]:
        """Create visualization of detected regions overlaid on screenshot.

        Args:
            consistency_map: Consistency map (H, W) with values [0, 1]
            regions: List of detected state regions
            screenshot: Base screenshot for visualization
            show_scores: Whether to display consistency scores on regions

        Returns:
            Visualization image with heatmap and bounding boxes
        """
        # Create heatmap from consistency map
        heatmap = cv2.applyColorMap(
            (consistency_map * 255).astype(np.uint8), cv2.COLORMAP_JET
        )

        # Convert screenshot to BGR if grayscale
        if len(screenshot.shape) == 2:
            screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_GRAY2BGR)
        else:
            screenshot_bgr = screenshot.copy()

        # Resize heatmap if needed
        if heatmap.shape[:2] != screenshot_bgr.shape[:2]:
            heatmap = cv2.resize(
                heatmap, (screenshot_bgr.shape[1], screenshot_bgr.shape[0])
            )

        # Overlay heatmap on screenshot (60% screenshot, 40% heatmap)
        overlay = cv2.addWeighted(screenshot_bgr, 0.6, heatmap, 0.4, 0)

        # Draw region bounding boxes
        for idx, region in enumerate(regions):
            x, y, w, h = region.bbox

            # Color based on rank (green for best, yellow for others)
            if idx == 0:
                color = (0, 255, 0)  # Green for best region
                thickness = 3
            else:
                color = (0, 255, 255)  # Yellow for other regions
                thickness = 2

            # Draw rectangle
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)

            # Add consistency score label if requested
            if show_scores:
                label = f"{region.consistency_score:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

                # Draw label background
                label_y = max(y - 10, label_size[1] + 5)
                cv2.rectangle(
                    overlay,
                    (x, label_y - label_size[1] - 5),
                    (x + label_size[0] + 5, label_y + 5),
                    color,
                    -1,  # Filled
                )

                # Draw label text
                cv2.putText(
                    overlay,
                    label,
                    (x + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),  # Black text
                    2,
                )

        return overlay

    def compute_consistency_map(  # type: ignore[override]
        self,
        transition_pairs: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]],
        method: str = "minmax",
    ) -> np.ndarray[Any, Any]:
        """Compute consistency map from transition pairs.

        Utility method to get the consistency map without extracting regions.
        Useful for visualization and debugging.

        Args:
            transition_pairs: List of (before, after) screenshot pairs
            method: Normalization method ('minmax' or 'zscore')

        Returns:
            Consistency map (H, W) with values in [0, 1]

        Example:
            >>> detector = DifferentialConsistencyDetector()
            >>> consistency = detector.compute_consistency_map(pairs)
            >>> import matplotlib.pyplot as plt
            >>> plt.imshow(consistency, cmap='jet')
            >>> plt.colorbar()
            >>> plt.show()
        """
        if len(transition_pairs) < 2:
            raise ValueError("Need at least 2 transition pairs")

        diff_images = self._compute_differences(transition_pairs)
        consistency_map = self._compute_consistency(diff_images, method=method)

        return consistency_map

    def get_param_grid(self) -> list[dict[str, Any]]:
        """Return parameter grid for hyperparameter search.

        Returns:
            List of parameter configurations for tuning
        """
        return [
            {
                "consistency_threshold": 0.6,
                "min_region_area": 500,
                "morphology_kernel_size": 3,
            },
            {
                "consistency_threshold": 0.7,
                "min_region_area": 500,
                "morphology_kernel_size": 5,
            },
            {
                "consistency_threshold": 0.8,
                "min_region_area": 1000,
                "morphology_kernel_size": 5,
            },
            {
                "consistency_threshold": 0.7,
                "min_region_area": 500,
                "morphology_kernel_size": 7,
            },
        ]
