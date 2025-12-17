"""
Window Border Detector.

This analyzer detects window boundaries by looking for rectangular regions with
distinct borders. It identifies windows based on edge detection, contours, and
rectangular shapes.

Performance: 80-200ms
Accuracy: 75-85% for windows with clear borders
"""

from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image

from qontinui.discovery.region_analysis.base import (
    BaseRegionAnalyzer,
    BoundingBox,
    DetectedRegion,
    RegionAnalysisInput,
    RegionAnalysisResult,
    RegionAnalysisType,
    RegionType,
)


class WindowBorderDetector(BaseRegionAnalyzer):
    """Detects window boundaries using border and edge detection."""

    @property
    def analysis_type(self) -> RegionAnalysisType:
        return RegionAnalysisType.EDGE_DETECTION

    @property
    def name(self) -> str:
        return "window_border_detector"

    @property
    def supported_region_types(self) -> list[RegionType]:
        return [RegionType.WINDOW]

    @property
    def version(self) -> str:
        return "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Window Border detector.

        Args:
            config: Optional configuration parameters
        """
        super().__init__(config)

        params = self.get_default_parameters()
        if config:
            params.update(config)

        self.min_window_width = params["min_window_width"]
        self.min_window_height = params["min_window_height"]
        self.min_window_area = params["min_window_area"]
        self.max_window_area_ratio = params["max_window_area_ratio"]
        self.border_thickness_threshold = params["border_thickness_threshold"]
        self.rectangularity_threshold = params["rectangularity_threshold"]

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "min_window_width": 150,
            "min_window_height": 100,
            "min_window_area": 15000,
            "max_window_area_ratio": 0.8,  # Max 80% of image
            "border_thickness_threshold": 2,
            "rectangularity_threshold": 0.85,
        }

    async def analyze(self, input_data: RegionAnalysisInput) -> RegionAnalysisResult:
        """Detect window borders."""
        all_regions = []

        # Process each screenshot
        for idx, screenshot_bytes in enumerate(input_data.screenshot_data):
            # Convert bytes to numpy array
            image = Image.open(BytesIO(screenshot_bytes))
            image_np = np.array(image)

            # Convert to grayscale
            if len(image_np.shape) == 3:
                if image_np.shape[2] == 4:
                    gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
                else:
                    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np

            # Detect windows
            regions = self._detect_windows(gray, idx)
            all_regions.extend(regions)

        # Calculate overall confidence
        overall_confidence = (
            sum(r.confidence for r in all_regions) / len(all_regions)
            if all_regions
            else 0.0
        )

        return RegionAnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            regions=all_regions,
            confidence=overall_confidence,
            metadata={
                "min_window_area": self.min_window_area,
                "total_windows": len(all_regions),
            },
        )

    def _detect_windows(
        self, gray: np.ndarray, screenshot_index: int
    ) -> list[DetectedRegion]:
        """Detect windows in an image."""
        detected_regions: list[DetectedRegion] = []
        img_area = gray.shape[0] * gray.shape[1]

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges slightly to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, hierarchy = cv2.findContours(
            dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Analyze each contour
        for i, contour in enumerate(contours):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Filter by size
            if w < self.min_window_width or h < self.min_window_height:
                continue
            if area < self.min_window_area:
                continue
            if area > img_area * self.max_window_area_ratio:
                continue

            # Check rectangularity
            contour_area = cv2.contourArea(contour)
            rectangularity = contour_area / area if area > 0 else 0

            if rectangularity < self.rectangularity_threshold:
                continue

            # Check for border-like characteristics
            has_border = self._check_for_border(gray, x, y, w, h)
            if not has_border:
                continue

            # Check if this might be a child window (has parent in hierarchy)
            is_child = hierarchy is not None and hierarchy[0][i][3] != -1

            # Calculate confidence
            confidence = self._calculate_window_confidence(
                gray, x, y, w, h, rectangularity, is_child
            )

            window = DetectedRegion(
                bounding_box=BoundingBox(x, y, w, h),
                confidence=confidence,
                region_type=RegionType.WINDOW,
                label=f"window_{len(detected_regions)}",
                screenshot_index=screenshot_index,
                metadata={
                    "area": int(area),
                    "rectangularity": float(rectangularity),
                    "is_child_window": is_child,
                    "detection_method": "border_detection",
                },
            )
            detected_regions.append(window)

        # Remove overlapping windows (keep larger ones)
        detected_regions = self._remove_overlapping_windows(detected_regions)

        return detected_regions

    def _check_for_border(
        self, gray: np.ndarray, x: int, y: int, w: int, h: int
    ) -> bool:
        """Check if the region has a visible border."""
        # Sample pixels along the edges
        top_edge = gray[y : y + self.border_thickness_threshold, x : x + w]
        bottom_edge = gray[y + h - self.border_thickness_threshold : y + h, x : x + w]
        left_edge = gray[y : y + h, x : x + self.border_thickness_threshold]
        right_edge = gray[y : y + h, x + w - self.border_thickness_threshold : x + w]

        # Check if edges have different intensity than interior
        interior = gray[
            y
            + self.border_thickness_threshold * 2 : y
            + h
            - self.border_thickness_threshold * 2,
            x
            + self.border_thickness_threshold * 2 : x
            + w
            - self.border_thickness_threshold * 2,
        ]

        if interior.size == 0:
            return False

        edge_mean = np.mean(
            [
                np.mean(top_edge) if top_edge.size > 0 else 0,
                np.mean(bottom_edge) if bottom_edge.size > 0 else 0,
                np.mean(left_edge) if left_edge.size > 0 else 0,
                np.mean(right_edge) if right_edge.size > 0 else 0,
            ]
        )

        interior_mean = np.mean(interior)

        # Borders are typically darker or lighter than content
        intensity_diff = abs(edge_mean - interior_mean)

        return intensity_diff > 10  # type: ignore[no-any-return]  # Threshold for border detection

    def _calculate_window_confidence(
        self,
        gray: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        rectangularity: float,
        is_child: bool,
    ) -> float:
        """Calculate confidence that this is a window."""
        confidence = 0.5  # Base confidence

        # Higher rectangularity = higher confidence
        rect_score = rectangularity
        confidence += rect_score * 0.2

        # Windows typically have title bars at the top
        has_title_bar_like_region = self._check_title_bar_region(gray, x, y, w, h)
        if has_title_bar_like_region:
            confidence += 0.2

        # Child windows are more likely to be actual windows
        if is_child:
            confidence += 0.1

        return min(confidence, 1.0)

    def _check_title_bar_region(
        self, gray: np.ndarray, x: int, y: int, w: int, h: int
    ) -> bool:
        """Check if there's a title-bar-like region at the top."""
        # Check top 10% of window for horizontal uniformity
        title_height = min(40, int(h * 0.1))
        if title_height < 15:
            return False

        title_region = gray[y : y + title_height, x : x + w]

        # Calculate horizontal consistency
        row_means = np.mean(title_region, axis=1)
        row_std = np.std(row_means)

        # Title bars typically have uniform rows
        return row_std < 15  # type: ignore[no-any-return]

    def _remove_overlapping_windows(
        self, windows: list[DetectedRegion]
    ) -> list[DetectedRegion]:
        """Remove overlapping windows, keeping larger ones."""
        if len(windows) <= 1:
            return windows

        # Sort by area (largest first)
        sorted_windows = sorted(
            windows,
            key=lambda w: w.bounding_box.width * w.bounding_box.height,
            reverse=True,
        )

        keep: list[Any] = []
        for window in sorted_windows:
            # Check if it overlaps significantly with any kept window
            overlaps = False
            for kept in keep:
                iou = self._calculate_iou(window.bounding_box, kept.bounding_box)
                if iou > 0.5:  # Significant overlap
                    overlaps = True
                    break

            if not overlaps:
                keep.append(window)

        return keep

    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        # Calculate intersection
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        box1_area = box1.width * box1.height
        box2_area = box2.width * box2.height
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0.0
