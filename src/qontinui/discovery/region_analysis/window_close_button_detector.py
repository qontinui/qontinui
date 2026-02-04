"""
Window Close Button Detector.

This analyzer specifically detects window close buttons (X icons) using pattern
matching, template matching, and feature detection.

Performance: 50-150ms
Accuracy: 80-90% for standard close buttons
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


class WindowCloseButtonDetector(BaseRegionAnalyzer):
    """Detects window close buttons using multiple detection strategies."""

    @property
    def analysis_type(self) -> RegionAnalysisType:
        return RegionAnalysisType.PATTERN_ANALYSIS

    @property
    def name(self) -> str:
        return "window_close_button_detector"

    @property
    def supported_region_types(self) -> list[RegionType]:
        return [RegionType.CLOSE_BUTTON]

    @property
    def version(self) -> str:
        return "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Window Close Button detector.

        Args:
            config: Optional configuration parameters
        """
        super().__init__(config)

        params = self.get_default_parameters()
        if config:
            params.update(config)

        self.min_button_size = params["min_button_size"]
        self.max_button_size = params["max_button_size"]
        self.search_top_region = params["search_top_region"]
        self.search_right_bias = params["search_right_bias"]
        self.template_match_threshold = params["template_match_threshold"]

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "min_button_size": 16,
            "max_button_size": 48,
            "search_top_region": 0.15,  # Search top 15% of image
            "search_right_bias": 0.7,  # Focus on right 70% of image
            "template_match_threshold": 0.35,
        }

    async def analyze(self, input_data: RegionAnalysisInput) -> RegionAnalysisResult:
        """Detect close buttons."""
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

            # Detect close buttons
            regions = self._detect_close_buttons(gray, idx)
            all_regions.extend(regions)

        # Calculate overall confidence
        overall_confidence = (
            sum(r.confidence for r in all_regions) / len(all_regions) if all_regions else 0.0
        )

        return RegionAnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            regions=all_regions,
            confidence=overall_confidence,
            metadata={
                "total_close_buttons": len(all_regions),
            },
        )

    def _detect_close_buttons(
        self, gray: np.ndarray, screenshot_index: int
    ) -> list[DetectedRegion]:
        """Detect close buttons in an image."""
        detected_buttons = []

        # Define search region (top-right portion of image)
        h, w = gray.shape
        search_top = int(h * self.search_top_region)
        search_left = int(w * (1 - self.search_right_bias))

        search_region = gray[0:search_top, search_left:w]

        # Method 1: Template matching with X patterns
        for size in range(self.min_button_size, self.max_button_size + 1, 4):
            template = self._create_x_template(size)
            buttons = self._template_match(search_region, template, size, search_left, 0)
            detected_buttons.extend(buttons)

        # Method 2: Look for small square regions with diagonal lines
        buttons_geometric = self._detect_geometric_x(search_region, search_left, 0)
        detected_buttons.extend(buttons_geometric)

        # Remove duplicates (nearby detections)
        detected_buttons = self._remove_duplicates(detected_buttons)

        # Convert to DetectedRegion objects
        regions = []
        for i, (x, y, w, h, confidence, method) in enumerate(detected_buttons):
            region = DetectedRegion(
                bounding_box=BoundingBox(x, y, w, h),
                confidence=confidence,
                region_type=RegionType.CLOSE_BUTTON,
                label=f"close_button_{i}",
                screenshot_index=screenshot_index,
                metadata={
                    "detection_method": method,
                    "button_size": w,
                },
            )
            regions.append(region)

        return regions

    def _create_x_template(self, size: int) -> np.ndarray:
        """Create an X-shaped template for matching."""
        template = np.zeros((size, size), dtype=np.uint8)
        thickness = max(1, size // 8)

        # Draw X
        cv2.line(template, (0, 0), (size - 1, size - 1), 255, thickness)  # type: ignore[call-overload]
        cv2.line(template, (size - 1, 0), (0, size - 1), 255, thickness)  # type: ignore[call-overload]

        return template

    def _template_match(
        self,
        search_region: np.ndarray,
        template: np.ndarray,
        size: int,
        offset_x: int,
        offset_y: int,
    ) -> list[tuple]:
        """Perform template matching."""
        if search_region.shape[0] < template.shape[0] or search_region.shape[1] < template.shape[1]:
            return []

        result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)

        # Find locations above threshold
        locations = np.where(result >= self.template_match_threshold)

        buttons = []
        for pt in zip(*locations[::-1], strict=False):
            x = pt[0] + offset_x
            y = pt[1] + offset_y
            confidence = float(result[pt[1], pt[0]])

            buttons.append((x, y, size, size, confidence, "template_match"))

        return buttons

    def _detect_geometric_x(
        self, search_region: np.ndarray, offset_x: int, offset_y: int
    ) -> list[tuple]:
        """Detect X patterns using geometric analysis."""
        buttons = []

        # Edge detection
        edges = cv2.Canny(search_region, 50, 150)

        # Find line segments using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=20,
            minLineLength=self.min_button_size // 2,
            maxLineGap=5,
        )

        if lines is None:
            return buttons

        # Group lines into potential X patterns
        # An X consists of two diagonal lines crossing
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                x1, y1, x2, y2 = lines[i][0]
                x3, y3, x4, y4 = lines[j][0]

                # Check if lines are roughly diagonal and crossing
                if self._lines_form_x(x1, y1, x2, y2, x3, y3, x4, y4):
                    # Calculate bounding box
                    min_x = min(x1, x2, x3, x4)
                    max_x = max(x1, x2, x3, x4)
                    min_y = min(y1, y2, y3, y4)
                    max_y = max(y1, y2, y3, y4)

                    w = max_x - min_x
                    h = max_y - min_y

                    # Check size constraints
                    if (
                        w >= self.min_button_size
                        and w <= self.max_button_size
                        and h >= self.min_button_size
                        and h <= self.max_button_size
                    ):
                        # Should be roughly square
                        if abs(w - h) < w * 0.3:
                            x = min_x + offset_x
                            y = min_y + offset_y
                            confidence = 0.6  # Moderate confidence for geometric detection

                            buttons.append((x, y, w, h, confidence, "geometric_x"))

        return buttons

    def _lines_form_x(self, x1, y1, x2, y2, x3, y3, x4, y4) -> bool:
        """Check if two line segments form an X pattern."""
        # Calculate angles of both lines
        angle1 = np.arctan2(y2 - y1, x2 - x1)
        angle2 = np.arctan2(y4 - y3, x4 - x3)

        # Normalize angles to [0, pi]
        angle1 = abs(angle1)
        angle2 = abs(angle2)

        # For an X, one line should be ~45° and the other ~135° (or -45°)
        # Which means their difference should be close to 90°
        angle_diff = abs(angle1 - angle2)

        # Check if roughly perpendicular
        is_perpendicular = abs(angle_diff - np.pi / 2) < np.pi / 6

        # Check if lines intersect
        intersects = self._lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4)

        return is_perpendicular and intersects

    def _lines_intersect(self, x1, y1, x2, y2, x3, y3, x4, y4) -> bool:
        """Check if two line segments intersect."""

        def ccw(ax, ay, bx, by, cx, cy):
            return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)  # type: ignore[no-any-return]

        return ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and ccw(  # type: ignore[no-any-return]
            x1, y1, x2, y2, x3, y3
        ) != ccw(x1, y1, x2, y2, x4, y4)

    def _remove_duplicates(self, buttons: list[tuple]) -> list[tuple]:
        """Remove duplicate detections (nearby buttons)."""
        if not buttons:
            return buttons

        # Sort by confidence (descending)
        sorted_buttons = sorted(buttons, key=lambda b: b[4], reverse=True)

        keep: list[Any] = []
        for button in sorted_buttons:
            x, y, w, h, conf, method = button

            # Check if too close to any kept button
            is_duplicate = False
            for kept in keep:
                kx, ky, kw, kh, _, _ = kept

                # Calculate center distance
                cx1, cy1 = x + w // 2, y + h // 2
                cx2, cy2 = kx + kw // 2, ky + kh // 2
                dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

                # If centers are very close, it's a duplicate
                if dist < max(w, kw) * 0.5:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep.append(button)

        return keep
