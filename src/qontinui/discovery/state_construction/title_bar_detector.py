"""Title bar detection for UI analysis.

This module provides specialized title bar detection capabilities by identifying
elongated horizontal regions at the top of panels, often with distinct background
colors and containing text or window controls.
"""

import cv2
import numpy as np

from qontinui.discovery.state_construction.element_identifier import (
    IdentifiedRegion,
    RegionType,
)


class TitleBarDetector:
    """Detects window title bars in screenshots.

    Identifies title bars by looking for elongated horizontal regions
    at the top of panels, often with distinct background colors and
    containing text or window controls.
    """

    def __init__(
        self,
        title_bar_height_range: tuple[int, int] = (20, 60),
        max_color_uniformity: float = 40.0,
        min_aspect_ratio: float = 3.0,
    ):
        """Initialize the title bar detector.

        Args:
            title_bar_height_range: (min, max) height range for title bars
            max_color_uniformity: Maximum std deviation for uniform colors
            min_aspect_ratio: Minimum width/height ratio for title bars
        """
        self.title_bar_height_range = title_bar_height_range
        self.max_color_uniformity = max_color_uniformity
        self.min_aspect_ratio = min_aspect_ratio

    def detect_title_bars(self, screenshot: np.ndarray) -> list[IdentifiedRegion]:
        """Detect window title bars.

        Identifies title bars by looking for elongated horizontal regions
        at the top of panels, often with distinct background colors and
        containing text or window controls.

        Args:
            screenshot: Screenshot image as numpy array (BGR format)

        Returns:
            List of detected title bar regions
        """
        title_bars = []
        height, width = screenshot.shape[:2]

        # Scan top portion of screen
        scan_height = min(height // 3, 200)

        # Look for horizontal bands with distinct colors
        for y in range(0, scan_height, 5):
            for bar_height in range(
                self.title_bar_height_range[0], self.title_bar_height_range[1], 5
            ):
                if y + bar_height >= height:
                    break

                # Extract potential title bar region
                region = screenshot[y : y + bar_height, 0:width]

                # Analyze characteristics
                if self._has_title_bar_characteristics(region):
                    # Look for actual width (might not span full screen)
                    actual_bounds = self._find_actual_title_bar_bounds(screenshot, y, bar_height)

                    if actual_bounds:
                        x, y_pos, w, h = actual_bounds
                        title_bar = IdentifiedRegion(
                            region_type=RegionType.TITLE_BAR,
                            bounds=(x, y_pos, w, h),
                            confidence=0.8,
                            properties={"scanned_y": y, "height": bar_height},
                            sub_elements=[],
                        )
                        title_bars.append(title_bar)

        # Remove overlapping/duplicate title bars
        title_bars = self._remove_overlapping_regions(title_bars)

        return title_bars

    def _has_title_bar_characteristics(self, region: np.ndarray) -> bool:
        """Check if region has title bar characteristics.

        Title bars are elongated horizontally and often have uniform colors.

        Args:
            region: Region image to check

        Returns:
            True if likely a title bar
        """
        if region.size == 0:
            return False

        h, w = region.shape[:2]

        # Title bars are elongated horizontally
        if w < h * self.min_aspect_ratio:
            return False

        # Check for uniform color (typical for title bars)
        std_color = np.std(region, axis=(0, 1))
        color_uniformity = np.mean(std_color)

        # Title bars often have low color variance
        if color_uniformity > self.max_color_uniformity:
            return False

        return True

    def _find_actual_title_bar_bounds(
        self, screenshot: np.ndarray, y: int, height: int
    ) -> tuple[int, int, int, int] | None:
        """Find actual bounds of a title bar (may not span full width).

        Uses edge detection to find vertical boundaries of the title bar.

        Args:
            screenshot: Full screenshot
            y: Y position to search
            height: Expected height of the title bar

        Returns:
            Bounding box (x, y, w, h) or None if invalid
        """
        screen_height, screen_width = screenshot.shape[:2]

        if y + height > screen_height or y < 0:
            return None

        # Extract the horizontal strip at the title bar position
        title_strip = screenshot[y : y + height, :]

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(title_strip, cv2.COLOR_BGR2GRAY)

        # Detect vertical edges to find left and right boundaries
        edges = cv2.Canny(gray, 30, 100)

        # Sum edge pixels vertically to get a horizontal profile
        vertical_edge_profile = np.sum(edges, axis=0)

        # Smooth the profile to reduce noise
        kernel_size = max(3, screen_width // 200)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = cv2.GaussianBlur(
            vertical_edge_profile.astype(np.float32).reshape(1, -1),
            (kernel_size, 1),
            0,
        ).flatten()

        # Find significant edge positions (potential boundaries)
        threshold = np.mean(smoothed) + np.std(smoothed)
        edge_positions = np.where(smoothed > threshold)[0]

        if len(edge_positions) < 2:
            # No clear edges found, return full width
            return (0, y, screen_width, height)

        # Find leftmost and rightmost significant edges
        left_bound = int(edge_positions[0])
        right_bound = int(edge_positions[-1])

        # Ensure reasonable bounds
        min_width = screen_width // 4
        if right_bound - left_bound < min_width:
            # Too narrow, likely not a real title bar
            return (0, y, screen_width, height)

        # Add small padding to include the full title bar
        padding = 5
        left_bound = max(0, left_bound - padding)
        right_bound = min(screen_width, right_bound + padding)

        return (left_bound, y, right_bound - left_bound, height)

    def _remove_overlapping_regions(
        self, regions: list[IdentifiedRegion]
    ) -> list[IdentifiedRegion]:
        """Remove overlapping regions, keeping higher confidence ones.

        Args:
            regions: List of regions

        Returns:
            Filtered list without significant overlaps
        """
        if not regions:
            return []

        # Sort by confidence (descending)
        sorted_regions = sorted(regions, key=lambda r: r.confidence, reverse=True)

        keep: list[IdentifiedRegion] = []
        for region in sorted_regions:
            # Check overlap with kept regions
            overlaps = False
            for kept_region in keep:
                if self._calculate_region_overlap(region, kept_region) > 0.5:
                    overlaps = True
                    break

            if not overlaps:
                keep.append(region)

        return keep

    def _calculate_region_overlap(
        self, region1: IdentifiedRegion, region2: IdentifiedRegion
    ) -> float:
        """Calculate overlap between two regions.

        Args:
            region1: First region
            region2: Second region

        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        x1, y1, w1, h1 = region1.bounds
        x2, y2, w2, h2 = region2.bounds

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = w1 * h1 + w2 * h2 - intersection

        return intersection / union if union > 0 else 0.0
