"""Grid pattern detection for UI analysis.

This module provides specialized grid detection capabilities using Hough line
detection and pattern analysis to identify grid structures like inventory slots,
skill bars, or tile layouts.
"""

import cv2
import numpy as np

from qontinui.discovery.state_construction.element_identifier import IdentifiedRegion, RegionType


class GridDetector:
    """Detects grid patterns in screenshots using computer vision.

    Uses Hough line detection to identify horizontal and vertical lines,
    clusters parallel lines, and extracts grid regions with calculated
    properties like rows, columns, and regularity.
    """

    def __init__(
        self,
        line_threshold: int = 100,
        grid_cell_min_size: int = 10,
        grid_min_cells: int = 4,
        min_region_size: tuple[int, int] = (20, 20),
        max_region_size: tuple[int, int] = (2000, 2000),
    ):
        """Initialize the grid detector.

        Args:
            line_threshold: Hough transform threshold for line detection
            grid_cell_min_size: Minimum cell size in pixels
            grid_min_cells: Minimum number of cells to constitute a grid
            min_region_size: Minimum region size (width, height)
            max_region_size: Maximum region size (width, height)
        """
        self.line_threshold = line_threshold
        self.grid_cell_min_size = grid_cell_min_size
        self.grid_min_cells = grid_min_cells
        self.min_region_size = min_region_size
        self.max_region_size = max_region_size

    def detect_grid_regions(self, screenshot: np.ndarray) -> list[IdentifiedRegion]:
        """Detect grid patterns in the screenshot.

        Uses Hough line detection and pattern analysis to identify
        grid structures like inventory slots, skill bars, or tile layouts.

        Args:
            screenshot: Screenshot image as numpy array (BGR format)

        Returns:
            List of detected grid regions
        """
        grid_regions = []
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.line_threshold,
            minLineLength=30,
            maxLineGap=10,
        )

        if lines is None:
            return grid_regions

        # Separate horizontal and vertical lines
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

            if angle < 10 or angle > 170:  # Horizontal
                h_lines.append((y1, x1, x2))  # (y, x_start, x_end)
            elif 80 < angle < 100:  # Vertical
                v_lines.append((x1, y1, y2))  # (x, y_start, y_end)

        # Cluster parallel lines
        h_clusters = self._cluster_parallel_lines(h_lines)
        v_clusters = self._cluster_parallel_lines(v_lines)

        # Find grid patterns where lines intersect regularly
        if len(h_clusters) >= 2 and len(v_clusters) >= 2:
            for h_group in h_clusters:
                for v_group in v_clusters:
                    if len(h_group) >= 2 and len(v_group) >= 2:
                        grid = self._extract_grid_region(h_group, v_group, screenshot)
                        if grid:
                            grid_regions.append(grid)

        return grid_regions

    def _cluster_parallel_lines(self, lines: list[tuple], tolerance: int = 10) -> list[list[tuple]]:
        """Cluster parallel lines that are close together.

        Args:
            lines: List of line coordinates
            tolerance: Distance tolerance for clustering in pixels

        Returns:
            List of line clusters
        """
        if not lines:
            return []

        # Sort lines by position
        sorted_lines = sorted(lines)
        clusters = [[sorted_lines[0]]]

        for line in sorted_lines[1:]:
            # Check if close to last cluster
            if abs(line[0] - clusters[-1][-1][0]) <= tolerance:
                clusters[-1].append(line)
            else:
                clusters.append([line])

        # Filter clusters with at least 2 lines
        return [cluster for cluster in clusters if len(cluster) >= 2]

    def _extract_grid_region(
        self,
        h_lines: list[tuple],
        v_lines: list[tuple],
        screenshot: np.ndarray,
    ) -> IdentifiedRegion | None:
        """Extract a grid region from horizontal and vertical line clusters.

        Args:
            h_lines: Horizontal lines as (y, x_start, x_end) tuples
            v_lines: Vertical lines as (x, y_start, y_end) tuples
            screenshot: Full screenshot image

        Returns:
            IdentifiedRegion if valid grid found, None otherwise
        """
        # Calculate bounding box
        h_positions = [line[0] for line in h_lines]
        v_positions = [line[0] for line in v_lines]

        y_min, y_max = min(h_positions), max(h_positions)
        x_min, x_max = min(v_positions), max(v_positions)

        width = x_max - x_min
        height = y_max - y_min

        # Validate grid size
        if not self._is_valid_size((width, height)):
            return None

        # Calculate grid properties
        rows = len(h_lines) - 1
        cols = len(v_lines) - 1

        if rows < 1 or cols < 1:
            return None

        cell_width = width / cols if cols > 0 else 0
        cell_height = height / rows if rows > 0 else 0

        # Cells should be reasonable size
        if cell_width < self.grid_cell_min_size or cell_height < self.grid_cell_min_size:
            return None

        properties = {
            "rows": rows,
            "cols": cols,
            "cell_width": float(cell_width),
            "cell_height": float(cell_height),
            "regularity": self._calculate_grid_regularity(h_lines, v_lines),
        }

        return IdentifiedRegion(
            region_type=RegionType.GRID,
            bounds=(x_min, y_min, width, height),
            confidence=0.85,
            properties=properties,
            sub_elements=[],
        )

    def _calculate_grid_regularity(self, h_lines: list[tuple], v_lines: list[tuple]) -> float:
        """Calculate how regular/uniform a grid pattern is.

        Measures spacing consistency using coefficient of variation.

        Args:
            h_lines: Horizontal lines
            v_lines: Vertical lines

        Returns:
            Regularity score (0.0 to 1.0, higher is more regular)
        """
        # Check spacing consistency
        h_spacings = [h_lines[i + 1][0] - h_lines[i][0] for i in range(len(h_lines) - 1)]
        v_spacings = [v_lines[i + 1][0] - v_lines[i][0] for i in range(len(v_lines) - 1)]

        if not h_spacings or not v_spacings:
            return 0.0

        # Lower variance = more regular
        h_std = np.std(h_spacings) if len(h_spacings) > 1 else 0
        v_std = np.std(v_spacings) if len(v_spacings) > 1 else 0
        h_mean = np.mean(h_spacings)
        v_mean = np.mean(v_spacings)

        if h_mean == 0 or v_mean == 0:
            return 0.0

        # Coefficient of variation (lower is better)
        h_cv = h_std / h_mean
        v_cv = v_std / v_mean

        # Convert to regularity score (inverse of CV, clamped)
        avg_cv = (h_cv + v_cv) / 2
        regularity = max(0.0, 1.0 - avg_cv)

        return float(regularity)

    def _is_valid_size(self, size: tuple[int, int]) -> bool:
        """Check if size is within valid range.

        Args:
            size: (width, height) tuple

        Returns:
            True if size is valid
        """
        w, h = size
        min_w, min_h = self.min_region_size
        max_w, max_h = self.max_region_size

        return min_w <= w <= max_w and min_h <= h <= max_h  # type: ignore[no-any-return]
