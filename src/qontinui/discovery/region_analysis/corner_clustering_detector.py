"""
Corner Clustering Detector - Corner-based grid detection

Uses corner detection (Harris/Shi-Tomasi) and clustering to identify
grid patterns from corner points.
"""

import logging
from typing import Any

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from qontinui.discovery.region_analysis.base import (
    BaseRegionAnalyzer,
    BoundingBox,
    DetectedRegion,
    RegionType,
)

logger = logging.getLogger(__name__)


class CornerClusteringDetector(BaseRegionAnalyzer):
    """
    Detects inventory grids using corner detection and clustering

    Algorithm:
    1. Detect corners using Shi-Tomasi or Harris corner detector
    2. Cluster corners into potential grid points
    3. Compute spacing histograms to find uniform spacing
    4. Validate grid structure
    5. Extract grid parameters and cells
    """

    @property
    def name(self) -> str:
        return "corner_clustering_detector"

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "corner_method": "shi_tomasi",  # or "harris"
            "max_corners": 1000,
            "quality_level": 0.01,
            "min_distance": 10,
            "min_cell_size": 24,
            "max_cell_size": 150,
            "spacing_tolerance": 0.15,  # 15% tolerance for spacing uniformity
            "min_grid_rows": 2,
            "min_grid_cols": 2,
        }

    def analyze(self, image: np.ndarray, **kwargs) -> list[DetectedRegion]:
        """Detect inventory grids using corner clustering"""
        params = {**self.get_default_parameters(), **kwargs}

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Detect corners
        corners = self._detect_corners(gray, params)

        if len(corners) < params["min_grid_rows"] * params["min_grid_cols"] * 4:
            return []

        # Find grid spacing from corner positions
        grid_spacings = self._find_grid_spacing_from_corners(corners, params)

        if not grid_spacings:
            return []

        # Extract grid regions
        regions = []
        for spacing_info in grid_spacings:
            grid_regions = self._extract_grid_from_spacing(
                corners, spacing_info, image.shape, params
            )
            regions.extend(grid_regions)

        return regions

    def _detect_corners(self, gray: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """Detect corners in the image"""
        if params["corner_method"] == "shi_tomasi":
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=params["max_corners"],
                qualityLevel=params["quality_level"],
                minDistance=params["min_distance"],
                blockSize=3,
            )
        else:  # harris
            gray_float = np.float32(gray)
            dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
            dst = cv2.dilate(dst, None)

            # Threshold for corner detection
            threshold = params["quality_level"] * dst.max()
            corners_y, corners_x = np.where(dst > threshold)
            corners = np.array([[[x, y]] for x, y in zip(corners_x, corners_y, strict=False)], dtype=np.float32)

        if corners is None:
            return np.array([])

        # Reshape to (N, 2)
        return corners.reshape(-1, 2)

    def _find_grid_spacing_from_corners(
        self, corners: np.ndarray, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Analyze corner positions to find uniform grid spacing

        Returns:
            List of potential grid spacing configurations
        """
        if len(corners) < 4:
            return []

        # Compute horizontal and vertical distances between corners
        h_distances = []
        v_distances = []

        for i, (x1, y1) in enumerate(corners):
            for x2, y2 in corners[i + 1 :]:
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                # Horizontal alignment (similar y)
                if dy < 5:
                    if params["min_cell_size"] <= dx <= params["max_cell_size"]:
                        h_distances.append(dx)

                # Vertical alignment (similar x)
                if dx < 5:
                    if params["min_cell_size"] <= dy <= params["max_cell_size"]:
                        v_distances.append(dy)

        if not h_distances or not v_distances:
            return []

        # Find most common spacing using histogram
        spacing_x = self._find_dominant_spacing(h_distances, params)
        spacing_y = self._find_dominant_spacing(v_distances, params)

        if not spacing_x or not spacing_y:
            return []

        return [
            {
                "spacing_x": spacing_x,
                "spacing_y": spacing_y,
                "cell_width": spacing_x,
                "cell_height": spacing_y,
            }
        ]

    def _find_dominant_spacing(
        self, distances: list[float], params: dict[str, Any]
    ) -> int | None:
        """Find the dominant spacing from a list of distances"""
        if not distances:
            return None

        # Bin distances with tolerance
        bins = {}
        for dist in distances:
            # Find existing bin
            found_bin = False
            for bin_center in bins.keys():
                if abs(dist - bin_center) / bin_center < params["spacing_tolerance"]:
                    bins[bin_center].append(dist)
                    found_bin = True
                    break

            if not found_bin:
                bins[dist] = [dist]

        if not bins:
            return None

        # Find bin with most entries
        dominant_bin = max(bins.items(), key=lambda x: len(x[1]))

        # Return average of distances in dominant bin
        return int(np.mean(dominant_bin[1]))

    def _extract_grid_from_spacing(
        self,
        corners: np.ndarray,
        spacing_info: dict[str, Any],
        img_shape: tuple[int, int, ...],
        params: dict[str, Any],
    ) -> list[DetectedRegion]:
        """Extract grid structure from corners and spacing"""
        spacing_x = spacing_info["spacing_x"]
        spacing_y = spacing_info["spacing_y"]
        tolerance = params["spacing_tolerance"]

        # Cluster corners into grid positions
        grid_points = {}

        for x, y in corners:
            # Find potential grid position
            grid_x = round(x / spacing_x)
            grid_y = round(y / spacing_y)

            # Verify it's close to a grid position
            expected_x = grid_x * spacing_x
            expected_y = grid_y * spacing_y

            if (
                abs(x - expected_x) < spacing_x * tolerance
                and abs(y - expected_y) < spacing_y * tolerance
            ):
                if (grid_x, grid_y) not in grid_points:
                    grid_points[(grid_x, grid_y)] = (x, y)

        if len(grid_points) < params["min_grid_rows"] * params["min_grid_cols"]:
            return []

        # Extract contiguous grid regions using DBSCAN
        grid_coords = np.array(list(grid_points.keys()))

        # DBSCAN to find contiguous grids
        clustering = DBSCAN(eps=1.5, min_samples=params["min_grid_rows"] * params["min_grid_cols"])
        labels = clustering.fit_predict(grid_coords)

        # Extract each grid cluster
        regions = []
        for label in set(labels):
            if label == -1:  # noise
                continue

            cluster_coords = grid_coords[labels == label]

            if len(cluster_coords) >= params["min_grid_rows"] * params["min_grid_cols"]:
                region = self._create_grid_region(cluster_coords, spacing_x, spacing_y, params)
                if region:
                    regions.append(region)

        return regions

    def _create_grid_region(
        self,
        grid_coords: np.ndarray,
        spacing_x: int,
        spacing_y: int,
        params: dict[str, Any],
    ) -> DetectedRegion | None:
        """Create a DetectedRegion from grid coordinates"""
        if len(grid_coords) < params["min_grid_rows"] * params["min_grid_cols"]:
            return None

        # Get grid bounds
        min_grid_x = grid_coords[:, 0].min()
        max_grid_x = grid_coords[:, 0].max()
        min_grid_y = grid_coords[:, 1].min()
        max_grid_y = grid_coords[:, 1].max()

        cols = int(max_grid_x - min_grid_x + 1)
        rows = int(max_grid_y - min_grid_y + 1)

        if rows < params["min_grid_rows"] or cols < params["min_grid_cols"]:
            return None

        # Calculate pixel coordinates
        x_start = int(min_grid_x * spacing_x)
        y_start = int(min_grid_y * spacing_y)
        width = int(cols * spacing_x)
        height = int(rows * spacing_y)

        # Generate cell metadata
        cells = []
        for row in range(rows):
            for col in range(cols):
                cells.append(
                    {
                        "row": row,
                        "col": col,
                        "x": x_start + col * spacing_x,
                        "y": y_start + row * spacing_y,
                        "width": spacing_x,
                        "height": spacing_y,
                    }
                )

        # Calculate confidence based on corner density
        expected_corners = rows * cols * 4  # 4 corners per cell
        actual_corners = len(grid_coords)
        corner_density = min(
            1.0, actual_corners / (expected_corners * 0.25)
        )  # At least 25% corners
        confidence = min(0.95, 0.6 + corner_density * 0.35)

        return DetectedRegion(
            bounding_box=BoundingBox(x_start, y_start, width, height),
            confidence=confidence,
            region_type=RegionType.INVENTORY_GRID,
            label="Inventory Grid",
            metadata={
                "grid_rows": rows,
                "grid_cols": cols,
                "cell_width": spacing_x,
                "cell_height": spacing_y,
                "spacing_x": spacing_x,
                "spacing_y": spacing_y,
                "cells": cells,
                "detector": "corner_clustering",
                "method": "corner_detection",
            },
        )
