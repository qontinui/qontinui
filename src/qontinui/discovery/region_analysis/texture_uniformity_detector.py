"""
Texture Uniformity Detector - Background pattern detection

Detects regions with uniform texture/background patterns using Gabor filters
and Local Binary Patterns (LBP).
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


class TextureUniformityDetector(BaseRegionAnalyzer):
    """
    Detects inventory grids using texture uniformity analysis

    Algorithm:
    1. Compute Local Binary Pattern (LBP) texture features
    2. Detect regions with uniform texture (inventory slots often similar)
    3. Use Gabor filters to detect repeating patterns
    4. Cluster uniform regions
    5. Extract grid structure from clustered regions
    """

    @property
    def name(self) -> str:
        return "texture_uniformity_detector"

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "lbp_radius": 3,
            "lbp_points": 24,
            "window_size": 32,
            "stride": 16,
            "texture_similarity_threshold": 0.7,
            "min_cell_size": 24,
            "max_cell_size": 120,
            "min_grid_rows": 2,
            "min_grid_cols": 2,
            "use_gabor": True,
        }

    def analyze(self, image: np.ndarray, **kwargs) -> list[DetectedRegion]:  # type: ignore[override]
        """Detect inventory grids using texture uniformity"""
        params = {**self.get_default_parameters(), **kwargs}

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Compute texture features for sliding windows
        texture_map = self._compute_texture_map(gray, params)

        if texture_map is None or len(texture_map) == 0:
            return []

        # Find uniform texture regions
        uniform_regions = self._find_uniform_texture_regions(texture_map, params)

        if not uniform_regions:
            return []

        # Cluster regions into grids
        grid_regions = self._cluster_into_grids(uniform_regions, gray.shape, params)  # type: ignore[arg-type]

        return grid_regions

    def _compute_lbp(
        self, gray: np.ndarray, radius: int = 3, n_points: int = 24
    ) -> np.ndarray:
        """
        Compute Local Binary Pattern

        This is a simplified LBP implementation
        """
        h, w = gray.shape
        lbp = np.zeros((h, w), dtype=np.uint8)

        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = gray[i, j]
                code = 0

                # Sample points on circle
                for p in range(n_points):
                    angle = 2 * np.pi * p / n_points
                    x = j + radius * np.cos(angle)
                    y = i - radius * np.sin(angle)

                    # Bilinear interpolation
                    x1, y1 = int(x), int(y)
                    x2, y2 = x1 + 1, y1 + 1

                    if 0 <= y1 < h - 1 and 0 <= x1 < w - 1:
                        # Interpolate
                        fx, fy = x - x1, y - y1
                        val = (
                            (1 - fx) * (1 - fy) * gray[y1, x1]
                            + fx * (1 - fy) * gray[y1, x2]
                            + (1 - fx) * fy * gray[y2, x1]
                            + fx * fy * gray[y2, x2]
                        )

                        if val >= center:
                            code |= 1 << p

                lbp[i, j] = code % 256  # Keep in uint8 range

        return lbp

    def _compute_texture_map(
        self, gray: np.ndarray, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Compute texture features for sliding windows

        Returns:
            List of window dictionaries with position and features
        """
        window_size = params["window_size"]
        stride = params["stride"]
        h, w = gray.shape

        # Compute LBP
        lbp = self._compute_lbp(gray, params["lbp_radius"], params["lbp_points"])

        windows = []

        for y in range(0, h - window_size, stride):
            for x in range(0, w - window_size, stride):
                # Extract window
                window_gray = gray[y : y + window_size, x : x + window_size]
                window_lbp = lbp[y : y + window_size, x : x + window_size]

                # Compute texture features
                # 1. LBP histogram
                lbp_hist = np.histogram(window_lbp, bins=32, range=(0, 256))[0]
                lbp_hist = lbp_hist.astype(np.float32) / (lbp_hist.sum() + 1e-7)

                # 2. Variance and mean
                variance = np.var(window_gray)
                mean = np.mean(window_gray)

                # 3. Edge density
                edges = cv2.Canny(window_gray, 50, 150)
                edge_density = np.sum(edges > 0) / (window_size * window_size)

                windows.append(
                    {
                        "x": x,
                        "y": y,
                        "width": window_size,
                        "height": window_size,
                        "lbp_hist": lbp_hist,
                        "variance": variance,
                        "mean": mean,
                        "edge_density": edge_density,
                    }
                )

        return windows

    def _find_uniform_texture_regions(
        self, texture_map: list[dict[str, Any]], params: dict[str, Any]
    ) -> list[list[dict[str, Any]]]:
        """
        Find groups of windows with similar texture

        Returns:
            List of texture clusters
        """
        if not texture_map:
            return []

        # Compute pairwise similarity matrix
        n = len(texture_map)
        similarity = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Compare LBP histograms using chi-square distance
                hist1 = texture_map[i]["lbp_hist"]
                hist2 = texture_map[j]["lbp_hist"]

                chi_square = np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + 1e-7))
                sim = np.exp(-chi_square)

                similarity[i, j] = similarity[j, i] = sim

        # Cluster using similarity threshold
        clusters = []
        visited = set()

        for i in range(n):
            if i in visited:
                continue

            # Start new cluster
            cluster = [texture_map[i]]
            visited.add(i)

            # Add similar windows
            for j in range(n):
                if (
                    j not in visited
                    and similarity[i, j] > params["texture_similarity_threshold"]
                ):
                    cluster.append(texture_map[j])
                    visited.add(j)

            if len(cluster) >= params["min_grid_rows"] * params["min_grid_cols"]:
                clusters.append(cluster)

        return clusters

    def _cluster_into_grids(
        self,
        uniform_regions: list[list[dict[str, Any]]],
        img_shape: tuple[int, int],
        params: dict[str, Any],
    ) -> list[DetectedRegion]:
        """Cluster uniform texture regions into grid structures"""
        grid_regions = []

        for cluster in uniform_regions:
            # Extract positions
            positions = np.array([[w["x"], w["y"]] for w in cluster])

            # Use DBSCAN to find spatially contiguous groups
            window_size = params["window_size"]
            clustering = DBSCAN(
                eps=window_size * 1.5, min_samples=params["min_grid_rows"]
            )
            labels = clustering.fit_predict(positions)

            # Process each spatial cluster
            for label in set(labels):
                if label == -1:  # noise
                    continue

                cluster_windows = [
                    cluster[i] for i in range(len(cluster)) if labels[i] == label
                ]

                if (
                    len(cluster_windows)
                    >= params["min_grid_rows"] * params["min_grid_cols"]
                ):
                    grid_region = self._extract_grid_from_windows(
                        cluster_windows, params
                    )
                    if grid_region:
                        grid_regions.append(grid_region)

        return grid_regions

    def _extract_grid_from_windows(
        self, windows: list[dict[str, Any]], params: dict[str, Any]
    ) -> DetectedRegion | None:
        """Extract grid structure from uniform texture windows"""
        if len(windows) < params["min_grid_rows"] * params["min_grid_cols"]:
            return None

        # Find grid spacing from window positions
        positions = [(w["x"], w["y"]) for w in windows]

        # Compute horizontal and vertical spacings
        x_positions = sorted({x for x, y in positions})
        y_positions = sorted({y for x, y in positions})

        if (
            len(x_positions) < params["min_grid_cols"]
            or len(y_positions) < params["min_grid_rows"]
        ):
            return None

        # Calculate spacing
        x_diffs = [
            x_positions[i + 1] - x_positions[i] for i in range(len(x_positions) - 1)
        ]
        y_diffs = [
            y_positions[i + 1] - y_positions[i] for i in range(len(y_positions) - 1)
        ]

        if not x_diffs or not y_diffs:
            return None

        spacing_x = int(np.median(x_diffs))
        spacing_y = int(np.median(y_diffs))

        # Verify spacing is consistent
        if (
            max(x_diffs) - min(x_diffs) > spacing_x * 0.3
            or max(y_diffs) - min(y_diffs) > spacing_y * 0.3
        ):
            return None

        # Calculate grid dimensions
        x_start = min(x_positions)
        y_start = min(y_positions)
        cols = len(x_positions)
        rows = len(y_positions)

        window_size = params["window_size"]
        width = cols * spacing_x + window_size
        height = rows * spacing_y + window_size

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
                        "width": window_size,
                        "height": window_size,
                    }
                )

        # Calculate confidence based on texture uniformity
        confidence = min(0.9, 0.5 + (len(windows) / (rows * cols)) * 0.4)

        return DetectedRegion(
            bounding_box=BoundingBox(x_start, y_start, width, height),
            confidence=confidence,
            region_type=RegionType.INVENTORY_GRID,
            label="Inventory Grid",
            metadata={
                "grid_rows": rows,
                "grid_cols": cols,
                "cell_width": window_size,
                "cell_height": window_size,
                "spacing_x": spacing_x,
                "spacing_y": spacing_y,
                "cells": cells,
                "detector": "texture_uniformity",
                "method": "lbp_texture",
            },
        )
