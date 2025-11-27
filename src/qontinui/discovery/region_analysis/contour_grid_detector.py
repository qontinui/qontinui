"""
Contour Grid Detector - Rectangle clustering approach

Finds all rectangular contours and clusters them into grid patterns based on
size similarity and spatial arrangement.
"""

import logging
from typing import Any

import cv2
import numpy as np
from sklearn.cluster import DBSCAN, KMeans

from qontinui.discovery.region_analysis.base import (
    BaseRegionAnalyzer,
    BoundingBox,
    DetectedRegion,
    RegionType,
)

logger = logging.getLogger(__name__)


class ContourGridDetector(BaseRegionAnalyzer):
    """
    Detects inventory grids using contour analysis and clustering

    Algorithm:
    1. Find all rectangular contours in the image
    2. Filter contours by size similarity
    3. Cluster contours spatially using DBSCAN
    4. Check for grid arrangement within clusters
    5. Extract grid parameters and cells
    """

    @property
    def name(self) -> str:
        return "contour_grid_detector"

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "canny_low": 50,
            "canny_high": 150,
            "min_cell_size": 24,
            "max_cell_size": 150,
            "size_tolerance": 0.2,  # 20% size variation allowed
            "spacing_tolerance": 0.2,
            "min_grid_rows": 2,
            "min_grid_cols": 2,
            "rectangularity_threshold": 0.75,
        }

    def analyze(self, image: np.ndarray, **kwargs) -> list[DetectedRegion]:
        """Detect inventory grids using contour clustering"""
        params = {**self.get_default_parameters(), **kwargs}

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Find rectangular contours
        rectangles = self._find_rectangular_contours(gray, params)

        if len(rectangles) < params["min_grid_rows"] * params["min_grid_cols"]:
            return []

        # Cluster by size
        size_clusters = self._cluster_by_size(rectangles, params)

        # For each size cluster, find spatial grid arrangements
        grid_regions = []
        for cluster in size_clusters:
            if len(cluster) >= params["min_grid_rows"] * params["min_grid_cols"]:
                grids = self._find_grid_arrangements(cluster, params)
                grid_regions.extend(grids)

        return grid_regions

    def _find_rectangular_contours(
        self, gray: np.ndarray, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Find rectangular contours in the image

        Returns:
            List of rectangle dictionaries with bbox and features
        """
        # Edge detection
        edges = cv2.Canny(gray, params["canny_low"], params["canny_high"])

        # Morphological operations to close gaps
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Size filter
            if not (
                params["min_cell_size"] <= w <= params["max_cell_size"]
                and params["min_cell_size"] <= h <= params["max_cell_size"]
            ):
                continue

            # Calculate rectangularity (how rectangular is the contour)
            area = cv2.contourArea(contour)
            bbox_area = w * h

            if bbox_area == 0:
                continue

            rectangularity = area / bbox_area

            if rectangularity < params["rectangularity_threshold"]:
                continue

            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 0

            rectangles.append(
                {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "area": area,
                    "aspect_ratio": aspect_ratio,
                    "rectangularity": rectangularity,
                }
            )

        return rectangles

    def _cluster_by_size(
        self, rectangles: list[dict[str, Any]], params: dict[str, Any]
    ) -> list[list[dict[str, Any]]]:
        """
        Cluster rectangles by similar size

        Returns:
            List of size-based clusters
        """
        if not rectangles:
            return []

        # Extract size features
        sizes = np.array([[r["width"], r["height"]] for r in rectangles])

        # Use K-means clustering for size grouping
        # Try different numbers of clusters
        best_clusters = []
        best_score = 0

        for n_clusters in range(2, min(6, len(rectangles) + 1)):
            if n_clusters > len(rectangles):
                break

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(sizes)

            # Evaluate clusters
            for label in range(n_clusters):
                cluster_rects = [
                    rectangles[i] for i in range(len(rectangles)) if labels[i] == label
                ]

                if len(cluster_rects) >= params["min_grid_rows"] * params["min_grid_cols"]:
                    # Check size consistency
                    widths = [r["width"] for r in cluster_rects]
                    heights = [r["height"] for r in cluster_rects]

                    if widths and heights:
                        width_std = np.std(widths) / np.mean(widths)
                        height_std = np.std(heights) / np.mean(heights)

                        if (
                            width_std < params["size_tolerance"]
                            and height_std < params["size_tolerance"]
                        ):
                            score = len(cluster_rects) * (1 - width_std) * (1 - height_std)
                            if score > best_score:
                                best_score = score

                            if cluster_rects not in best_clusters:
                                best_clusters.append(cluster_rects)

        # Also try treating all as one cluster if size is uniform
        if rectangles:
            widths = [r["width"] for r in rectangles]
            heights = [r["height"] for r in rectangles]

            width_std = np.std(widths) / (np.mean(widths) + 1e-7)
            height_std = np.std(heights) / (np.mean(heights) + 1e-7)

            if width_std < params["size_tolerance"] and height_std < params["size_tolerance"]:
                if rectangles not in best_clusters:
                    best_clusters.append(rectangles)

        return best_clusters if best_clusters else [rectangles]

    def _find_grid_arrangements(
        self, rectangles: list[dict[str, Any]], params: dict[str, Any]
    ) -> list[DetectedRegion]:
        """Find grid arrangements within a cluster of similar-sized rectangles"""
        if len(rectangles) < params["min_grid_rows"] * params["min_grid_cols"]:
            return []

        # Extract positions
        positions = np.array([[r["x"], r["y"]] for r in rectangles])

        # Spatial clustering using DBSCAN
        avg_width = np.mean([r["width"] for r in rectangles])
        avg_height = np.mean([r["height"] for r in rectangles])

        eps = max(avg_width, avg_height) * 2  # Allow some spacing between cells
        clustering = DBSCAN(eps=eps, min_samples=params["min_grid_rows"])
        labels = clustering.fit_predict(positions)

        grid_regions = []

        # Process each spatial cluster
        for label in set(labels):
            if label == -1:  # noise
                continue

            cluster_rects = [rectangles[i] for i in range(len(rectangles)) if labels[i] == label]

            if len(cluster_rects) >= params["min_grid_rows"] * params["min_grid_cols"]:
                # Check if cluster forms a grid
                grid_region = self._extract_grid_structure(cluster_rects, params)
                if grid_region:
                    grid_regions.append(grid_region)

        return grid_regions

    def _extract_grid_structure(
        self, rectangles: list[dict[str, Any]], params: dict[str, Any]
    ) -> DetectedRegion | None:
        """Extract grid structure from a cluster of rectangles"""
        if len(rectangles) < params["min_grid_rows"] * params["min_grid_cols"]:
            return None

        # Calculate average cell size
        avg_width = int(np.mean([r["width"] for r in rectangles]))
        avg_height = int(np.mean([r["height"] for r in rectangles]))

        # Get positions
        positions = [(r["x"], r["y"]) for r in rectangles]

        # Sort positions to find grid structure
        x_coords = sorted(set(x for x, y in positions))
        y_coords = sorted(set(y for x, y in positions))

        # Calculate spacing
        if len(x_coords) < params["min_grid_cols"] or len(y_coords) < params["min_grid_rows"]:
            return None

        x_diffs = [x_coords[i + 1] - x_coords[i] for i in range(len(x_coords) - 1)]
        y_diffs = [y_coords[i + 1] - y_coords[i] for i in range(len(y_coords) - 1)]

        if not x_diffs or not y_diffs:
            return None

        spacing_x = int(np.median(x_diffs))
        spacing_y = int(np.median(y_diffs))

        # Check spacing consistency
        if x_diffs:
            x_spacing_std = np.std(x_diffs) / (spacing_x + 1e-7)
            if x_spacing_std > params["spacing_tolerance"]:
                return None

        if y_diffs:
            y_spacing_std = np.std(y_diffs) / (spacing_y + 1e-7)
            if y_spacing_std > params["spacing_tolerance"]:
                return None

        # Calculate grid dimensions
        cols = len(x_coords)
        rows = len(y_coords)

        if rows < params["min_grid_rows"] or cols < params["min_grid_cols"]:
            return None

        # Grid bounding box
        x_start = min(x_coords)
        y_start = min(y_coords)
        width = (cols - 1) * spacing_x + avg_width
        height = (rows - 1) * spacing_y + avg_height

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
                        "width": avg_width,
                        "height": avg_height,
                    }
                )

        # Calculate confidence based on grid completeness
        expected_cells = rows * cols
        actual_cells = len(rectangles)
        completeness = min(1.0, actual_cells / expected_cells)
        confidence = min(0.95, 0.5 + completeness * 0.45)

        return DetectedRegion(
            bounding_box=BoundingBox(x_start, y_start, width, height),
            confidence=confidence,
            region_type=RegionType.INVENTORY_GRID,
            label="Inventory Grid",
            metadata={
                "grid_rows": rows,
                "grid_cols": cols,
                "cell_width": avg_width,
                "cell_height": avg_height,
                "spacing_x": spacing_x,
                "spacing_y": spacing_y,
                "cells": cells,
                "detector": "contour_grid",
                "method": "contour_clustering",
            },
        )
