"""
Color Quantization Detector - Color-based segmentation

Uses K-means color clustering to segment inventory regions by their distinct
background colors, then detects grid structure within segmented areas.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

from qontinui.discovery.region_analysis.base import BaseRegionAnalyzer, BoundingBox, DetectedRegion, RegionType

logger = logging.getLogger(__name__)


class ColorQuantizationDetector(BaseRegionAnalyzer):
    """
    Detects inventory grids using color-based segmentation

    Algorithm:
    1. Apply K-means clustering to quantize colors
    2. Segment regions with distinct colors (inventory backgrounds)
    3. Find contiguous regions in each color cluster
    4. Detect grid structure within segmented regions
    5. Validate and extract grid parameters
    """

    @property
    def name(self) -> str:
        return "color_quantization_detector"

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "n_colors": 8,  # Number of color clusters
            "min_region_area": 2000,
            "min_cell_size": 24,
            "max_cell_size": 150,
            "spacing_tolerance": 0.2,
            "min_grid_rows": 2,
            "min_grid_cols": 2,
        }

    def analyze(self, image: np.ndarray, **kwargs) -> List[DetectedRegion]:
        """Detect inventory grids using color quantization"""
        params = {**self.get_default_parameters(), **kwargs}

        # Ensure color image
        if len(image.shape) == 2:
            color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            color_img = image.copy()

        # Perform color quantization
        quantized, labels = self._quantize_colors(color_img, params)

        # Find regions for each color cluster
        color_regions = self._find_color_regions(labels, params)

        # Detect grid structure in each region
        grid_regions = []
        for region_mask in color_regions:
            grids = self._detect_grid_in_region(color_img, region_mask, params)
            grid_regions.extend(grids)

        return grid_regions

    def _quantize_colors(
        self, image: np.ndarray, params: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform color quantization using K-means

        Returns:
            (quantized_image, labels) tuple
        """
        h, w = image.shape[:2]

        # Reshape image for K-means
        pixels = image.reshape(-1, 3).astype(np.float32)

        # Apply K-means
        kmeans = KMeans(
            n_clusters=params["n_colors"], random_state=42, n_init=10, max_iter=100
        )
        labels = kmeans.fit_predict(pixels)

        # Reconstruct quantized image
        centers = kmeans.cluster_centers_.astype(np.uint8)
        quantized = centers[labels].reshape(h, w, 3)

        labels = labels.reshape(h, w)

        return quantized, labels

    def _find_color_regions(
        self, labels: np.ndarray, params: Dict[str, Any]
    ) -> List[np.ndarray]:
        """
        Find contiguous regions for each color cluster

        Returns:
            List of binary masks for significant regions
        """
        regions = []

        # For each color cluster
        for cluster_id in range(params["n_colors"]):
            # Create binary mask
            mask = (labels == cluster_id).astype(np.uint8) * 255

            # Morphological operations to clean up
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            # Find contiguous regions
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)

                if area >= params["min_region_area"]:
                    # Create mask for this region
                    region_mask = np.zeros_like(mask)
                    cv2.drawContours(region_mask, [contour], -1, 255, -1)
                    regions.append(region_mask)

        return regions

    def _detect_grid_in_region(
        self, image: np.ndarray, region_mask: np.ndarray, params: Dict[str, Any]
    ) -> List[DetectedRegion]:
        """Detect grid structure within a color-segmented region"""
        # Apply mask to image
        masked_img = cv2.bitwise_and(image, image, mask=region_mask)

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

        # Find rectangular structures within region
        rectangles = self._find_rectangles_in_region(gray, region_mask, params)

        if len(rectangles) < params["min_grid_rows"] * params["min_grid_cols"]:
            return []

        # Extract grid structure
        grid_region = self._extract_grid_from_rectangles(rectangles, params)

        if grid_region:
            return [grid_region]

        return []

    def _find_rectangles_in_region(
        self, gray: np.ndarray, mask: np.ndarray, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find rectangular structures in the masked region"""
        # Edge detection on masked region
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.bitwise_and(edges, edges, mask=mask)

        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        rectangles = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Size filter
            if not (
                params["min_cell_size"] <= w <= params["max_cell_size"]
                and params["min_cell_size"] <= h <= params["max_cell_size"]
            ):
                continue

            # Check rectangularity
            area = cv2.contourArea(contour)
            bbox_area = w * h

            if bbox_area == 0:
                continue

            rectangularity = area / bbox_area

            if rectangularity < 0.7:
                continue

            rectangles.append(
                {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                }
            )

        return rectangles

    def _extract_grid_from_rectangles(
        self, rectangles: List[Dict[str, Any]], params: Dict[str, Any]
    ) -> Optional[DetectedRegion]:
        """Extract grid structure from rectangles"""
        if len(rectangles) < params["min_grid_rows"] * params["min_grid_cols"]:
            return None

        # Calculate average cell size
        avg_width = int(np.mean([r["width"] for r in rectangles]))
        avg_height = int(np.mean([r["height"] for r in rectangles]))

        # Get unique positions
        x_positions = sorted(set(r["x"] for r in rectangles))
        y_positions = sorted(set(r["y"] for r in rectangles))

        # Check if we have enough positions for a grid
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

        # Check spacing consistency
        if x_diffs:
            x_std = np.std(x_diffs) / (spacing_x + 1e-7)
            if x_std > params["spacing_tolerance"]:
                return None

        if y_diffs:
            y_std = np.std(y_diffs) / (spacing_y + 1e-7)
            if y_std > params["spacing_tolerance"]:
                return None

        # Grid dimensions
        cols = len(x_positions)
        rows = len(y_positions)

        if rows < params["min_grid_rows"] or cols < params["min_grid_cols"]:
            return None

        # Calculate bounding box
        x_start = min(x_positions)
        y_start = min(y_positions)
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

        # Calculate confidence
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
                "detector": "color_quantization",
                "method": "kmeans_segmentation",
            },
        )
