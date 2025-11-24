"""
Slot Border Detector - Border-based detection

Detects inventory grids by finding rectangular borders around slots using
morphological operations and edge detection.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from qontinui.discovery.region_analysis.base import BaseRegionAnalyzer, BoundingBox, DetectedRegion, RegionType

logger = logging.getLogger(__name__)


class SlotBorderDetector(BaseRegionAnalyzer):
    """
    Detects inventory grids by finding slot borders

    Algorithm:
    1. Enhance borders using morphological operations
    2. Detect rectangular borders
    3. Group borders by size similarity
    4. Check for grid pattern in border positions
    5. Validate spacing consistency
    6. Extract grid structure
    """

    @property
    def name(self) -> str:
        return "slot_border_detector"

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "canny_low": 30,
            "canny_high": 100,
            "min_cell_size": 24,
            "max_cell_size": 150,
            "border_thickness_range": (1, 5),
            "size_tolerance": 0.2,
            "spacing_tolerance": 0.2,
            "min_grid_rows": 2,
            "min_grid_cols": 2,
        }

    def analyze(self, image: np.ndarray, **kwargs) -> List[DetectedRegion]:
        """Detect inventory grids by finding slot borders"""
        params = {**self.get_default_parameters(), **kwargs}

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Find borders
        borders = self._detect_borders(gray, params)

        if len(borders) < params["min_grid_rows"] * params["min_grid_cols"]:
            return []

        # Group borders by size
        size_groups = self._group_by_size(borders, params)

        # Extract grid patterns from each size group
        grid_regions = []
        for group in size_groups:
            if len(group) >= params["min_grid_rows"] * params["min_grid_cols"]:
                grids = self._extract_grid_pattern(group, params)
                grid_regions.extend(grids)

        return grid_regions

    def _detect_borders(
        self, gray: np.ndarray, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect rectangular borders in the image

        Returns:
            List of border rectangles with properties
        """
        # Apply edge detection
        edges = cv2.Canny(gray, params["canny_low"], params["canny_high"])

        # Morphological operations to enhance borders
        # Close small gaps in borders
        kernel_close = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)

        # Find contours
        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        borders = []

        if hierarchy is None:
            return borders

        hierarchy = hierarchy[0]

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            # Size filter
            if not (
                params["min_cell_size"] <= w <= params["max_cell_size"]
                and params["min_cell_size"] <= h <= params["max_cell_size"]
            ):
                continue

            # Calculate border properties
            area = cv2.contourArea(contour)
            bbox_area = w * h

            if bbox_area == 0:
                continue

            # Check if it's a border (hollow rectangle)
            # A border should have less area than its bounding box
            fill_ratio = area / bbox_area

            # Borders are typically 0.2-0.9 filled (not completely hollow, not completely filled)
            if not (0.2 <= fill_ratio <= 0.9):
                continue

            # Check aspect ratio (should be reasonable)
            aspect_ratio = w / h if h > 0 else 0
            if not (0.5 <= aspect_ratio <= 2.0):
                continue

            # Check if contour approximates a rectangle
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Should have 4 corners for a rectangle
            if not (4 <= len(approx) <= 8):  # Allow some tolerance
                continue

            borders.append(
                {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "area": area,
                    "fill_ratio": fill_ratio,
                    "aspect_ratio": aspect_ratio,
                    "corners": len(approx),
                }
            )

        return borders

    def _group_by_size(
        self, borders: List[Dict[str, Any]], params: Dict[str, Any]
    ) -> List[List[Dict[str, Any]]]:
        """Group borders by similar size"""
        if not borders:
            return []

        groups = []
        used = set()

        # Sort by area for grouping
        borders_sorted = sorted(borders, key=lambda b: b["width"] * b["height"])

        for i, border in enumerate(borders_sorted):
            if i in used:
                continue

            # Start new group
            group = [border]
            used.add(i)

            ref_width = border["width"]
            ref_height = border["height"]

            # Find similar sized borders
            for j, other in enumerate(borders_sorted[i + 1 :], start=i + 1):
                if j in used:
                    continue

                width_diff = abs(other["width"] - ref_width) / ref_width
                height_diff = abs(other["height"] - ref_height) / ref_height

                if (
                    width_diff < params["size_tolerance"]
                    and height_diff < params["size_tolerance"]
                ):
                    group.append(other)
                    used.add(j)

            if len(group) >= params["min_grid_rows"] * params["min_grid_cols"]:
                groups.append(group)

        return groups

    def _extract_grid_pattern(
        self, borders: List[Dict[str, Any]], params: Dict[str, Any]
    ) -> List[DetectedRegion]:
        """Extract grid pattern from grouped borders"""
        if len(borders) < params["min_grid_rows"] * params["min_grid_cols"]:
            return []

        # Get unique x and y positions
        x_positions = sorted(set(b["x"] for b in borders))
        y_positions = sorted(set(b["y"] for b in borders))

        # Calculate average cell size
        avg_width = int(np.mean([b["width"] for b in borders]))
        avg_height = int(np.mean([b["height"] for b in borders]))

        # Find grid spacing
        x_spacings = [
            x_positions[i + 1] - x_positions[i] for i in range(len(x_positions) - 1)
        ]
        y_spacings = [
            y_positions[i + 1] - y_positions[i] for i in range(len(y_positions) - 1)
        ]

        if not x_spacings or not y_spacings:
            return []

        spacing_x = int(np.median(x_spacings))
        spacing_y = int(np.median(y_spacings))

        # Check spacing consistency
        if x_spacings:
            x_std = np.std(x_spacings) / (spacing_x + 1e-7)
            if x_std > params["spacing_tolerance"]:
                return []

        if y_spacings:
            y_std = np.std(y_spacings) / (spacing_y + 1e-7)
            if y_std > params["spacing_tolerance"]:
                return []

        # Cluster borders into grid positions
        grid_map = {}
        x_min = min(x_positions)
        y_min = min(y_positions)

        for border in borders:
            # Calculate grid position
            grid_x = round((border["x"] - x_min) / spacing_x)
            grid_y = round((border["y"] - y_min) / spacing_y)

            # Verify it's close to grid position
            expected_x = x_min + grid_x * spacing_x
            expected_y = y_min + grid_y * spacing_y

            if (
                abs(border["x"] - expected_x) < spacing_x * params["spacing_tolerance"]
                and abs(border["y"] - expected_y)
                < spacing_y * params["spacing_tolerance"]
            ):
                grid_map[(grid_x, grid_y)] = border

        if len(grid_map) < params["min_grid_rows"] * params["min_grid_cols"]:
            return []

        # Extract grid dimensions
        grid_coords = list(grid_map.keys())
        grid_xs = [c[0] for c in grid_coords]
        grid_ys = [c[1] for c in grid_coords]

        cols = max(grid_xs) - min(grid_xs) + 1
        rows = max(grid_ys) - min(grid_ys) + 1

        if rows < params["min_grid_rows"] or cols < params["min_grid_cols"]:
            return []

        # Calculate grid bounding box
        x_start = x_min
        y_start = y_min
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
        actual_cells = len(grid_map)
        completeness = actual_cells / expected_cells
        confidence = min(0.95, 0.6 + completeness * 0.35)

        region = DetectedRegion(
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
                "detector": "slot_border",
                "method": "border_detection",
            },
        )

        return [region]
