"""
RANSAC Grid Detector - Robust grid fitting

Uses RANSAC to robustly fit a grid model to detected rectangles, handling
noise and missing/occluded slots.
"""

import logging
from random import sample
from typing import Any

import cv2
import numpy as np

from qontinui.discovery.region_analysis.base import (
    BaseRegionAnalyzer,
    BoundingBox,
    DetectedRegion,
    RegionType,
)

logger = logging.getLogger(__name__)


class RANSACGridDetector(BaseRegionAnalyzer):
    """
    Detects inventory grids using RANSAC for robust grid model fitting

    Algorithm:
    1. Detect candidate rectangular regions
    2. Use RANSAC to fit grid model (origin, spacing, angle)
    3. Handle missing or occluded slots robustly
    4. Validate grid parameters
    5. Extract complete grid structure
    """

    @property
    def name(self) -> str:
        return "ransac_grid_detector"

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "min_cell_size": 24,
            "max_cell_size": 150,
            "ransac_iterations": 100,
            "ransac_threshold": 0.2,  # Fraction of cell size for inliers
            "min_inliers": 6,  # Minimum points to form a grid
            "min_grid_rows": 2,
            "min_grid_cols": 2,
        }

    def analyze(self, image: np.ndarray, **kwargs) -> list[DetectedRegion]:  # type: ignore[override]
        """Detect inventory grids using RANSAC"""
        params = {**self.get_default_parameters(), **kwargs}

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Find candidate rectangles
        candidates = self._find_candidate_rectangles(gray, params)

        if len(candidates) < params["min_inliers"]:
            return []

        # Use RANSAC to fit grid model
        grid_models = self._ransac_grid_fitting(candidates, params)

        if not grid_models:
            return []

        # Convert models to detected regions
        regions: list[Any] = []
        for model in grid_models:
            region = self._model_to_region(model, params)
            if region:
                regions.append(region)

        return regions

    def _find_candidate_rectangles(
        self, gray: np.ndarray, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Find candidate rectangular regions that could be grid cells"""
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates: list[Any] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Size filter
            if not (
                params["min_cell_size"] <= w <= params["max_cell_size"]
                and params["min_cell_size"] <= h <= params["max_cell_size"]
            ):
                continue

            # Calculate rectangularity
            area = cv2.contourArea(contour)
            bbox_area = w * h

            if bbox_area == 0:
                continue

            rectangularity = area / bbox_area

            if rectangularity < 0.7:
                continue

            # Get center point
            cx = x + w // 2
            cy = y + h // 2

            candidates.append(
                {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "cx": cx,
                    "cy": cy,
                }
            )

        return candidates

    def _ransac_grid_fitting(
        self, candidates: list[dict[str, Any]], params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Use RANSAC to fit grid models to candidate rectangles

        Returns:
            List of grid model dictionaries
        """
        best_models: list[Any] = []
        remaining_candidates = candidates.copy()

        # Try to find multiple grids
        while len(remaining_candidates) >= params["min_inliers"]:
            best_model = None
            best_inliers: list[Any] = []
            best_score = 0

            # RANSAC iterations
            for _ in range(params["ransac_iterations"]):
                # Sample minimum points needed to define a grid
                if len(remaining_candidates) < 4:
                    break

                sample_size = min(4, len(remaining_candidates))
                sample_points = sample(remaining_candidates, sample_size)

                # Fit grid model to sample
                model = self._fit_grid_model(sample_points, params)

                if model is None:
                    continue

                # Count inliers
                inliers = self._count_inliers(remaining_candidates, model, params)

                if len(inliers) > len(best_inliers):
                    # Refit model with all inliers
                    refined_model = self._fit_grid_model(inliers, params)

                    if refined_model:
                        # Calculate score
                        score = len(inliers) * refined_model.get("confidence", 0.5)

                        if score > best_score:
                            best_model = refined_model
                            best_inliers = inliers
                            best_score = score

            # If found a good model
            if best_model and len(best_inliers) >= params["min_inliers"]:
                best_model["inliers"] = best_inliers
                best_models.append(best_model)

                # Remove inliers from remaining candidates
                inlier_set = {id(p) for p in best_inliers}
                remaining_candidates = [
                    c for c in remaining_candidates if id(c) not in inlier_set
                ]
            else:
                break

        return best_models

    def _fit_grid_model(
        self, points: list[dict[str, Any]], params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Fit a grid model to a set of points

        Returns:
            Grid model with origin, spacing, and dimensions
        """
        if len(points) < 4:
            return None

        # Extract center points
        centers = np.array([(p["cx"], p["cy"]) for p in points])

        # Find potential spacing by looking at distances
        distances: list[Any] = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                if params["min_cell_size"] <= dist <= params["max_cell_size"] * 2:
                    distances.append(dist)

        if not distances:
            return None

        # Use histogram to find most common spacing
        hist, bins = np.histogram(distances, bins=20)
        most_common_bin = np.argmax(hist)
        spacing = (bins[most_common_bin] + bins[most_common_bin + 1]) / 2

        # Find origin (point closest to min x, min y)
        min_x, min_y = centers.min(axis=0)
        origin_idx = np.argmin(np.sum((centers - [min_x, min_y]) ** 2, axis=1))
        origin = centers[origin_idx]

        # Calculate grid dimensions
        # Project points onto grid
        grid_positions: list[Any] = []
        for cx, cy in centers:
            dx, dy = cx - origin[0], cy - origin[1]
            grid_x = round(dx / spacing)
            grid_y = round(dy / spacing)
            grid_positions.append((grid_x, grid_y))

        if not grid_positions:
            return None

        grid_positions = list(set(grid_positions))  # Remove duplicates
        grid_xs = [pos[0] for pos in grid_positions]
        grid_ys = [pos[1] for pos in grid_positions]

        cols = max(grid_xs) - min(grid_xs) + 1 if grid_xs else 0
        rows = max(grid_ys) - min(grid_ys) + 1 if grid_ys else 0

        if rows < params["min_grid_rows"] or cols < params["min_grid_cols"]:
            return None

        # Average cell size
        avg_width = int(np.mean([p["width"] for p in points]))
        avg_height = int(np.mean([p["height"] for p in points]))

        return {
            "origin": origin,
            "spacing": spacing,
            "spacing_x": int(spacing),
            "spacing_y": int(spacing),
            "rows": rows,
            "cols": cols,
            "cell_width": avg_width,
            "cell_height": avg_height,
            "grid_positions": grid_positions,
            "confidence": min(1.0, len(points) / (rows * cols)),
        }

    def _count_inliers(
        self,
        candidates: list[dict[str, Any]],
        model: dict[str, Any],
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Count candidates that fit the grid model"""
        inliers: list[Any] = []
        origin = model["origin"]
        spacing = model["spacing"]
        threshold = spacing * params["ransac_threshold"]

        for candidate in candidates:
            cx, cy = candidate["cx"], candidate["cy"]

            # Calculate expected grid position
            dx = cx - origin[0]
            dy = cy - origin[1]

            grid_x = round(dx / spacing)
            grid_y = round(dy / spacing)

            # Expected position
            expected_x = origin[0] + grid_x * spacing
            expected_y = origin[1] + grid_y * spacing

            # Check if within threshold
            distance = np.sqrt((cx - expected_x) ** 2 + (cy - expected_y) ** 2)

            if distance < threshold:
                inliers.append(candidate)

        return inliers

    def _model_to_region(
        self, model: dict[str, Any], params: dict[str, Any]
    ) -> DetectedRegion | None:
        """Convert grid model to DetectedRegion"""
        rows = model["rows"]
        cols = model["cols"]

        if rows < params["min_grid_rows"] or cols < params["min_grid_cols"]:
            return None

        origin = model["origin"]
        spacing_x = model["spacing_x"]
        spacing_y = model["spacing_y"]
        cell_width = model["cell_width"]
        cell_height = model["cell_height"]

        # Calculate bounding box
        x_start = int(origin[0] - cell_width // 2)
        y_start = int(origin[1] - cell_height // 2)
        width = cols * spacing_x + cell_width
        height = rows * spacing_y + cell_height

        # Generate cell metadata
        cells: list[Any] = []
        for row in range(rows):
            for col in range(cols):
                cells.append(
                    {
                        "row": row,
                        "col": col,
                        "x": x_start + col * spacing_x,
                        "y": y_start + row * spacing_y,
                        "width": cell_width,
                        "height": cell_height,
                    }
                )

        confidence = model.get("confidence", 0.7)
        confidence = min(0.95, 0.5 + confidence * 0.45)

        return DetectedRegion(
            bounding_box=BoundingBox(x_start, y_start, width, height),
            confidence=confidence,
            region_type=RegionType.INVENTORY_GRID,
            label="Inventory Grid",
            metadata={
                "grid_rows": rows,
                "grid_cols": cols,
                "cell_width": cell_width,
                "cell_height": cell_height,
                "spacing_x": spacing_x,
                "spacing_y": spacing_y,
                "cells": cells,
                "detector": "ransac_grid",
                "method": "ransac_fitting",
            },
        )
