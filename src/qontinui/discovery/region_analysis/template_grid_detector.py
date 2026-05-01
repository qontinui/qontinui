"""
Template Grid Detector - Template matching for grid cells

Automatically detects a representative inventory cell and uses it as a template
to find all similar cells, then extracts the grid structure.
"""

import logging
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


class TemplateGridDetector(BaseRegionAnalyzer):
    """
    Detects inventory grids using template matching

    Algorithm:
    1. Detect candidate cell templates using edge/color analysis
    2. Extract most promising template
    3. Use template matching to find all similar cells
    4. Cluster matches into grid structure
    5. Validate uniform spacing
    6. Extract grid parameters
    """

    @property
    def name(self) -> str:
        return "template_grid_detector"

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "min_template_size": 24,
            "max_template_size": 120,
            "template_method": cv2.TM_CCOEFF_NORMED,
            "match_threshold": 0.6,
            "spacing_tolerance": 0.2,
            "min_grid_rows": 2,
            "min_grid_cols": 2,
            "edge_based": True,
        }

    def analyze(self, image: np.ndarray, **kwargs) -> list[DetectedRegion]:  # type: ignore[override]
        """Detect inventory grids using template matching"""
        params = {**self.get_default_parameters(), **kwargs}

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Find candidate templates
        templates = self._extract_candidate_templates(gray, image, params)

        if not templates:
            return []

        # Try each template and find best grid
        best_regions: list[Any] = []
        best_score = 0

        for template_info in templates[:3]:  # Try top 3 templates
            regions = self._match_template_and_extract_grid(
                gray, image, template_info, params
            )

            if regions:
                # Score based on number of cells found
                score = sum(
                    r.metadata.get("grid_rows", 0) * r.metadata.get("grid_cols", 0)
                    for r in regions
                )
                if score > best_score:
                    best_score = score
                    best_regions = regions

        return best_regions

    def _extract_candidate_templates(
        self, gray: np.ndarray, color_img: np.ndarray, params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Extract candidate cell templates from the image

        Returns:
            List of template dictionaries with template, bbox, and score
        """
        candidates: list[Any] = []

        # Use edge detection to find rectangular regions
        edges = cv2.Canny(gray, 50, 150)

        # Dilate to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours for potential inventory cells
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Check size
            if not (
                params["min_template_size"] <= w <= params["max_template_size"]
                and params["min_template_size"] <= h <= params["max_template_size"]
            ):
                continue

            # Check aspect ratio (should be roughly square)
            aspect_ratio = w / h if h > 0 else 0
            if not (0.7 <= aspect_ratio <= 1.3):
                continue

            # Extract template
            template = gray[y : y + h, x : x + w]

            # Calculate template quality score
            edge_density = np.sum(edges[y : y + h, x : x + w] > 0) / (w * h)
            variance = np.var(template)

            score = edge_density * 0.5 + (variance / 255.0) * 0.5

            candidates.append(
                {
                    "template": template,
                    "color_template": (
                        color_img[y : y + h, x : x + w]
                        if len(color_img.shape) == 3
                        else None
                    ),
                    "bbox": (x, y, w, h),
                    "score": score,
                }
            )

        # Sort by score
        candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates

    def _match_template_and_extract_grid(
        self,
        gray: np.ndarray,
        color_img: np.ndarray,
        template_info: dict[str, Any],
        params: dict[str, Any],
    ) -> list[DetectedRegion]:
        """Match template and extract grid structure"""
        template = template_info["template"]
        t_h, t_w = template.shape[:2]

        # Perform template matching
        result = cv2.matchTemplate(gray, template, params["template_method"])

        # Find matches above threshold
        locations = np.where(result >= params["match_threshold"])
        matches = list(zip(locations[1], locations[0], strict=False))  # (x, y)

        if len(matches) < params["min_grid_rows"] * params["min_grid_cols"]:
            return []

        # Non-maximum suppression to remove overlapping matches
        matches = self._non_max_suppression(
            matches, t_w, t_h, result, params["match_threshold"]
        )

        if len(matches) < params["min_grid_rows"] * params["min_grid_cols"]:
            return []

        # Extract grid structure from matches
        return self._extract_grid_from_matches(matches, t_w, t_h, gray.shape, params)  # type: ignore[arg-type]

    def _non_max_suppression(
        self,
        matches: list[tuple[int, int]],
        width: int,
        height: int,
        scores: np.ndarray,
        threshold: float,
    ) -> list[tuple[int, int]]:
        """Remove overlapping matches, keeping only the best ones"""
        if not matches:
            return []

        # Get scores for each match
        match_scores = [(x, y, scores[y, x]) for x, y in matches]
        match_scores.sort(key=lambda x: x[2], reverse=True)

        kept_matches: list[Any] = []

        for x, y, _score in match_scores:
            # Check if this match overlaps with any kept match
            overlaps = False
            for kx, ky in kept_matches:
                # Check overlap
                if abs(x - kx) < width * 0.5 and abs(y - ky) < height * 0.5:
                    overlaps = True
                    break

            if not overlaps:
                kept_matches.append((x, y))

        return kept_matches

    def _extract_grid_from_matches(
        self,
        matches: list[tuple[int, int]],
        cell_width: int,
        cell_height: int,
        img_shape: tuple[int, int],
        params: dict[str, Any],
    ) -> list[DetectedRegion]:
        """Extract grid structure from template matches"""
        if len(matches) < params["min_grid_rows"] * params["min_grid_cols"]:
            return []

        # Compute spacing between matches
        spacings_x: list[Any] = []
        spacings_y: list[Any] = []

        for i, (x1, y1) in enumerate(matches):
            for x2, y2 in matches[i + 1 :]:
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                if (
                    dy < 5 and cell_width * 0.8 <= dx <= cell_width * 3
                ):  # Horizontal neighbors
                    spacings_x.append(dx)
                if (
                    dx < 5 and cell_height * 0.8 <= dy <= cell_height * 3
                ):  # Vertical neighbors
                    spacings_y.append(dy)

        if not spacings_x or not spacings_y:
            # Use cell size as spacing
            spacing_x = cell_width
            spacing_y = cell_height
        else:
            # Use median spacing
            spacing_x = int(np.median(spacings_x))
            spacing_y = int(np.median(spacings_y))

        # Cluster matches into grid positions
        grid_positions: dict[str, Any] = {}
        matches_array = np.array(matches)
        x_min, y_min = matches_array.min(axis=0)

        for x, y in matches:
            grid_x = round((x - x_min) / spacing_x)
            grid_y = round((y - y_min) / spacing_y)

            # Check if close enough to grid position
            expected_x = x_min + grid_x * spacing_x
            expected_y = y_min + grid_y * spacing_y

            if (
                abs(x - expected_x) < spacing_x * params["spacing_tolerance"]
                and abs(y - expected_y) < spacing_y * params["spacing_tolerance"]
            ):
                if (grid_x, grid_y) not in grid_positions:
                    grid_positions[(grid_x, grid_y)] = (x, y)  # type: ignore[index]

        if len(grid_positions) < params["min_grid_rows"] * params["min_grid_cols"]:
            return []

        # Extract grid bounds
        grid_coords = list(grid_positions.keys())
        grid_xs = [pos[0] for pos in grid_coords]
        grid_ys = [pos[1] for pos in grid_coords]

        cols = int(max(grid_xs)) - int(min(grid_xs)) + 1
        rows = int(max(grid_ys)) - int(min(grid_ys)) + 1

        if rows < params["min_grid_rows"] or cols < params["min_grid_cols"]:
            return []

        # Calculate bounding box
        x_start = int(x_min)
        y_start = int(y_min)
        width = int(cols * spacing_x + cell_width)
        height = int(rows * spacing_y + cell_height)

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

        # Calculate confidence
        expected_cells = rows * cols
        actual_cells = len(grid_positions)
        confidence = min(0.95, 0.5 + (actual_cells / expected_cells) * 0.45)

        region = DetectedRegion(
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
                "detector": "template_grid",
                "method": "template_matching",
            },
        )

        return [region]
