"""
Grid Pattern Detector - Autocorrelation-based grid detection

Uses autocorrelation to find repeating patterns in the image,
identifying grid structures with uniform spacing.
"""

import logging
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

from qontinui.discovery.region_analysis.base import (
    BaseRegionAnalyzer,
    BoundingBox,
    DetectedRegion,
    RegionAnalysisInput,
    RegionAnalysisResult,
    RegionAnalysisType,
    RegionType,
)

logger = logging.getLogger(__name__)


class GridPatternDetector(BaseRegionAnalyzer):
    """
    Detects inventory grids using autocorrelation for pattern recognition

    Algorithm:
    1. Convert image to grayscale
    2. Compute 2D autocorrelation
    3. Find peaks in autocorrelation (indicates repeating patterns)
    4. Extract grid spacing from peak positions
    5. Locate grid region using pattern matching
    6. Validate grid structure and extract cells
    """

    @property
    def analysis_type(self) -> RegionAnalysisType:
        return RegionAnalysisType.PATTERN_ANALYSIS

    @property
    def name(self) -> str:
        return "grid_pattern_detector"

    @property
    def supported_region_types(self) -> list[RegionType]:
        return [RegionType.INVENTORY_GRID]

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "min_cell_size": 32,
            "max_cell_size": 128,
            "min_grid_rows": 2,
            "min_grid_cols": 2,
            "autocorr_threshold": 0.3,
            "edge_detect": True,
        }

    async def analyze(self, input_data: RegionAnalysisInput) -> RegionAnalysisResult:
        """Detect inventory grids using autocorrelation"""
        all_regions = []

        # Process each screenshot
        for idx, screenshot_bytes in enumerate(input_data.screenshot_data):
            # Convert bytes to numpy array
            image = Image.open(BytesIO(screenshot_bytes))
            image_np = np.array(image)

            # Detect regions in this screenshot
            regions = self._analyze_image(image_np, idx, input_data.parameters)
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
            metadata={"total_grids_detected": len(all_regions)},
        )

    def _analyze_image(
        self, image: np.ndarray, screenshot_index: int, params: dict[str, Any]
    ) -> list[DetectedRegion]:
        """Detect inventory grids using autocorrelation"""
        params = {**self.get_default_parameters(), **params}

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Use edges for better pattern detection
        if params["edge_detect"]:
            edges = cv2.Canny(gray, 50, 150)
            analysis_img = edges
        else:
            analysis_img = gray

        # Find grid spacing using autocorrelation
        grid_info = self._find_grid_spacing(analysis_img, params)

        if not grid_info:
            return []

        # Locate grid regions
        regions = self._locate_grids(analysis_img, gray, grid_info, params, screenshot_index)

        return regions

    def _find_grid_spacing(
        self, image: np.ndarray, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Use autocorrelation to find grid spacing

        Returns:
            Dictionary with spacing_x, spacing_y, cell_width, cell_height
        """
        # Normalize image
        img_float = image.astype(np.float32) / 255.0

        # Compute autocorrelation using FFT
        f = np.fft.fft2(img_float)
        autocorr = np.fft.ifft2(f * np.conj(f)).real
        autocorr = np.fft.fftshift(autocorr)

        # Normalize autocorrelation
        autocorr = autocorr / autocorr.max()

        # Find peaks in autocorrelation
        # Center is always the highest peak, so we look for secondary peaks
        h, w = autocorr.shape
        center_y, center_x = h // 2, w // 2

        # Mask out center region
        mask_size = 20
        autocorr_masked = autocorr.copy()
        autocorr_masked[
            center_y - mask_size : center_y + mask_size,
            center_x - mask_size : center_x + mask_size,
        ] = 0

        # Find local maxima
        local_max = maximum_filter(autocorr_masked, size=10)
        peaks = (autocorr_masked == local_max) & (autocorr_masked > params["autocorr_threshold"])

        # Get peak coordinates
        peak_coords = np.argwhere(peaks)

        if len(peak_coords) < 2:
            return None

        # Calculate distances from center
        distances = []
        for y, x in peak_coords:
            dy = abs(y - center_y)
            dx = abs(x - center_x)
            if (
                params["min_cell_size"] <= dx <= params["max_cell_size"]
                or params["min_cell_size"] <= dy <= params["max_cell_size"]
            ):
                distances.append((dx, dy, autocorr[y, x]))

        if not distances:
            return None

        # Find most common spacing (mode of distances)
        distances.sort(key=lambda x: x[2], reverse=True)  # Sort by correlation strength

        # Extract horizontal and vertical spacing
        spacing_x = spacing_y = None
        for dx, dy, _strength in distances:
            if spacing_x is None and dx > params["min_cell_size"]:
                spacing_x = dx
            if spacing_y is None and dy > params["min_cell_size"]:
                spacing_y = dy
            if spacing_x and spacing_y:
                break

        if not spacing_x or not spacing_y:
            return None

        return {
            "spacing_x": int(spacing_x),
            "spacing_y": int(spacing_y),
            "cell_width": int(spacing_x),
            "cell_height": int(spacing_y),
        }

    def _locate_grids(
        self,
        edge_img: np.ndarray,
        gray_img: np.ndarray,
        grid_info: dict[str, Any],
        params: dict[str, Any],
        screenshot_index: int,
    ) -> list[DetectedRegion]:
        """Locate actual grid regions in the image"""
        regions = []

        spacing_x = grid_info["spacing_x"]
        spacing_y = grid_info["spacing_y"]

        # Use template matching to find grid cells
        # Create a simple template (edge square)
        template_size = min(spacing_x, spacing_y, 64)
        template = np.zeros((template_size, template_size), dtype=np.uint8)
        cv2.rectangle(template, (2, 2), (template_size - 3, template_size - 3), 255, 2)

        # Match template
        if edge_img.shape[0] >= template_size and edge_img.shape[1] >= template_size:
            result = cv2.matchTemplate(edge_img, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.3
            locations = np.where(result >= threshold)

            if len(locations[0]) < params["min_grid_rows"] * params["min_grid_cols"]:
                return []

            # Cluster locations into grid
            points = list(zip(locations[1], locations[0], strict=False))  # (x, y)

            if len(points) >= params["min_grid_rows"] * params["min_grid_cols"]:
                # Find grid bounds and structure
                grid_region = self._extract_grid_structure(
                    points,
                    spacing_x,
                    spacing_y,
                    gray_img.shape,
                    params,
                    screenshot_index,
                )

                if grid_region:
                    regions.append(grid_region)

        return regions

    def _extract_grid_structure(
        self,
        points: list[tuple[int, int]],
        spacing_x: int,
        spacing_y: int,
        img_shape: tuple[int, int],
        params: dict[str, Any],
        screenshot_index: int,
    ) -> DetectedRegion | None:
        """Extract grid structure from detected points"""
        if not points:
            return None

        points = np.array(points)

        # Cluster points into grid positions
        # Round to nearest grid position
        grid_positions = set()
        x_min, y_min = points.min(axis=0)

        for x, y in points:
            grid_x = round((x - x_min) / spacing_x)
            grid_y = round((y - y_min) / spacing_y)
            grid_positions.add((grid_x, grid_y))

        if len(grid_positions) < params["min_grid_rows"] * params["min_grid_cols"]:
            return None

        # Extract grid dimensions
        grid_positions_list = list(grid_positions)
        grid_xs = [pos[0] for pos in grid_positions_list]
        grid_ys = [pos[1] for pos in grid_positions_list]

        cols = max(grid_xs) - min(grid_xs) + 1
        rows = max(grid_ys) - min(grid_ys) + 1

        if rows < params["min_grid_rows"] or cols < params["min_grid_cols"]:
            return None

        # Calculate bounding box
        x_start = int(x_min)
        y_start = int(y_min)
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

        # Calculate confidence based on grid completeness
        expected_cells = rows * cols
        actual_cells = len(grid_positions)
        completeness = actual_cells / expected_cells
        confidence = min(0.95, 0.5 + completeness * 0.45)

        return DetectedRegion(
            bounding_box=BoundingBox(x_start, y_start, width, height),
            confidence=confidence,
            region_type=RegionType.INVENTORY_GRID,
            label="Inventory Grid",
            screenshot_index=screenshot_index,
            metadata={
                "grid_rows": rows,
                "grid_cols": cols,
                "cell_width": spacing_x,
                "cell_height": spacing_y,
                "spacing_x": spacing_x,
                "spacing_y": spacing_y,
                "cells": cells,
                "detector": "grid_pattern",
                "method": "autocorrelation",
            },
        )
