"""
Frequency Analysis Detector - FFT-based grid detection

Uses 2D Fourier Transform to detect periodic patterns in inventory grids,
extracting grid spacing from frequency domain peaks.
"""

import logging
from typing import Any

import cv2
import numpy as np
from scipy.fft import fft2, fftshift
from scipy.ndimage import maximum_filter

from qontinui.discovery.region_analysis.base import (
    BaseRegionAnalyzer,
    BoundingBox,
    DetectedRegion,
    RegionType,
)

logger = logging.getLogger(__name__)


class FrequencyAnalysisDetector(BaseRegionAnalyzer):
    """
    Detects inventory grids using frequency analysis (FFT)

    Algorithm:
    1. Apply 2D FFT to image
    2. Identify strong frequency components (periodic patterns)
    3. Extract grid spacing from frequency peaks
    4. Locate grid region using inverse FFT or spatial filtering
    5. Validate and extract grid structure
    """

    @property
    def name(self) -> str:
        return "frequency_analysis_detector"

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "min_cell_size": 24,
            "max_cell_size": 150,
            "frequency_threshold": 0.1,
            "peak_threshold": 0.3,
            "min_grid_rows": 2,
            "min_grid_cols": 2,
            "use_edges": True,
        }

    def analyze(self, image: np.ndarray, **kwargs) -> list[DetectedRegion]:  # type: ignore[override]
        """Detect inventory grids using frequency analysis"""
        params = {**self.get_default_parameters(), **kwargs}

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Optionally use edges for better frequency detection
        if params["use_edges"]:
            edges = cv2.Canny(gray, 50, 150)
            analysis_img = edges
        else:
            analysis_img = gray

        # Perform frequency analysis
        grid_params = self._frequency_analysis(analysis_img, params)

        if not grid_params:
            return []

        # Locate grid regions using the detected parameters
        regions = self._locate_grid_regions(analysis_img, gray, grid_params, params)

        return regions

    def _frequency_analysis(
        self, image: np.ndarray, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Perform 2D FFT to detect periodic patterns

        Returns:
            Dictionary with detected grid spacing
        """
        # Normalize image
        img_float = image.astype(np.float32) / 255.0

        # Apply window to reduce edge effects
        h, w = img_float.shape
        window_y = np.hanning(h).reshape(-1, 1)
        window_x = np.hanning(w).reshape(1, -1)
        window = window_y * window_x
        img_windowed = img_float * window

        # Compute 2D FFT
        f = fft2(img_windowed)
        fshift = fftshift(f)

        # Magnitude spectrum
        magnitude = np.abs(fshift)
        magnitude = np.log1p(magnitude)  # Log scale for better visualization

        # Normalize
        magnitude = magnitude / magnitude.max()

        # Find peaks in frequency domain
        # Exclude DC component (center)
        center_y, center_x = h // 2, w // 2
        mask_size = 10
        magnitude[
            center_y - mask_size : center_y + mask_size,
            center_x - mask_size : center_x + mask_size,
        ] = 0

        # Find local maxima
        local_max = maximum_filter(magnitude, size=10)
        peaks = (magnitude == local_max) & (magnitude > params["peak_threshold"])

        # Get peak coordinates
        peak_coords = np.argwhere(peaks)

        if len(peak_coords) < 2:
            return None

        # Calculate frequencies from peak positions
        # Distance from center corresponds to frequency
        frequencies = []
        for y, x in peak_coords:
            fy = (y - center_y) / h
            fx = (x - center_x) / w

            # Calculate period (spacing) from frequency
            if abs(fx) > 1e-6:
                period_x = 1 / abs(fx)
                if params["min_cell_size"] <= period_x <= params["max_cell_size"]:
                    frequencies.append(("x", period_x, magnitude[y, x]))

            if abs(fy) > 1e-6:
                period_y = 1 / abs(fy)
                if params["min_cell_size"] <= period_y <= params["max_cell_size"]:
                    frequencies.append(("y", period_y, magnitude[y, x]))

        if not frequencies:
            return None

        # Find dominant frequencies
        x_frequencies = [
            (period, strength) for axis, period, strength in frequencies if axis == "x"
        ]
        y_frequencies = [
            (period, strength) for axis, period, strength in frequencies if axis == "y"
        ]

        if not x_frequencies or not y_frequencies:
            return None

        # Sort by strength and take strongest
        x_frequencies.sort(key=lambda x: x[1], reverse=True)
        y_frequencies.sort(key=lambda x: x[1], reverse=True)

        spacing_x = int(x_frequencies[0][0])
        spacing_y = int(y_frequencies[0][0])

        return {
            "spacing_x": spacing_x,
            "spacing_y": spacing_y,
            "cell_width": spacing_x,
            "cell_height": spacing_y,
        }

    def _locate_grid_regions(
        self,
        edge_img: np.ndarray,
        gray_img: np.ndarray,
        grid_params: dict[str, Any],
        params: dict[str, Any],
    ) -> list[DetectedRegion]:
        """Locate actual grid regions using detected spacing"""
        spacing_x = grid_params["spacing_x"]
        spacing_y = grid_params["spacing_y"]

        # Create template for grid cell
        template_size_x = min(spacing_x, 64)
        template_size_y = min(spacing_y, 64)

        # Use edge-based template
        template = np.zeros((template_size_y, template_size_x), dtype=np.uint8)
        cv2.rectangle(template, (2, 2), (template_size_x - 3, template_size_y - 3), 255, 2)

        # Template matching
        if edge_img.shape[0] >= template_size_y and edge_img.shape[1] >= template_size_x:
            result = cv2.matchTemplate(edge_img, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.3

            locations = np.where(result >= threshold)
            matches = list(zip(locations[1], locations[0], strict=False))  # (x, y)

            if len(matches) < params["min_grid_rows"] * params["min_grid_cols"]:
                return []

            # Cluster matches into grid
            grid_regions = self._extract_grid_from_matches(
                matches,
                spacing_x,
                spacing_y,
                gray_img.shape,  # type: ignore[arg-type]
                params,
            )

            return grid_regions

        return []

    def _extract_grid_from_matches(
        self,
        matches: list[tuple[int, int]],
        spacing_x: int,
        spacing_y: int,
        img_shape: tuple[int, int],
        params: dict[str, Any],
    ) -> list[DetectedRegion]:
        """Extract grid structure from template matches"""
        if len(matches) < params["min_grid_rows"] * params["min_grid_cols"]:
            return []

        matches_array = np.array(matches)
        x_min, y_min = matches_array.min(axis=0)

        # Cluster into grid positions
        grid_positions = {}

        for x, y in matches:
            grid_x = round((x - x_min) / spacing_x)
            grid_y = round((y - y_min) / spacing_y)

            # Check if close to grid position
            expected_x = x_min + grid_x * spacing_x
            expected_y = y_min + grid_y * spacing_y

            if abs(x - expected_x) < spacing_x * 0.3 and abs(y - expected_y) < spacing_y * 0.3:
                if (grid_x, grid_y) not in grid_positions:
                    grid_positions[(grid_x, grid_y)] = (x, y)

        if len(grid_positions) < params["min_grid_rows"] * params["min_grid_cols"]:
            return []

        # Extract grid dimensions
        grid_coords = list(grid_positions.keys())
        grid_xs = [pos[0] for pos in grid_coords]
        grid_ys = [pos[1] for pos in grid_coords]

        cols = max(grid_xs) - min(grid_xs) + 1
        rows = max(grid_ys) - min(grid_ys) + 1

        if rows < params["min_grid_rows"] or cols < params["min_grid_cols"]:
            return []

        # Calculate bounding box
        x_start = int(x_min)
        y_start = int(y_min)
        width = cols * spacing_x + spacing_x
        height = rows * spacing_y + spacing_y

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

        # Calculate confidence
        expected_cells = rows * cols
        actual_cells = len(grid_positions)
        confidence = min(0.95, 0.6 + (actual_cells / expected_cells) * 0.35)

        region = DetectedRegion(
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
                "detector": "frequency_analysis",
                "method": "fft",
            },
        )

        return [region]
