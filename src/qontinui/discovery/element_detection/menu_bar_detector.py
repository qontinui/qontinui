"""
Menu Bar Detector

Detects menu bars at the top of windows/applications.
Characteristics:
- Horizontal regions at top of screen (top 50-100px)
- Evenly-spaced text items
- Common menu items ("File", "Edit", "View", "Help")
- Hover effects may be visible
"""

import logging
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .analysis_base import (
    AnalysisInput,
    AnalysisResult,
    AnalysisType,
    BaseAnalyzer,
    BoundingBox,
    DetectedElement,
)

logger = logging.getLogger(__name__)


class MenuBarDetector(BaseAnalyzer):
    """
    Detects menu bars at the top of windows

    Algorithm:
    1. Focus on top region of screenshot (top 50-100px)
    2. Find horizontal regions with evenly-spaced text
    3. Detect text items using edge detection
    4. Look for menu bar characteristics (height, position, content)
    5. Score based on position and text distribution
    """

    @property
    def analysis_type(self) -> AnalysisType:
        return AnalysisType.SINGLE_SHOT

    @property
    def name(self) -> str:
        return "menu_bar_detector"

    @property
    def supports_multi_screenshot(self) -> bool:
        return True

    @property
    def required_screenshots(self) -> int:
        return 1

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "top_region_height": 100,  # Search in top N pixels
            "min_menu_height": 20,
            "max_menu_height": 60,
            "min_width_ratio": 0.3,  # Menu bar should span at least 30% of width
            "edge_threshold_low": 50,
            "edge_threshold_high": 150,
            "text_region_spacing_tolerance": 0.3,  # For evenly-spaced detection
        }

    async def analyze(self, input_data: AnalysisInput) -> AnalysisResult:
        """Detect menu bars in screenshots"""
        logger.info(
            f"Running menu bar detection on "
            f"{len(input_data.screenshots)} screenshots"
        )

        params = {**self.get_default_parameters(), **input_data.parameters}

        # Load images
        images_color = self._load_images_color(input_data.screenshot_data)
        images_gray = self._load_images_grayscale(input_data.screenshot_data)

        # Analyze each screenshot
        all_elements = []
        for screenshot_idx, (img_color, img_gray) in enumerate(
            zip(images_color, images_gray, strict=False)
        ):
            elements = await self._analyze_screenshot(
                img_color, img_gray, screenshot_idx, params
            )
            all_elements.extend(elements)

        logger.info(f"Detected {len(all_elements)} menu bars")

        return AnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            elements=all_elements,
            confidence=0.72,
            metadata={
                "num_screenshots": len(images_color),
                "method": "menu_bar_detection",
                "parameters": params,
            },
        )

    def _load_images_color(self, screenshot_data: list[bytes]) -> list[np.ndarray]:
        """Load screenshots in color"""
        images = []
        for data in screenshot_data:
            img = Image.open(BytesIO(data)).convert("RGB")
            images.append(np.array(img, dtype=np.uint8))
        return images

    def _load_images_grayscale(self, screenshot_data: list[bytes]) -> list[np.ndarray]:
        """Load screenshots as grayscale"""
        images = []
        for data in screenshot_data:
            img = Image.open(BytesIO(data)).convert("L")
            images.append(np.array(img, dtype=np.uint8))
        return images

    async def _analyze_screenshot(
        self,
        img_color: np.ndarray,
        img_gray: np.ndarray,
        screenshot_idx: int,
        params: dict[str, Any],
    ) -> list[DetectedElement]:
        """Analyze a single screenshot for menu bars"""
        elements: list[DetectedElement] = []

        height, width = img_gray.shape

        # Focus on top region
        top_region_h = min(params["top_region_height"], height // 3)
        top_region = img_gray[:top_region_h, :]

        # Apply edge detection
        edges = cv2.Canny(
            top_region, params["edge_threshold_low"], params["edge_threshold_high"]
        )

        # Find horizontal projections (sum edges across width)
        horizontal_projection = np.sum(edges, axis=1)

        # Find potential menu bar rows (high horizontal edge density)
        threshold = np.mean(horizontal_projection) + np.std(horizontal_projection)
        candidate_rows = np.where(horizontal_projection > threshold)[0]

        if len(candidate_rows) == 0:
            return elements

        # Group consecutive rows into regions
        regions = self._group_consecutive_rows(candidate_rows, params)

        # Analyze each potential menu bar region
        for y_start, y_end in regions:
            menu_height = y_end - y_start

            # Check height constraints
            if not (
                params["min_menu_height"] <= menu_height <= params["max_menu_height"]
            ):
                continue

            # Menu bars typically span most of the width
            menu_width = width
            menu_x = 0

            # Check if it spans enough width
            if menu_width < width * params["min_width_ratio"]:
                continue

            # Extract menu region for analysis
            menu_region = img_gray[y_start:y_end, menu_x : menu_x + menu_width]

            # Detect text items in the menu region
            text_items = self._detect_text_items(menu_region)

            # Menu bars should have multiple text items
            if len(text_items) < 2:
                continue

            # Check if text items are somewhat evenly spaced
            is_evenly_spaced = self._check_even_spacing(
                text_items, params["text_region_spacing_tolerance"]
            )

            # Calculate confidence
            confidence = self._calculate_confidence(
                y_start, menu_height, len(text_items), is_evenly_spaced, width
            )

            if confidence < 0.5:
                continue

            elements.append(
                DetectedElement(
                    bounding_box=BoundingBox(
                        x=int(menu_x),
                        y=int(y_start),
                        width=int(menu_width),
                        height=int(menu_height),
                    ),
                    confidence=confidence,
                    label="Menu Bar",
                    element_type="menu",
                    screenshot_index=screenshot_idx,
                    metadata={
                        "method": "menu_bar_detection",
                        "num_text_items": len(text_items),
                        "evenly_spaced": is_evenly_spaced,
                        "position": "top",
                    },
                )
            )

        return elements

    def _group_consecutive_rows(
        self, rows: np.ndarray, params: dict[str, Any]
    ) -> list[tuple]:
        """Group consecutive row indices into regions"""
        if len(rows) == 0:
            return []

        regions = []
        start = rows[0]
        prev = rows[0]

        for row in rows[1:]:
            if row - prev > 2:  # Gap detected
                regions.append((start, prev + 1))
                start = row
            prev = row

        regions.append((start, prev + 1))
        return regions

    def _detect_text_items(self, menu_region: np.ndarray) -> list[int]:
        """Detect text items in menu region by finding vertical gaps"""
        # Find vertical projection (sum across height)
        vertical_projection = np.sum(menu_region < 200, axis=0)  # Dark pixels

        # Smooth the projection
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(vertical_projection, kernel, mode="same")

        # Find peaks (text regions)
        threshold = np.mean(smoothed) * 0.5
        text_regions = smoothed > threshold

        # Find transitions (text item boundaries)
        transitions = np.diff(text_regions.astype(int))
        starts = np.where(transitions == 1)[0]

        return list(starts)

    def _check_even_spacing(self, text_items: list[int], tolerance: float) -> bool:
        """Check if text items are somewhat evenly spaced"""
        if len(text_items) < 3:
            return True  # Not enough items to check

        # Calculate gaps between items
        gaps = [text_items[i + 1] - text_items[i] for i in range(len(text_items) - 1)]

        if len(gaps) == 0:
            return False

        # Calculate variation in gaps
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)

        # Check if variation is within tolerance
        return bool((std_gap / mean_gap) <= tolerance) if mean_gap > 0 else False

    def _calculate_confidence(
        self,
        y_position: int,
        height: int,
        num_text_items: int,
        is_evenly_spaced: bool,
        screen_width: int,
    ) -> float:
        """Calculate confidence score"""
        confidence = 0.5  # Base confidence

        # Position at very top is strong indicator
        if y_position < 10:
            confidence += 0.2
        elif y_position < 30:
            confidence += 0.1

        # Height in typical range (25-40px)
        if 25 <= height <= 40:
            confidence += 0.15

        # Multiple text items
        if num_text_items >= 4:
            confidence += 0.15
        elif num_text_items >= 2:
            confidence += 0.08

        # Evenly spaced text items
        if is_evenly_spaced:
            confidence += 0.1

        return min(0.92, confidence)
