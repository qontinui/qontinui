"""
Sidebar Detector

Detects sidebar navigation elements typically found on left or right edges.
Characteristics:
- Vertical edge regions (left or right 150-250px)
- Vertically stacked clickable items
- Icon + text combinations
- Common in modern web applications
"""

import logging
from io import BytesIO
from typing import Any, Dict, List, Tuple

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


class SidebarDetector(BaseAnalyzer):
    """
    Detects sidebar navigation elements

    Algorithm:
    1. Focus on left/right edge regions (150-250px)
    2. Find vertically stacked elements
    3. Detect icon + text patterns
    4. Look for consistent spacing and alignment
    5. Verify vertical extent (should span significant height)
    """

    @property
    def analysis_type(self) -> AnalysisType:
        return AnalysisType.SINGLE_SHOT

    @property
    def name(self) -> str:
        return "sidebar_detector"

    @property
    def supports_multi_screenshot(self) -> bool:
        return True

    @property
    def required_screenshots(self) -> int:
        return 1

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "min_sidebar_width": 150,
            "max_sidebar_width": 350,
            "min_height_ratio": 0.5,  # Sidebar should span at least 50% of screen height
            "edge_region_width": 300,  # Width of edge region to analyze
            "min_items": 3,  # Minimum number of stacked items
            "edge_threshold_low": 30,
            "edge_threshold_high": 100,
            "check_both_edges": True,  # Check both left and right edges
        }

    async def analyze(self, input_data: AnalysisInput) -> AnalysisResult:
        """Detect sidebars in screenshots"""
        logger.info(
            f"Running sidebar detection on "
            f"{len(input_data.screenshots)} screenshots"
        )

        params = {**self.get_default_parameters(), **input_data.parameters}

        # Load images
        images_color = self._load_images_color(input_data.screenshot_data)
        images_gray = self._load_images_grayscale(input_data.screenshot_data)

        # Analyze each screenshot
        all_elements = []
        for screenshot_idx, (img_color, img_gray) in enumerate(
            zip(images_color, images_gray)
        ):
            elements = await self._analyze_screenshot(
                img_color, img_gray, screenshot_idx, params
            )
            all_elements.extend(elements)

        logger.info(f"Detected {len(all_elements)} sidebars")

        return AnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            elements=all_elements,
            confidence=0.76,
            metadata={
                "num_screenshots": len(images_color),
                "method": "sidebar_detection",
                "parameters": params,
            },
        )

    def _load_images_color(self, screenshot_data: List[bytes]) -> List[np.ndarray]:
        """Load screenshots in color"""
        images = []
        for data in screenshot_data:
            img = Image.open(BytesIO(data)).convert("RGB")
            images.append(np.array(img, dtype=np.uint8))
        return images

    def _load_images_grayscale(self, screenshot_data: List[bytes]) -> List[np.ndarray]:
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
        params: Dict[str, Any],
    ) -> List[DetectedElement]:
        """Analyze a single screenshot for sidebars"""
        elements = []

        height, width = img_gray.shape

        # Check left edge
        left_sidebar = self._detect_sidebar_in_region(
            img_gray, 0, params["edge_region_width"], "left", params
        )
        if left_sidebar:
            bbox, confidence, metadata = left_sidebar
            elements.append(
                DetectedElement(
                    bounding_box=bbox,
                    confidence=confidence,
                    label="Sidebar (Left)",
                    element_type="sidebar",
                    screenshot_index=screenshot_idx,
                    metadata=metadata,
                )
            )

        # Check right edge if requested
        if params["check_both_edges"]:
            right_start = max(0, width - params["edge_region_width"])
            right_sidebar = self._detect_sidebar_in_region(
                img_gray, right_start, width, "right", params
            )
            if right_sidebar:
                bbox, confidence, metadata = right_sidebar
                elements.append(
                    DetectedElement(
                        bounding_box=bbox,
                        confidence=confidence,
                        label="Sidebar (Right)",
                        element_type="sidebar",
                        screenshot_index=screenshot_idx,
                        metadata=metadata,
                    )
                )

        return elements

    def _detect_sidebar_in_region(
        self,
        img_gray: np.ndarray,
        x_start: int,
        x_end: int,
        side: str,
        params: Dict[str, Any],
    ) -> Tuple[BoundingBox, float, Dict[str, Any]] | None:
        """Detect sidebar in a specific edge region"""
        height, width = img_gray.shape

        # Extract edge region
        edge_region = img_gray[:, x_start:x_end]
        region_width = x_end - x_start

        # Apply edge detection
        edges = cv2.Canny(
            edge_region, params["edge_threshold_low"], params["edge_threshold_high"]
        )

        # Find vertical line (sidebar boundary)
        # Look for strong vertical edges
        vertical_projection = np.sum(edges, axis=0)

        # Find the strongest vertical line (likely sidebar boundary)
        if side == "left":
            # Look for boundary in right portion of edge region
            search_start = min(params["min_sidebar_width"], region_width // 2)
            search_end = min(params["max_sidebar_width"], region_width)
            search_region = vertical_projection[search_start:search_end]
            if len(search_region) == 0:
                return None
            boundary_offset = search_start + np.argmax(search_region)
            sidebar_width = boundary_offset
            sidebar_x = x_start
        else:  # right
            # Look for boundary in left portion of edge region
            search_start = max(0, region_width - params["max_sidebar_width"])
            search_end = max(0, region_width - params["min_sidebar_width"])
            search_region = vertical_projection[search_start:search_end]
            if len(search_region) == 0:
                return None
            boundary_offset = search_start + np.argmax(search_region)
            sidebar_width = region_width - boundary_offset
            sidebar_x = x_start + boundary_offset

        # Validate sidebar width
        if not (
            params["min_sidebar_width"] <= sidebar_width <= params["max_sidebar_width"]
        ):
            return None

        # Extract sidebar region
        if side == "left":
            sidebar_region = edge_region[:, :sidebar_width]
        else:
            sidebar_region = edge_region[:, -sidebar_width:]

        # Detect vertically stacked items
        items = self._detect_stacked_items(sidebar_region)

        # Must have minimum number of items
        if len(items) < params["min_items"]:
            return None

        # Calculate vertical extent
        if len(items) > 0:
            first_item = items[0]
            last_item = items[-1]
            vertical_extent = last_item - first_item
        else:
            vertical_extent = 0

        # Check if sidebar spans enough vertical space
        min_height = height * params["min_height_ratio"]
        if vertical_extent < min_height:
            return None

        # Calculate confidence
        confidence = self._calculate_confidence(
            sidebar_width, vertical_extent, height, len(items), params
        )

        if confidence < 0.5:
            return None

        bbox = BoundingBox(
            x=int(sidebar_x), y=0, width=int(sidebar_width), height=int(height)
        )

        metadata = {
            "method": "sidebar_detection",
            "side": side,
            "num_items": len(items),
            "vertical_extent_ratio": float(vertical_extent / height),
        }

        return (bbox, confidence, metadata)

    def _detect_stacked_items(self, sidebar_region: np.ndarray) -> List[int]:
        """Detect vertically stacked items in sidebar"""
        # Find horizontal projection (sum edges across width)
        horizontal_projection = np.sum(sidebar_region, axis=1)

        # Smooth the projection
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(horizontal_projection, kernel, mode="same")

        # Find peaks (item rows)
        threshold = np.mean(smoothed) * 0.8
        item_rows = smoothed > threshold

        # Find transitions (item boundaries)
        transitions = np.diff(item_rows.astype(int))
        starts = np.where(transitions == 1)[0]

        return list(starts)

    def _calculate_confidence(
        self,
        sidebar_width: int,
        vertical_extent: int,
        total_height: int,
        num_items: int,
        params: Dict[str, Any],
    ) -> float:
        """Calculate confidence score"""
        confidence = 0.5  # Base confidence

        # Width in typical range (180-250px)
        if 180 <= sidebar_width <= 250:
            confidence += 0.15
        elif (
            params["min_sidebar_width"] <= sidebar_width <= params["max_sidebar_width"]
        ):
            confidence += 0.08

        # Spans most of screen height
        height_ratio = vertical_extent / total_height
        if height_ratio >= 0.8:
            confidence += 0.2
        elif height_ratio >= 0.6:
            confidence += 0.12

        # Multiple stacked items
        if num_items >= 6:
            confidence += 0.15
        elif num_items >= 4:
            confidence += 0.1
        elif num_items >= 3:
            confidence += 0.05

        return min(0.90, confidence)
