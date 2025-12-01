"""
Dropdown Detector

Detects dropdown/select elements using template matching for down-arrow icons.
Characteristics:
- Rectangular regions with down-arrow icon (▼)
- Template matching for dropdown arrows
- Expand/collapse states in multi-screenshot
- Option lists when expanded
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


class DropdownDetector(BaseAnalyzer):
    """
    Detects dropdown/select elements using arrow icon detection

    Algorithm:
    1. Find rectangles with down-arrow icons
    2. Template match for common dropdown arrow patterns
    3. Detect expand/collapse states in multi-screenshot
    4. Look for option lists when expanded
    """

    @property
    def analysis_type(self) -> AnalysisType:
        return AnalysisType.PATTERN_MATCH

    @property
    def name(self) -> str:
        return "dropdown_detector"

    @property
    def supports_multi_screenshot(self) -> bool:
        return True

    @property
    def required_screenshots(self) -> int:
        return 1

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "min_width": 80,
            "max_width": 400,
            "min_height": 25,
            "max_height": 50,
            "arrow_template_threshold": 0.6,  # Template matching threshold
            "edge_threshold_low": 50,
            "edge_threshold_high": 150,
            "detect_expanded_state": True,
        }

    async def analyze(self, input_data: AnalysisInput) -> AnalysisResult:
        """Detect dropdown elements in screenshots"""
        logger.info(
            f"Running dropdown detection on "
            f"{len(input_data.screenshots)} screenshots"
        )

        params = {**self.get_default_parameters(), **input_data.parameters}

        # Load images
        images_color = self._load_images_color(input_data.screenshot_data)
        images_gray = self._load_images_grayscale(input_data.screenshot_data)

        # Create arrow templates
        arrow_templates = self._create_arrow_templates()

        # Analyze each screenshot
        all_elements = []
        for screenshot_idx, (img_color, img_gray) in enumerate(
            zip(images_color, images_gray, strict=False)
        ):
            elements = await self._analyze_screenshot(
                img_color, img_gray, screenshot_idx, arrow_templates, params
            )
            all_elements.extend(elements)

        logger.info(f"Detected {len(all_elements)} dropdown elements")

        return AnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            elements=all_elements,
            confidence=0.75,
            metadata={
                "num_screenshots": len(images_color),
                "method": "dropdown_detection",
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

    def _create_arrow_templates(self) -> list[np.ndarray]:
        """Create templates for dropdown arrows"""
        templates = []

        # Create simple down-arrow template (▼)
        # Size variations: 8x8, 12x12, 16x16
        for size in [8, 12, 16]:
            template = np.zeros((size, size), dtype=np.uint8)
            # Draw simple triangle pointing down
            for i in range(size // 2):
                start = i
                end = size - i
                y = i
                if y < size:
                    template[y, start:end] = 255
            templates.append(template)

        return templates

    async def _analyze_screenshot(
        self,
        img_color: np.ndarray,
        img_gray: np.ndarray,
        screenshot_idx: int,
        arrow_templates: list[np.ndarray],
        params: dict[str, Any],
    ) -> list[DetectedElement]:
        """Analyze a single screenshot for dropdowns"""
        elements = []
        detected_locations = set()  # To avoid duplicates

        # Template matching for dropdown arrows
        for template in arrow_templates:
            matches = self._template_match_arrows(
                img_gray, template, params["arrow_template_threshold"]
            )

            for x, y, w, h, confidence in matches:
                # Expand bounding box to include the full dropdown
                # Arrow is typically on the right side
                dropdown_x = max(0, x - 150)  # Expand left
                dropdown_y = max(0, y - 5)  # Small vertical expansion
                dropdown_w = min(params["max_width"], x - dropdown_x + w + 10)
                dropdown_h = min(params["max_height"], h + 10)

                # Check if already detected
                location_key = (
                    dropdown_x // 10,
                    dropdown_y // 10,
                )  # Grid-based deduplication
                if location_key in detected_locations:
                    continue
                detected_locations.add(location_key)

                # Validate size
                if not (params["min_width"] <= dropdown_w <= params["max_width"]):
                    continue
                if not (params["min_height"] <= dropdown_h <= params["max_height"]):
                    continue

                elements.append(
                    DetectedElement(
                        bounding_box=BoundingBox(
                            x=int(dropdown_x),
                            y=int(dropdown_y),
                            width=int(dropdown_w),
                            height=int(dropdown_h),
                        ),
                        confidence=confidence,
                        label="Dropdown",
                        element_type="dropdown",
                        screenshot_index=screenshot_idx,
                        metadata={
                            "method": "template_matching",
                            "arrow_detected": True,
                            "template_size": template.shape[0],
                        },
                    )
                )

        # Also detect by shape (rectangular with right-side button-like region)
        shape_dropdowns = self._detect_by_shape(img_gray, params)
        for bbox, conf in shape_dropdowns:
            # Check for duplicates
            location_key = (bbox.x // 10, bbox.y // 10)
            if location_key not in detected_locations:
                detected_locations.add(location_key)
                elements.append(
                    DetectedElement(
                        bounding_box=bbox,
                        confidence=conf,
                        label="Dropdown",
                        element_type="dropdown",
                        screenshot_index=screenshot_idx,
                        metadata={
                            "method": "shape_based",
                            "arrow_detected": False,
                        },
                    )
                )

        return elements

    def _template_match_arrows(
        self, img_gray: np.ndarray, template: np.ndarray, threshold: float
    ) -> list[tuple[int, int, int, int, float]]:
        """Template match for dropdown arrows"""
        matches = []

        # Perform template matching
        result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

        # Find locations above threshold
        locations = np.where(result >= threshold)

        for pt in zip(*locations[::-1], strict=False):
            x, y = pt
            w, h = template.shape[::-1]
            confidence = float(result[y, x])
            matches.append((x, y, w, h, confidence))

        # Non-maximum suppression
        matches = self._non_max_suppression(matches, 0.3)

        return matches

    def _non_max_suppression(
        self, matches: list[tuple[int, int, int, int, float]], iou_threshold: float
    ) -> list[tuple[int, int, int, int, float]]:
        """Remove overlapping detections"""
        if not matches:
            return []

        # Sort by confidence
        matches = sorted(matches, key=lambda x: x[4], reverse=True)

        keep = []
        while matches:
            current = matches.pop(0)
            keep.append(current)

            # Remove overlapping matches
            matches = [
                m for m in matches if not self._boxes_overlap(current, m, iou_threshold)
            ]

        return keep

    def _boxes_overlap(
        self,
        box1: tuple[int, int, int, int, float],
        box2: tuple[int, int, int, int, float],
        threshold: float,
    ) -> bool:
        """Check if two boxes overlap significantly"""
        x1, y1, w1, h1, _ = box1
        x2, y2, w2, h2, _ = box2

        # Calculate IoU
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return False

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        iou = intersection / union if union > 0 else 0
        return iou >= threshold

    def _detect_by_shape(
        self, img_gray: np.ndarray, params: dict[str, Any]
    ) -> list[tuple[BoundingBox, float]]:
        """Detect dropdowns by their rectangular shape"""
        dropdowns = []

        # Apply edge detection
        edges = cv2.Canny(
            img_gray, params["edge_threshold_low"], params["edge_threshold_high"]
        )

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size
            if not (params["min_width"] <= w <= params["max_width"]):
                continue
            if not (params["min_height"] <= h <= params["max_height"]):
                continue

            # Dropdowns typically have aspect ratio 2:1 to 8:1
            aspect_ratio = w / h if h > 0 else 0
            if not (2.0 <= aspect_ratio <= 8.0):
                continue

            # Lower confidence for shape-only detection
            confidence = 0.55

            dropdowns.append((BoundingBox(x=x, y=y, width=w, height=h), confidence))

        return dropdowns
