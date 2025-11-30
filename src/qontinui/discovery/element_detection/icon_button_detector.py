"""
Icon Button Detector

Detects icon-only buttons using template matching and size filtering.
Characteristics:
- Template matching for common icons (☰, ⚙, 🔍, ❌, ⋮)
- Size filtering (16x16 to 64x64)
- Square or circular regions
- Often found in toolbars or headers
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


class IconButtonDetector(BaseAnalyzer):
    """
    Detects icon-only buttons using template matching

    Algorithm:
    1. Create templates for common icons
    2. Template match across screenshots
    3. Filter by size (16x16 to 64x64)
    4. Look for square or circular regions
    5. Check position (often in headers/toolbars)
    """

    @property
    def analysis_type(self) -> AnalysisType:
        return AnalysisType.PATTERN_MATCH

    @property
    def name(self) -> str:
        return "icon_button_detector"

    @property
    def supports_multi_screenshot(self) -> bool:
        return True

    @property
    def required_screenshots(self) -> int:
        return 1

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "min_size": 16,
            "max_size": 64,
            "template_threshold": 0.65,
            "aspect_ratio_tolerance": 0.3,  # How much deviation from 1:1 to allow
            "detect_in_header": True,  # Prioritize header region
            "header_height": 100,
        }

    async def analyze(self, input_data: AnalysisInput) -> AnalysisResult:
        """Detect icon buttons in screenshots"""
        logger.info(
            f"Running icon button detection on " f"{len(input_data.screenshots)} screenshots"
        )

        params = {**self.get_default_parameters(), **input_data.parameters}

        # Load images
        images_color = self._load_images_color(input_data.screenshot_data)
        images_gray = self._load_images_grayscale(input_data.screenshot_data)

        # Create icon templates
        icon_templates = self._create_icon_templates()

        # Analyze each screenshot
        all_elements = []
        for screenshot_idx, (img_color, img_gray) in enumerate(
            zip(images_color, images_gray, strict=False)
        ):
            elements = await self._analyze_screenshot(
                img_color, img_gray, screenshot_idx, icon_templates, params
            )
            all_elements.extend(elements)

        logger.info(f"Detected {len(all_elements)} icon buttons")

        return AnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            elements=all_elements,
            confidence=0.74,
            metadata={
                "num_screenshots": len(images_color),
                "method": "icon_button_detection",
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

    def _create_icon_templates(self) -> dict[str, list[np.ndarray]]:
        """Create templates for common icon types"""
        templates = {}

        # Hamburger menu icon (☰)
        templates["hamburger"] = self._create_hamburger_templates()

        # Settings/gear icon (⚙)
        templates["settings"] = self._create_settings_templates()

        # Search/magnifying glass icon (🔍)
        templates["search"] = self._create_search_templates()

        # Close/X icon (❌)
        templates["close"] = self._create_close_templates()

        # More/three dots icon (⋮)
        templates["more"] = self._create_more_templates()

        return templates

    def _create_hamburger_templates(self) -> list[np.ndarray]:
        """Create hamburger menu icon templates"""
        templates = []

        for size in [20, 24, 32]:
            template = np.zeros((size, size), dtype=np.uint8)
            bar_height = max(2, size // 10)
            spacing = size // 4

            # Draw three horizontal bars
            y1 = spacing
            y2 = size // 2
            y3 = size - spacing - bar_height

            template[y1 : y1 + bar_height, 4 : size - 4] = 255
            template[y2 - bar_height // 2 : y2 + bar_height // 2, 4 : size - 4] = 255
            template[y3 : y3 + bar_height, 4 : size - 4] = 255

            templates.append(template)

        return templates

    def _create_settings_templates(self) -> list[np.ndarray]:
        """Create settings/gear icon templates"""
        templates = []

        for size in [20, 24, 32]:
            template = np.zeros((size, size), dtype=np.uint8)
            center = size // 2
            radius = size // 3

            # Draw a simplified gear (circle with teeth)
            cv2.circle(template, (center, center), radius, 255, 2)  # type: ignore[call-overload]

            # Add 6 teeth around the circle
            for angle in range(0, 360, 60):
                rad = np.radians(angle)
                x = int(center + radius * 1.3 * np.cos(rad))
                y = int(center + radius * 1.3 * np.sin(rad))
                cv2.line(template, (center, center), (x, y), 255, 2)  # type: ignore[call-overload]

            templates.append(template)

        return templates

    def _create_search_templates(self) -> list[np.ndarray]:
        """Create search/magnifying glass icon templates"""
        templates = []

        for size in [20, 24, 32]:
            template = np.zeros((size, size), dtype=np.uint8)
            center = size // 3
            radius = size // 4

            # Draw circle (lens)
            cv2.circle(template, (center, center), radius, 255, 2)  # type: ignore[call-overload]

            # Draw handle
            handle_start = (int(center + radius * 0.7), int(center + radius * 0.7))
            handle_end = (size - 4, size - 4)
            cv2.line(template, handle_start, handle_end, 255, 2)  # type: ignore[call-overload]

            templates.append(template)

        return templates

    def _create_close_templates(self) -> list[np.ndarray]:
        """Create close/X icon templates"""
        templates = []

        for size in [16, 20, 24, 32]:
            template = np.zeros((size, size), dtype=np.uint8)
            margin = size // 4

            # Draw X
            cv2.line(template, (margin, margin), (size - margin, size - margin), 255, 2)  # type: ignore[call-overload]
            cv2.line(template, (size - margin, margin), (margin, size - margin), 255, 2)  # type: ignore[call-overload]

            templates.append(template)

        return templates

    def _create_more_templates(self) -> list[np.ndarray]:
        """Create more/three dots icon templates"""
        templates = []

        for size in [20, 24, 32]:
            template = np.zeros((size, size), dtype=np.uint8)
            center_x = size // 2
            dot_radius = max(2, size // 10)

            # Draw three vertical dots
            y1 = size // 4
            y2 = size // 2
            y3 = 3 * size // 4

            cv2.circle(template, (center_x, y1), dot_radius, 255, -1)  # type: ignore[call-overload]
            cv2.circle(template, (center_x, y2), dot_radius, 255, -1)  # type: ignore[call-overload]
            cv2.circle(template, (center_x, y3), dot_radius, 255, -1)  # type: ignore[call-overload]

            templates.append(template)

        return templates

    async def _analyze_screenshot(
        self,
        img_color: np.ndarray,
        img_gray: np.ndarray,
        screenshot_idx: int,
        icon_templates: dict[str, list[np.ndarray]],
        params: dict[str, Any],
    ) -> list[DetectedElement]:
        """Analyze a single screenshot for icon buttons"""
        elements = []
        detected_locations = set()

        # Template matching for each icon type
        for icon_type, templates in icon_templates.items():
            for template in templates:
                matches = self._template_match_icons(
                    img_gray, template, params["template_threshold"]
                )

                for x, y, w, h, confidence in matches:
                    # Size filtering
                    if not (params["min_size"] <= w <= params["max_size"]):
                        continue
                    if not (params["min_size"] <= h <= params["max_size"]):
                        continue

                    # Check aspect ratio (should be roughly square)
                    aspect_ratio = w / h if h > 0 else 0
                    if abs(aspect_ratio - 1.0) > params["aspect_ratio_tolerance"]:
                        continue

                    # Check for duplicates
                    location_key = (x // 5, y // 5)
                    if location_key in detected_locations:
                        continue
                    detected_locations.add(location_key)

                    # Boost confidence if in header region
                    final_confidence = confidence
                    if params["detect_in_header"] and y < params["header_height"]:
                        final_confidence = min(0.95, confidence + 0.1)

                    elements.append(
                        DetectedElement(
                            bounding_box=BoundingBox(
                                x=int(x), y=int(y), width=int(w), height=int(h)
                            ),
                            confidence=final_confidence,
                            label=f"{icon_type.title()} Icon",
                            element_type="icon_button",
                            screenshot_index=screenshot_idx,
                            metadata={
                                "method": "template_matching",
                                "icon_type": icon_type,
                                "in_header": y < params["header_height"],
                            },
                        )
                    )

        # Also detect by shape (small square regions with high edge density)
        shape_icons = self._detect_by_shape(img_gray, params)
        for bbox, conf in shape_icons:
            location_key = (bbox.x // 5, bbox.y // 5)
            if location_key not in detected_locations:
                detected_locations.add(location_key)
                elements.append(
                    DetectedElement(
                        bounding_box=bbox,
                        confidence=conf,
                        label="Icon Button",
                        element_type="icon_button",
                        screenshot_index=screenshot_idx,
                        metadata={
                            "method": "shape_based",
                            "icon_type": "unknown",
                        },
                    )
                )

        return elements

    def _template_match_icons(
        self, img_gray: np.ndarray, template: np.ndarray, threshold: float
    ) -> list[tuple[int, int, int, int, float]]:
        """Template match for icons"""
        matches = []

        # Perform template matching
        result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

        # Find locations above threshold
        locations = np.where(result >= threshold)

        for pt in zip(*locations[::-1], strict=False):
            x, y = pt
            h, w = template.shape
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

        matches = sorted(matches, key=lambda x: x[4], reverse=True)

        keep = []
        while matches:
            current = matches.pop(0)
            keep.append(current)
            matches = [m for m in matches if not self._boxes_overlap(current, m, iou_threshold)]

        return keep

    def _boxes_overlap(
        self,
        box1: tuple[int, int, int, int, float],
        box2: tuple[int, int, int, int, float],
        threshold: float,
    ) -> bool:
        """Check if two boxes overlap"""
        x1, y1, w1, h1, _ = box1
        x2, y2, w2, h2, _ = box2

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
        """Detect icon buttons by shape (small, square regions)"""
        icons = []

        # Apply edge detection
        edges = cv2.Canny(img_gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Size filtering
            if not (params["min_size"] <= w <= params["max_size"]):
                continue
            if not (params["min_size"] <= h <= params["max_size"]):
                continue

            # Must be roughly square
            aspect_ratio = w / h if h > 0 else 0
            if abs(aspect_ratio - 1.0) > params["aspect_ratio_tolerance"]:
                continue

            # Lower confidence for shape-only detection
            confidence = 0.60

            icons.append((BoundingBox(x=x, y=y, width=w, height=h), confidence))

        return icons
