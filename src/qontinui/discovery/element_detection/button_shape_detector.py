"""
Button Shape Detector

Detects buttons using shape-based analysis:
- Contour detection for rectangles
- Aspect ratio filtering (2:1 to 5:1 typical for buttons)
- Size filtering (80-300px wide, 30-60px tall)
- Rounded corner detection
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


class ButtonShapeDetector(BaseAnalyzer):
    """
    Detects buttons using shape-based computer vision

    Algorithm:
    1. Convert to grayscale and apply edge detection
    2. Find contours representing closed shapes
    3. Filter by size (typical button dimensions)
    4. Filter by aspect ratio (buttons are wider than tall)
    5. Detect rounded corners using corner detection
    6. Score based on shape regularity
    """

    @property
    def analysis_type(self) -> AnalysisType:
        return AnalysisType.CUSTOM

    @property
    def name(self) -> str:
        return "button_shape_detector"

    @property
    def supports_multi_screenshot(self) -> bool:
        return True

    @property
    def required_screenshots(self) -> int:
        return 1

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            # Size constraints (typical button dimensions)
            "min_width": 80,
            "max_width": 300,
            "min_height": 30,
            "max_height": 60,
            # Aspect ratio constraints (width/height)
            "min_aspect_ratio": 2.0,  # 2:1 - button is wider than tall
            "max_aspect_ratio": 5.0,  # 5:1 - not too elongated
            # Edge detection parameters
            "canny_low": 50,
            "canny_high": 150,
            "morph_kernel_size": 3,
            # Shape filtering
            "min_rectangularity": 0.7,  # How rectangular the shape is
            "detect_rounded_corners": True,
            "corner_radius_threshold": 5,  # Minimum corner radius in pixels
            # Confidence thresholds
            "min_confidence": 0.5,
        }

    async def analyze(self, input_data: AnalysisInput) -> AnalysisResult:
        """Perform shape-based button detection"""
        logger.info(f"Running button shape detection on {len(input_data.screenshots)} screenshots")

        params = {**self.get_default_parameters(), **input_data.parameters}

        # Load images
        images_gray = self._load_images_grayscale(input_data.screenshot_data)
        images_color = self._load_images_color(input_data.screenshot_data)

        # Analyze each screenshot
        all_elements = []
        for screenshot_idx, (img_gray, img_color) in enumerate(zip(images_gray, images_color, strict=False)):
            elements = await self._analyze_screenshot(img_gray, img_color, screenshot_idx, params)
            all_elements.extend(elements)

        logger.info(f"Detected {len(all_elements)} button candidates using shape analysis")

        return AnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            elements=all_elements,
            confidence=0.78,  # Shape-based detection has good precision
            metadata={
                "num_screenshots": len(images_gray),
                "method": "shape_detection",
                "parameters": params,
                "detector_type": "button_shape",
            },
        )

    def _load_images_grayscale(self, screenshot_data: list[bytes]) -> list[np.ndarray]:
        """Load screenshots as grayscale"""
        images = []
        for data in screenshot_data:
            img = Image.open(BytesIO(data)).convert("L")
            images.append(np.array(img, dtype=np.uint8))
        return images

    def _load_images_color(self, screenshot_data: list[bytes]) -> list[np.ndarray]:
        """Load screenshots in color (BGR for OpenCV)"""
        images = []
        for data in screenshot_data:
            img = Image.open(BytesIO(data)).convert("RGB")
            # Convert RGB to BGR for OpenCV
            images.append(cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        return images

    async def _analyze_screenshot(
        self,
        img_gray: np.ndarray,
        img_color: np.ndarray,
        screenshot_idx: int,
        params: dict[str, Any],
    ) -> list[DetectedElement]:
        """Analyze a single screenshot for button shapes"""
        elements = []

        # Step 1: Apply Canny edge detection
        edges = cv2.Canny(img_gray, params["canny_low"], params["canny_high"])

        # Step 2: Dilate edges to close gaps
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (params["morph_kernel_size"], params["morph_kernel_size"]),
        )
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Step 3: Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        logger.debug(f"Found {len(contours)} contours in screenshot {screenshot_idx}")

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Step 4: Filter by size (typical button dimensions)
            if not (params["min_width"] <= w <= params["max_width"]):
                continue
            if not (params["min_height"] <= h <= params["max_height"]):
                continue

            # Step 5: Filter by aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if not (params["min_aspect_ratio"] <= aspect_ratio <= params["max_aspect_ratio"]):
                continue

            # Step 6: Calculate rectangularity (how close to a perfect rectangle)
            area = w * h
            contour_area = cv2.contourArea(contour)
            rectangularity = contour_area / area if area > 0 else 0

            if rectangularity < params["min_rectangularity"]:
                continue

            # Step 7: Detect rounded corners if requested
            has_rounded_corners = False
            corner_score = 0.0
            if params["detect_rounded_corners"]:
                has_rounded_corners, corner_score = self._detect_rounded_corners(
                    img_gray[y : y + h, x : x + w],
                    params["corner_radius_threshold"],
                )

            # Step 8: Calculate confidence score
            confidence = self._calculate_confidence(
                w,
                h,
                aspect_ratio,
                rectangularity,
                has_rounded_corners,
                corner_score,
                params,
            )

            if confidence < params["min_confidence"]:
                continue

            # Create detected element
            elements.append(
                DetectedElement(
                    bounding_box=BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h)),
                    confidence=confidence,
                    label="Button",
                    element_type="button",
                    screenshot_index=screenshot_idx,
                    metadata={
                        "method": "shape_detection",
                        "width": int(w),
                        "height": int(h),
                        "aspect_ratio": float(aspect_ratio),
                        "rectangularity": float(rectangularity),
                        "has_rounded_corners": has_rounded_corners,
                        "corner_score": float(corner_score),
                    },
                )
            )

        return elements

    def _detect_rounded_corners(
        self, region: np.ndarray, radius_threshold: int
    ) -> tuple[bool, float]:
        """
        Detect if a region has rounded corners

        Strategy:
        1. Use Harris corner detection to find corners
        2. Check if corners are inside the border (indicating rounded corners)
        3. Score based on corner density and position

        Returns:
            (has_rounded_corners, corner_score)
        """
        if region.size == 0 or region.shape[0] < 10 or region.shape[1] < 10:
            return False, 0.0

        h, w = region.shape[:2]

        # Harris corner detection
        corners = cv2.cornerHarris(region, blockSize=2, ksize=3, k=0.04)
        corners = cv2.dilate(corners, None)

        # Threshold to get strong corners
        corner_threshold = 0.01 * corners.max() if corners.max() > 0 else 0
        corner_points = np.where(corners > corner_threshold)

        if len(corner_points[0]) == 0:
            return False, 0.0

        # Check corners near edges (rounded corners would show up slightly inside)
        edge_margin = min(radius_threshold, min(h, w) // 4)
        corners_near_edges = 0

        for y_corner, x_corner in zip(corner_points[0], corner_points[1], strict=False):
            # Check if corner is near any edge but not exactly on it
            near_top = 0 < y_corner < edge_margin
            near_bottom = h - edge_margin < y_corner < h
            near_left = 0 < x_corner < edge_margin
            near_right = w - edge_margin < x_corner < w

            if (near_top or near_bottom) and (near_left or near_right):
                corners_near_edges += 1

        # Score based on number of rounded corners detected (expect 4)
        corner_score = min(1.0, corners_near_edges / 4.0)
        has_rounded_corners = corners_near_edges >= 2

        return has_rounded_corners, corner_score

    def _calculate_confidence(
        self,
        width: int,
        height: int,
        aspect_ratio: float,
        rectangularity: float,
        has_rounded_corners: bool,
        corner_score: float,
        params: dict[str, Any],
    ) -> float:
        """
        Calculate confidence score for button detection

        Factors:
        - Rectangularity: how close to a perfect rectangle
        - Aspect ratio: prefer typical button proportions (2.5:1 to 3:1)
        - Rounded corners: modern buttons often have rounded corners
        - Size: prefer medium sizes
        """
        confidence = 0.0

        # Base score from rectangularity (0.0 to 0.4)
        confidence += rectangularity * 0.4

        # Aspect ratio score (0.0 to 0.3)
        # Optimal aspect ratio is around 2.5 to 3.5
        optimal_ratio = 3.0
        ratio_diff = abs(aspect_ratio - optimal_ratio)
        ratio_score = max(0, 1.0 - (ratio_diff / 2.0))
        confidence += ratio_score * 0.3

        # Rounded corners bonus (0.0 to 0.2)
        if has_rounded_corners:
            confidence += corner_score * 0.2

        # Size appropriateness (0.0 to 0.1)
        # Prefer buttons around 150px wide and 40px tall
        width_score = 1.0 - abs(width - 150) / 150.0
        height_score = 1.0 - abs(height - 40) / 40.0
        size_score = (max(0, width_score) + max(0, height_score)) / 2.0
        confidence += size_score * 0.1

        return min(1.0, max(0.0, confidence))
