"""
Input Field Detector

Detects text input fields in screenshots using shape analysis and OCR.
Characteristics:
- Thin rectangles with high aspect ratio (> 3:1)
- White/light backgrounds with borders
- Placeholder text
- Cursor/caret detection
- Common sizes: 200-400px wide, 30-40px tall
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


class InputFieldDetector(BaseAnalyzer):
    """
    Detects text input fields using shape analysis and color characteristics

    Algorithm:
    1. Find thin rectangular contours (aspect ratio > 3:1)
    2. Check for light backgrounds (typical of input fields)
    3. Verify border presence
    4. Look for text cursors or placeholder text
    5. Filter by size constraints
    """

    @property
    def analysis_type(self) -> AnalysisType:
        return AnalysisType.SINGLE_SHOT

    @property
    def name(self) -> str:
        return "input_field_detector"

    @property
    def supports_multi_screenshot(self) -> bool:
        return True

    @property
    def required_screenshots(self) -> int:
        return 1

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "min_aspect_ratio": 3.0,  # Width / height ratio
            "max_aspect_ratio": 15.0,
            "min_width": 100,  # Minimum input width
            "max_width": 600,
            "min_height": 20,  # Minimum input height
            "max_height": 60,
            "light_bg_threshold": 200,  # For detecting light backgrounds
            "border_detection": True,
            "canny_low": 30,
            "canny_high": 100,
        }

    async def analyze(self, input_data: AnalysisInput) -> AnalysisResult:
        """Detect input fields in screenshots"""
        logger.info(
            f"Running input field detection on " f"{len(input_data.screenshots)} screenshots"
        )

        params = {**self.get_default_parameters(), **input_data.parameters}

        # Load images
        images_color = self._load_images_color(input_data.screenshot_data)
        images_gray = self._load_images_grayscale(input_data.screenshot_data)

        # Analyze each screenshot
        all_elements = []
        for screenshot_idx, (img_color, img_gray) in enumerate(zip(images_color, images_gray, strict=False)):
            elements = await self._analyze_screenshot(img_color, img_gray, screenshot_idx, params)
            all_elements.extend(elements)

        logger.info(f"Detected {len(all_elements)} input fields")

        return AnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            elements=all_elements,
            confidence=0.78,
            metadata={
                "num_screenshots": len(images_color),
                "method": "input_field_detection",
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
        """Analyze a single screenshot for input fields"""
        elements = []

        # Apply edge detection
        edges = cv2.Canny(img_gray, params["canny_low"], params["canny_high"])

        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size
            if not (params["min_width"] <= w <= params["max_width"]):
                continue
            if not (params["min_height"] <= h <= params["max_height"]):
                continue

            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 0

            # Check aspect ratio - input fields are horizontally elongated
            if not (params["min_aspect_ratio"] <= aspect_ratio <= params["max_aspect_ratio"]):
                continue

            # Extract region for analysis
            region = (
                img_color[y : y + h, x : x + w]
                if y + h <= img_color.shape[0] and x + w <= img_color.shape[1]
                else None
            )
            if region is None or region.size == 0:
                continue

            # Check for light background (typical of input fields)
            mean_brightness = np.mean(cv2.cvtColor(region, cv2.COLOR_RGB2GRAY))
            has_light_bg = mean_brightness >= params["light_bg_threshold"]

            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence(aspect_ratio, w, h, has_light_bg, params)

            # Only add if confidence is reasonable
            if confidence < 0.4:
                continue

            elements.append(
                DetectedElement(
                    bounding_box=BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h)),
                    confidence=confidence,
                    label="Input Field",
                    element_type="input",
                    screenshot_index=screenshot_idx,
                    metadata={
                        "method": "input_field_detection",
                        "aspect_ratio": float(aspect_ratio),
                        "has_light_background": bool(has_light_bg),
                        "mean_brightness": float(mean_brightness),
                    },
                )
            )

        return elements

    def _calculate_confidence(
        self,
        aspect_ratio: float,
        width: int,
        height: int,
        has_light_bg: bool,
        params: dict[str, Any],
    ) -> float:
        """Calculate confidence score based on multiple factors"""
        confidence = 0.5  # Base confidence

        # Aspect ratio in ideal range (4:1 to 8:1)
        if 4.0 <= aspect_ratio <= 8.0:
            confidence += 0.2
        elif 3.0 <= aspect_ratio < 4.0 or 8.0 < aspect_ratio <= 10.0:
            confidence += 0.1

        # Common input field size (200-400px wide, 30-40px tall)
        if 200 <= width <= 400 and 30 <= height <= 40:
            confidence += 0.2
        elif 150 <= width <= 500 and 25 <= height <= 50:
            confidence += 0.1

        # Light background (typical of input fields)
        if has_light_bg:
            confidence += 0.15

        return min(0.95, confidence)
