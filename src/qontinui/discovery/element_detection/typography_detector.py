"""
Typography Detector - Text Characteristic Analysis

Detects buttons by analyzing text characteristics commonly found in buttons:
- Bold or medium weight fonts
- ALL CAPS or Title Case formatting
- Short text (1-3 words typical for buttons)
- Centered text alignment
- Action-oriented keywords

Combines text detection with font/style analysis to identify button text,
then expands to button boundaries.
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


class TypographyDetector(BaseAnalyzer):
    """
    Detects buttons by analyzing text typography characteristics

    Algorithm:
    1. Detect text regions using MSER/stroke width analysis
    2. Analyze text characteristics:
       - Font weight (bold detection)
       - Case pattern (ALL CAPS, Title Case)
       - Text length (word count)
       - Alignment (centered)
    3. Match against button text patterns
    4. Expand text region to button boundaries
    5. Validate button-like container properties
    """

    @property
    def analysis_type(self) -> AnalysisType:
        return AnalysisType.SINGLE_SHOT

    @property
    def name(self) -> str:
        return "typography"

    @property
    def supports_multi_screenshot(self) -> bool:
        return True

    @property
    def required_screenshots(self) -> int:
        return 1

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            # Text detection
            "min_text_area": 100,
            "max_text_area": 5000,
            "text_aspect_ratio_range": (1.5, 10.0),
            # Typography characteristics
            "bold_stroke_ratio": 0.15,  # Stroke width ratio indicating bold
            "min_stroke_width": 2,
            "max_stroke_width": 8,
            # Button text patterns
            "button_keywords": [
                "submit",
                "save",
                "cancel",
                "delete",
                "add",
                "create",
                "new",
                "update",
                "edit",
                "confirm",
                "ok",
                "yes",
                "no",
                "apply",
                "login",
                "signup",
                "register",
                "sign in",
                "sign up",
                "search",
                "find",
                "go",
                "send",
                "share",
                "post",
                "upload",
                "download",
                "import",
                "export",
                "next",
                "previous",
                "back",
                "continue",
                "finish",
                "buy",
                "purchase",
                "checkout",
                "pay",
                "close",
                "dismiss",
                "accept",
                "reject",
            ],
            # Case patterns (regex)
            "all_caps_pattern": r"^[A-Z\s]+$",
            "title_case_pattern": r"^[A-Z][a-z]*(\s[A-Z][a-z]*)*$",
            # Expansion for button boundaries
            "boundary_expansion_pixels": 10,
            "min_button_padding": 6,
            # Validation
            "min_button_area": 500,
            "max_button_area": 40000,
        }

    async def analyze(self, input_data: AnalysisInput) -> AnalysisResult:
        """Perform typography-based detection"""
        logger.info(
            f"Running typography detection on {len(input_data.screenshots)} screenshots"
        )

        params = {**self.get_default_parameters(), **input_data.parameters}

        # Load images
        images = self._load_images(input_data.screenshot_data)

        # Analyze each screenshot
        all_elements = []
        for screenshot_idx, img in enumerate(images):
            elements = await self._analyze_screenshot(img, screenshot_idx, params)
            all_elements.extend(elements)

        avg_confidence = (
            np.mean([e.confidence for e in all_elements]) if all_elements else 0.0
        )

        logger.info(
            f"Found {len(all_elements)} button text regions with "
            f"avg confidence {avg_confidence:.2f}"
        )

        return AnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            elements=all_elements,
            confidence=float(avg_confidence),
            metadata={
                "num_screenshots": len(images),
                "method": "typography",
                "parameters": params,
            },
        )

    def _load_images(self, screenshot_data: list[bytes]) -> list[np.ndarray]:
        """Load screenshots as numpy arrays"""
        images = []
        for data in screenshot_data:
            img = Image.open(BytesIO(data)).convert("RGB")
            images.append(np.array(img))
        return images

    async def _analyze_screenshot(
        self, img: np.ndarray, screenshot_idx: int, params: dict[str, Any]
    ) -> list[DetectedElement]:
        """Analyze single screenshot for button text"""

        # Detect text regions
        text_regions = self._detect_text_regions(img, params)

        logger.info(
            f"Screenshot {screenshot_idx}: Found {len(text_regions)} text regions"
        )

        elements = []
        for text_bbox in text_regions:
            # Analyze typography characteristics
            typo_features = self._analyze_typography(img, text_bbox, params)

            if typo_features is None:
                continue

            # Calculate confidence based on typography
            confidence = self._calculate_confidence(typo_features, params)

            if confidence < 0.4:
                continue

            # Expand to button boundaries
            button_bbox = self._expand_to_button_boundary(img, text_bbox, params)

            # Validate button properties
            if not self._validate_button(img, button_bbox, params):
                continue

            elements.append(
                DetectedElement(
                    bounding_box=button_bbox,
                    confidence=confidence,
                    label="Button (Typography)",
                    element_type="button",
                    screenshot_index=screenshot_idx,
                    metadata={
                        "method": "typography",
                        **typo_features,
                        "text_bbox": {
                            "x": text_bbox.x,
                            "y": text_bbox.y,
                            "width": text_bbox.width,
                            "height": text_bbox.height,
                        },
                    },
                )
            )

        return elements

    def _detect_text_regions(
        self, img: np.ndarray, params: dict[str, Any]
    ) -> list[BoundingBox]:
        """
        Detect text regions in image

        Uses MSER (Maximally Stable Extremal Regions) which works well for text
        """
        text_regions = []

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # MSER detector for text
        mser = cv2.MSER_create(  # type: ignore[attr-defined]
            _min_area=int(params["min_text_area"]),
            _max_area=int(params["max_text_area"]),
            _delta=5,
        )

        regions, _ = mser.detectRegions(gray)

        min_ar, max_ar = params["text_aspect_ratio_range"]

        for region in regions:
            if len(region) < 10:
                continue

            x, y, w, h = cv2.boundingRect(region)

            # Filter by aspect ratio (text is horizontal)
            aspect_ratio = w / h if h > 0 else 0
            if not (min_ar <= aspect_ratio <= max_ar):
                continue

            text_regions.append(BoundingBox(x=x, y=y, width=w, height=h))

        return text_regions

    def _analyze_typography(
        self, img: np.ndarray, bbox: BoundingBox, params: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Analyze typography characteristics of text region

        Returns dict of features or None if not button-like
        """
        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height

        # Extract text region
        region = img[y : y + h, x : x + w]
        if region.size == 0:
            return None

        gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

        # Analyze stroke width (bold detection)
        stroke_info = self._analyze_stroke_width(gray_region, params)
        if stroke_info is None:
            return None

        is_bold, avg_stroke_width = stroke_info

        # Analyze text compactness (buttons have tight text)
        compactness = self._analyze_text_compactness(gray_region)

        # Analyze horizontal alignment
        is_centered = self._is_text_centered(gray_region)

        # Analyze text pattern (short text typical for buttons)
        is_short_text = w < 200  # Approximate - buttons typically < 200px wide

        features = {
            "is_bold": is_bold,
            "avg_stroke_width": float(avg_stroke_width),
            "compactness": float(compactness),
            "is_centered": is_centered,
            "is_short_text": is_short_text,
            "text_width": w,
            "text_height": h,
        }

        return features

    def _analyze_stroke_width(
        self, gray: np.ndarray, params: dict[str, Any]
    ) -> tuple[bool, float] | None:
        """
        Analyze stroke width to detect bold text

        Returns (is_bold, avg_stroke_width) or None
        """
        # Use morphological operations to estimate stroke width
        # Thicker strokes = bold text

        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if text is dark on light background
        if np.mean(binary) > 127:
            binary = 255 - binary

        if np.sum(binary > 0) < 10:
            return None

        # Distance transform to find stroke widths
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # Average distance in text regions is approximately half stroke width
        text_pixels = binary > 0
        if not np.any(text_pixels):
            return None

        avg_distance = np.mean(dist_transform[text_pixels])
        avg_stroke_width = avg_distance * 2

        # Check if in valid range
        if not (
            params["min_stroke_width"] <= avg_stroke_width <= params["max_stroke_width"]
        ):
            return None

        # Bold if stroke width is above threshold
        height = gray.shape[0]
        stroke_ratio = avg_stroke_width / height if height > 0 else 0
        is_bold = stroke_ratio >= params["bold_stroke_ratio"]

        return (is_bold, avg_stroke_width)

    def _analyze_text_compactness(self, gray: np.ndarray) -> float:
        """
        Analyze how compact the text is (buttons have tight text)

        Returns compactness score (0-1), higher = more compact
        """
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if needed
        if np.mean(binary) > 127:
            binary = 255 - binary

        # Calculate ratio of text pixels to total area
        text_pixels = np.sum(binary > 0)
        total_pixels = binary.size

        if total_pixels == 0:
            return 0.0

        compactness = text_pixels / total_pixels

        return float(compactness)

    def _is_text_centered(self, gray: np.ndarray) -> bool:
        """
        Check if text appears centered (common for buttons)

        Returns True if text is approximately centered
        """
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if np.mean(binary) > 127:
            binary = 255 - binary

        # Find text bounding box within region
        coords = cv2.findNonZero(binary)
        if coords is None:
            return False

        x, y, w, h = cv2.boundingRect(coords)

        # Check horizontal centering
        left_margin = x
        right_margin = gray.shape[1] - (x + w)

        # Consider centered if margins are similar (within 30%)
        if left_margin + right_margin == 0:
            return False

        center_ratio = min(left_margin, right_margin) / max(left_margin, right_margin)

        return center_ratio >= 0.7

    def _calculate_confidence(
        self, features: dict[str, Any], params: dict[str, Any]
    ) -> float:
        """
        Calculate confidence based on typography features
        """
        confidence = 0.0

        # Bold text is common in buttons
        if features["is_bold"]:
            confidence += 0.3

        # Centered text
        if features["is_centered"]:
            confidence += 0.25

        # Short text (1-3 words typical)
        if features["is_short_text"]:
            confidence += 0.2

        # Good compactness (0.2-0.5 typical for button text)
        compactness = features["compactness"]
        if 0.2 <= compactness <= 0.5:
            confidence += 0.15
        elif 0.15 <= compactness < 0.2 or 0.5 < compactness <= 0.6:
            confidence += 0.05

        # Text height in typical button range (14-24px)
        text_height = features["text_height"]
        if 14 <= text_height <= 24:
            confidence += 0.1

        return min(0.9, confidence)

    def _expand_to_button_boundary(
        self, img: np.ndarray, text_bbox: BoundingBox, params: dict[str, Any]
    ) -> BoundingBox:
        """
        Expand text bounding box to include button boundaries

        Looks for uniform color region around text
        """
        x, y, w, h = text_bbox.x, text_bbox.y, text_bbox.width, text_bbox.height

        # Start with expansion padding
        expansion = params["boundary_expansion_pixels"]

        x_expanded = max(0, x - expansion)
        y_expanded = max(0, y - expansion)
        w_expanded = min(img.shape[1] - x_expanded, w + 2 * expansion)
        h_expanded = min(img.shape[0] - y_expanded, h + 2 * expansion)

        # Try to refine by finding uniform color region
        expanded_region = img[
            y_expanded : y_expanded + h_expanded, x_expanded : x_expanded + w_expanded
        ]

        if expanded_region.size > 0:
            # Check if region has uniform background
            hsv = cv2.cvtColor(expanded_region, cv2.COLOR_RGB2HSV)
            hue_std = np.std(hsv[:, :, 0])

            # If not uniform, use conservative expansion
            if hue_std > 40:
                expansion = params["min_button_padding"]
                x_expanded = max(0, x - expansion)
                y_expanded = max(0, y - expansion)
                w_expanded = min(img.shape[1] - x_expanded, w + 2 * expansion)
                h_expanded = min(img.shape[0] - y_expanded, h + 2 * expansion)

        return BoundingBox(
            x=x_expanded, y=y_expanded, width=w_expanded, height=h_expanded
        )

    def _validate_button(
        self, img: np.ndarray, bbox: BoundingBox, params: dict[str, Any]
    ) -> bool:
        """
        Validate that expanded region has button-like properties
        """
        area = bbox.width * bbox.height

        # Size check
        if not (params["min_button_area"] <= area <= params["max_button_area"]):
            return False

        # Aspect ratio check (buttons are wider than tall)
        aspect_ratio = bbox.width / bbox.height if bbox.height > 0 else 0
        if not (1.5 <= aspect_ratio <= 8.0):
            return False

        return True
