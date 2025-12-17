"""
Button Shadow Detector

Detects buttons by finding shadows and depth effects:
- Drop shadows using morphological operations
- Raised/embossed effects (brightness gradients)
- Border contrast changes
- Modern flat design with subtle shadows
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


class ButtonShadowDetector(BaseAnalyzer):
    """
    Detects buttons by identifying shadow and depth effects

    Algorithm:
    1. Detect drop shadows using morphological operations
    2. Find raised/embossed effects through brightness gradients
    3. Look for border contrast changes (3D effects)
    4. Detect modern flat design with subtle shadows
    5. Combine multiple shadow indicators for robust detection
    """

    @property
    def analysis_type(self) -> AnalysisType:
        return AnalysisType.CUSTOM

    @property
    def name(self) -> str:
        return "button_shadow_detector"

    @property
    def supports_multi_screenshot(self) -> bool:
        return True

    @property
    def required_screenshots(self) -> int:
        return 1

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            # Shadow detection parameters
            "shadow_blur_size": 5,
            "shadow_threshold": 20,  # Darkness threshold for shadows
            "min_shadow_area": 100,
            "max_shadow_offset": 10,  # Max shadow offset in pixels
            # Gradient detection for raised effects
            "gradient_threshold": 30,
            "detect_raised_buttons": True,
            # Border contrast detection
            "border_contrast_threshold": 40,
            "detect_border_effects": True,
            # Size constraints
            "min_button_width": 60,
            "max_button_width": 400,
            "min_button_height": 25,
            "max_button_height": 80,
            # Confidence thresholds
            "min_confidence": 0.5,
        }

    async def analyze(self, input_data: AnalysisInput) -> AnalysisResult:
        """Perform shadow-based button detection"""
        logger.info(f"Running button shadow detection on {len(input_data.screenshots)} screenshots")

        params = {**self.get_default_parameters(), **input_data.parameters}

        # Load images
        images_gray = self._load_images_grayscale(input_data.screenshot_data)
        images_color = self._load_images_color(input_data.screenshot_data)

        # Analyze each screenshot
        all_elements = []
        for screenshot_idx, (img_gray, img_color) in enumerate(
            zip(images_gray, images_color, strict=False)
        ):
            elements = await self._analyze_screenshot(img_gray, img_color, screenshot_idx, params)
            all_elements.extend(elements)

        logger.info(f"Detected {len(all_elements)} button candidates using shadow analysis")

        return AnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            elements=all_elements,
            confidence=0.74,  # Shadow detection can be sensitive to lighting
            metadata={
                "num_screenshots": len(images_gray),
                "method": "shadow_detection",
                "parameters": params,
                "detector_type": "button_shadow",
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
            images.append(cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        return images

    async def _analyze_screenshot(
        self,
        img_gray: np.ndarray,
        img_color: np.ndarray,
        screenshot_idx: int,
        params: dict[str, Any],
    ) -> list[DetectedElement]:
        """Analyze a single screenshot for shadow-based buttons"""
        elements = []

        # Strategy 1: Detect drop shadows
        drop_shadow_candidates = self._detect_drop_shadows(img_gray, params)

        # Strategy 2: Detect raised/embossed buttons (gradient analysis)
        if params["detect_raised_buttons"]:
            raised_candidates = self._detect_raised_buttons(img_gray, params)
        else:
            raised_candidates = []

        # Strategy 3: Detect border contrast effects
        if params["detect_border_effects"]:
            border_candidates = self._detect_border_effects(img_gray, params)
        else:
            border_candidates = []

        # Combine all candidates and remove duplicates
        all_candidates = drop_shadow_candidates + raised_candidates + border_candidates
        merged_candidates = self._merge_overlapping_candidates(all_candidates)

        # Create detected elements
        for bbox, shadow_info in merged_candidates:
            confidence = self._calculate_confidence(shadow_info, params)

            if confidence < params["min_confidence"]:
                continue

            elements.append(
                DetectedElement(
                    bounding_box=bbox,
                    confidence=confidence,
                    label="Button",
                    element_type="button",
                    screenshot_index=screenshot_idx,
                    metadata={
                        "method": "shadow_detection",
                        "shadow_info": shadow_info,
                    },
                )
            )

        return elements

    def _detect_drop_shadows(
        self, img_gray: np.ndarray, params: dict[str, Any]
    ) -> list[tuple[BoundingBox, dict[str, Any]]]:
        """
        Detect drop shadows beneath UI elements

        Drop shadows are darker regions slightly offset from the main element
        """
        candidates = []
        h, w = img_gray.shape

        # Apply Gaussian blur to smooth out noise
        blurred = cv2.GaussianBlur(
            img_gray, (params["shadow_blur_size"], params["shadow_blur_size"]), 0
        )

        # Find dark regions (potential shadows)
        # Shadows are darker than surrounding area
        mean_brightness = np.mean(blurred)
        shadow_mask = blurred < (mean_brightness - params["shadow_threshold"])
        shadow_mask = shadow_mask.astype(np.uint8) * 255

        # Morphological operations to clean up shadow mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in shadow mask
        contours, _ = cv2.findContours(shadow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Get shadow bounding box
            sx, sy, sw, sh = cv2.boundingRect(contour)
            shadow_area = cv2.contourArea(contour)

            # Skip very small shadows
            if shadow_area < params["min_shadow_area"]:
                continue

            # Look for brighter region above/left of shadow (the actual button)
            # Shadows are typically offset slightly down and/or right
            search_offsets = [
                (
                    -params["max_shadow_offset"],
                    -params["max_shadow_offset"],
                ),  # Top-left
                (0, -params["max_shadow_offset"]),  # Top
                (-params["max_shadow_offset"], 0),  # Left
            ]

            for offset_x, offset_y in search_offsets:
                bx = sx + offset_x
                by = sy + offset_y

                # Check bounds
                if bx < 0 or by < 0 or bx + sw > w or by + sh > h:
                    continue

                # Extract the potential button region
                button_region = img_gray[by : by + sh, bx : bx + sw]
                shadow_region = img_gray[sy : sy + sh, sx : sx + sw]

                # Button should be brighter than shadow
                button_brightness = np.mean(button_region)
                shadow_brightness = np.mean(shadow_region)

                if button_brightness > shadow_brightness + params["shadow_threshold"] * 0.5:
                    # Check size constraints
                    if not (params["min_button_width"] <= sw <= params["max_button_width"]):
                        continue
                    if not (params["min_button_height"] <= sh <= params["max_button_height"]):
                        continue

                    bbox = BoundingBox(x=int(bx), y=int(by), width=int(sw), height=int(sh))
                    shadow_info = {
                        "type": "drop_shadow",
                        "shadow_offset_x": offset_x,
                        "shadow_offset_y": offset_y,
                        "brightness_diff": float(button_brightness - shadow_brightness),
                    }
                    candidates.append((bbox, shadow_info))
                    break  # Found shadow for this contour

        return candidates

    def _detect_raised_buttons(
        self, img_gray: np.ndarray, params: dict[str, Any]
    ) -> list[tuple[BoundingBox, dict[str, Any]]]:
        """
        Detect raised/embossed buttons through gradient analysis

        Raised buttons have:
        - Lighter top edge
        - Darker bottom edge
        - Gradient from light to dark (top to bottom)
        """
        candidates = []

        # Calculate gradient in Y direction (vertical)
        gradient_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_y = np.abs(gradient_y).astype(np.uint8)

        # Threshold gradient to find strong vertical transitions
        _, gradient_mask = cv2.threshold(
            gradient_y, params["gradient_threshold"], 255, cv2.THRESH_BINARY
        )

        # Find contours in gradient mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        gradient_mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(gradient_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Check size constraints
            if not (params["min_button_width"] <= w <= params["max_button_width"]):
                continue
            if not (params["min_button_height"] <= h <= params["max_button_height"]):
                continue

            # Check if there's a gradient from top to bottom
            if h < 10:  # Too small to analyze gradient
                continue

            region = img_gray[y : y + h, x : x + w]

            # Analyze vertical gradient
            top_third = region[: h // 3, :]
            bottom_third = region[2 * h // 3 :, :]

            top_brightness = np.mean(top_third)
            bottom_brightness = np.mean(bottom_third)

            # Raised buttons are lighter at top, darker at bottom
            brightness_diff = top_brightness - bottom_brightness

            if brightness_diff > 10:  # Threshold for gradient
                bbox = BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h))
                shadow_info = {
                    "type": "raised_effect",
                    "gradient_strength": float(brightness_diff),
                    "top_brightness": float(top_brightness),
                    "bottom_brightness": float(bottom_brightness),
                }
                candidates.append((bbox, shadow_info))

        return candidates

    def _detect_border_effects(
        self, img_gray: np.ndarray, params: dict[str, Any]
    ) -> list[tuple[BoundingBox, dict[str, Any]]]:
        """
        Detect buttons with contrasting borders (3D border effects)

        3D borders have:
        - Light border on top/left
        - Dark border on bottom/right
        """
        candidates = []

        # Find edges
        edges = cv2.Canny(img_gray, 50, 150)

        # Dilate to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Check size constraints
            if not (params["min_button_width"] <= w <= params["max_button_width"]):
                continue
            if not (params["min_button_height"] <= h <= params["max_button_height"]):
                continue

            # Analyze border contrast
            border_info = self._analyze_border_contrast(img_gray, x, y, w, h, params)

            if border_info["has_3d_effect"]:
                bbox = BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h))
                shadow_info = {
                    "type": "border_effect",
                    **border_info,
                }
                candidates.append((bbox, shadow_info))

        return candidates

    def _analyze_border_contrast(
        self,
        img_gray: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Analyze border contrast to detect 3D effects

        Returns dict with:
        - has_3d_effect: bool
        - top_brightness: float
        - bottom_brightness: float
        - left_brightness: float
        - right_brightness: float
        """
        h_img, w_img = img_gray.shape
        border_width = 2

        # Extract border regions (safely handling boundaries)
        top_border = img_gray[
            max(0, y) : min(h_img, y + border_width), max(0, x) : min(w_img, x + w)
        ]
        bottom_border = img_gray[
            max(0, y + h - border_width) : min(h_img, y + h),
            max(0, x) : min(w_img, x + w),
        ]
        left_border = img_gray[
            max(0, y) : min(h_img, y + h), max(0, x) : min(w_img, x + border_width)
        ]
        right_border = img_gray[
            max(0, y) : min(h_img, y + h),
            max(0, x + w - border_width) : min(w_img, x + w),
        ]

        # Calculate average brightness
        top_brightness = np.mean(top_border) if top_border.size > 0 else 0
        bottom_brightness = np.mean(bottom_border) if bottom_border.size > 0 else 0
        left_brightness = np.mean(left_border) if left_border.size > 0 else 0
        right_brightness = np.mean(right_border) if right_border.size > 0 else 0

        # 3D effect: top/left lighter than bottom/right
        top_bottom_diff = top_brightness - bottom_brightness
        left_right_diff = left_brightness - right_brightness

        has_3d_effect = (
            top_bottom_diff > params["border_contrast_threshold"] * 0.3
            and left_right_diff > params["border_contrast_threshold"] * 0.3
        )

        return {
            "has_3d_effect": has_3d_effect,
            "top_brightness": float(top_brightness),
            "bottom_brightness": float(bottom_brightness),
            "left_brightness": float(left_brightness),
            "right_brightness": float(right_brightness),
            "vertical_contrast": float(top_bottom_diff),
            "horizontal_contrast": float(left_right_diff),
        }

    def _merge_overlapping_candidates(
        self, candidates: list[tuple[BoundingBox, dict[str, Any]]]
    ) -> list[tuple[BoundingBox, dict[str, Any]]]:
        """
        Merge overlapping candidate detections

        If multiple shadow detection methods find the same button,
        combine them and use the best bounding box
        """
        if not candidates:
            return []

        # Sort by confidence/quality
        merged: list[tuple[BoundingBox, dict[str, Any]]] = []

        for bbox, info in candidates:
            # Check if this overlaps with any existing merged candidate
            overlapped = False
            for i, (existing_bbox, existing_info) in enumerate(merged):
                if bbox.iou(existing_bbox) > 0.5:
                    # Merge - keep the one with more evidence
                    existing_types = existing_info.get("detection_types", [existing_info["type"]])
                    new_types = existing_types + [info["type"]]

                    # Use average bounding box
                    avg_bbox = BoundingBox(
                        x=(bbox.x + existing_bbox.x) // 2,
                        y=(bbox.y + existing_bbox.y) // 2,
                        width=(bbox.width + existing_bbox.width) // 2,
                        height=(bbox.height + existing_bbox.height) // 2,
                    )

                    # Merge info
                    merged_info = {
                        **existing_info,
                        "detection_types": new_types,
                        "num_detections": len(set(new_types)),
                    }

                    merged[i] = (avg_bbox, merged_info)
                    overlapped = True
                    break

            if not overlapped:
                merged.append(
                    (
                        bbox,
                        {
                            **info,
                            "detection_types": [info["type"]],
                            "num_detections": 1,
                        },
                    )
                )

        return merged

    def _calculate_confidence(self, shadow_info: dict[str, Any], params: dict[str, Any]) -> float:
        """
        Calculate confidence score for shadow-based button detection

        Factors:
        - Number of different shadow detection methods that found this button
        - Strength of shadow/gradient effects
        - Quality of 3D effects
        """
        confidence = 0.4  # Base confidence

        # Bonus for multiple detection methods agreeing
        num_detections = shadow_info.get("num_detections", 1)
        confidence += min(0.3, num_detections * 0.15)

        # Type-specific bonuses
        shadow_type = shadow_info.get("type", "unknown")

        if shadow_type == "drop_shadow":
            brightness_diff = shadow_info.get("brightness_diff", 0)
            confidence += min(0.2, brightness_diff / 100)

        elif shadow_type == "raised_effect":
            gradient_strength = shadow_info.get("gradient_strength", 0)
            confidence += min(0.2, gradient_strength / 50)

        elif shadow_type == "border_effect":
            if shadow_info.get("has_3d_effect", False):
                confidence += 0.2

        return min(1.0, max(0.0, confidence))  # type: ignore[no-any-return]
