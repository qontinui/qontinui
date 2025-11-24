"""
Button Ensemble Detector - Ensemble of Specialist Detectors

Combines multiple specialized detectors, each optimized for different button styles:
- Detector A: Flat design buttons (Material Design, Bootstrap)
- Detector B: 3D/Skeuomorphic buttons (raised, shadowed)
- Detector C: Icon-only buttons
- Detector D: Text-link buttons (minimal styling)

Aggregates results with confidence weighting based on detection strength.
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


class ButtonEnsembleDetector(BaseAnalyzer):
    """
    Ensemble of specialized button detectors

    Each detector specializes in a different button style:
    1. Flat Design Detector: Modern UI frameworks (Material, Bootstrap)
    2. 3D/Skeuomorphic Detector: Traditional raised buttons with shadows
    3. Icon Button Detector: Icon-only or icon+text buttons
    4. Text Link Detector: Minimal text-based buttons

    Uses confidence-weighted aggregation with non-maximum suppression.
    """

    @property
    def analysis_type(self) -> AnalysisType:
        return AnalysisType.SINGLE_SHOT

    @property
    def name(self) -> str:
        return "button_ensemble"

    @property
    def supports_multi_screenshot(self) -> bool:
        return True

    @property
    def required_screenshots(self) -> int:
        return 1

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Detector enable flags
            "enable_flat_detector": True,
            "enable_3d_detector": True,
            "enable_icon_detector": True,
            "enable_text_link_detector": True,
            # NMS parameters
            "nms_iou_threshold": 0.4,  # IOU threshold for non-maximum suppression
            # Flat design params
            "flat_min_saturation": 40,
            "flat_border_thickness": 1,
            # 3D design params
            "shadow_detection_threshold": 20,
            "gradient_detection_threshold": 30,
            # Icon params
            "icon_min_circularity": 0.6,
            "icon_size_range": (20, 80),
            # Text link params
            "text_aspect_ratio_range": (2.0, 8.0),
            "text_height_range": (12, 30),
        }

    async def analyze(self, input_data: AnalysisInput) -> AnalysisResult:
        """Perform ensemble detection"""
        logger.info(
            f"Running ensemble detection on {len(input_data.screenshots)} screenshots"
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
            f"Found {len(all_elements)} buttons with avg confidence {avg_confidence:.2f}"
        )

        return AnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            elements=all_elements,
            confidence=float(avg_confidence),
            metadata={
                "num_screenshots": len(images),
                "method": "ensemble",
                "detectors_enabled": {
                    "flat": params["enable_flat_detector"],
                    "3d": params["enable_3d_detector"],
                    "icon": params["enable_icon_detector"],
                    "text_link": params["enable_text_link_detector"],
                },
                "parameters": params,
            },
        )

    def _load_images(self, screenshot_data: List[bytes]) -> List[np.ndarray]:
        """Load screenshots as numpy arrays"""
        images = []
        for data in screenshot_data:
            img = Image.open(BytesIO(data)).convert("RGB")
            images.append(np.array(img))
        return images

    async def _analyze_screenshot(
        self, img: np.ndarray, screenshot_idx: int, params: Dict[str, Any]
    ) -> List[DetectedElement]:
        """Analyze single screenshot using ensemble approach"""

        all_detections = []

        # Run each specialist detector
        if params["enable_flat_detector"]:
            flat_detections = self._detect_flat_buttons(img, params)
            all_detections.extend(
                [(bbox, conf, "flat") for bbox, conf in flat_detections]
            )

        if params["enable_3d_detector"]:
            d3_detections = self._detect_3d_buttons(img, params)
            all_detections.extend([(bbox, conf, "3d") for bbox, conf in d3_detections])

        if params["enable_icon_detector"]:
            icon_detections = self._detect_icon_buttons(img, params)
            all_detections.extend(
                [(bbox, conf, "icon") for bbox, conf in icon_detections]
            )

        if params["enable_text_link_detector"]:
            text_detections = self._detect_text_link_buttons(img, params)
            all_detections.extend(
                [(bbox, conf, "text_link") for bbox, conf in text_detections]
            )

        logger.info(
            f"Screenshot {screenshot_idx}: {len(all_detections)} raw detections"
        )

        # Apply non-maximum suppression
        final_detections = self._apply_nms(all_detections, params)

        # Convert to DetectedElement
        elements = []
        for bbox, conf, detector_type in final_detections:
            elements.append(
                DetectedElement(
                    bounding_box=bbox,
                    confidence=conf,
                    label=f"Button ({detector_type})",
                    element_type="button",
                    screenshot_index=screenshot_idx,
                    metadata={
                        "method": "ensemble",
                        "detector": detector_type,
                    },
                )
            )

        return elements

    def _detect_flat_buttons(
        self, img: np.ndarray, params: Dict[str, Any]
    ) -> List[Tuple[BoundingBox, float]]:
        """
        Detect flat design buttons (Material Design, Bootstrap style)

        Characteristics:
        - Uniform solid color background
        - Clean edges (minimal or no border)
        - Rectangular shape with possible rounded corners
        - High saturation colors common
        """
        detections = []

        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find regions with uniform color
        saturation = hsv[:, :, 1]
        mask = saturation > params["flat_min_saturation"]

        # Clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Size filter
            area = w * h
            if not (500 <= area <= 40000):
                continue

            # Aspect ratio filter (flat buttons are wider than tall)
            aspect_ratio = w / h if h > 0 else 0
            if not (1.5 <= aspect_ratio <= 8.0):
                continue

            # Check for uniform color
            region_hsv = hsv[y : y + h, x : x + w]
            hue_std = np.std(region_hsv[:, :, 0])
            uniformity = max(0, 1 - (hue_std / 180))

            if uniformity < 0.7:
                continue

            # Flat buttons have clean edges
            edge_region = gray[max(0, y - 2) : y + h + 2, max(0, x - 2) : x + w + 2]
            edges = cv2.Canny(edge_region, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            # Moderate edge density (has border but not too complex)
            if edge_density > 0.3:
                continue

            # Confidence based on uniformity and typical characteristics
            confidence = min(0.9, 0.5 + uniformity * 0.3 + (1 - edge_density) * 0.1)

            detections.append((BoundingBox(x=x, y=y, width=w, height=h), confidence))

        return detections

    def _detect_3d_buttons(
        self, img: np.ndarray, params: Dict[str, Any]
    ) -> List[Tuple[BoundingBox, float]]:
        """
        Detect 3D/skeuomorphic buttons with shadows and gradients

        Characteristics:
        - Gradients (light to dark or vice versa)
        - Shadows below button
        - Often have borders
        - Raised appearance
        """
        detections = []

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Detect edges strongly (3D buttons have defined borders)
        edges = cv2.Canny(gray, 100, 200)

        # Dilate to connect edge fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            area = w * h
            if not (500 <= area <= 40000):
                continue

            aspect_ratio = w / h if h > 0 else 0
            if not (1.5 <= aspect_ratio <= 8.0):
                continue

            # Check for gradient (vertical gradient is common in 3D buttons)
            region = gray[y : y + h, x : x + w]
            if region.size == 0:
                continue

            # Calculate vertical gradient strength
            top_brightness = np.mean(region[: h // 3, :])
            bottom_brightness = np.mean(region[2 * h // 3 :, :])
            gradient_strength = abs(top_brightness - bottom_brightness)

            if gradient_strength < params["gradient_detection_threshold"]:
                continue

            # Check for shadow below
            shadow_region_y = min(y + h, img.shape[0] - 3)
            shadow_region = gray[
                shadow_region_y : min(shadow_region_y + 3, img.shape[0]), x : x + w
            ]

            if shadow_region.size > 0:
                shadow_darkness = (
                    np.mean(shadow_region)
                    < np.mean(region) - params["shadow_detection_threshold"]
                )
            else:
                shadow_darkness = False

            # Confidence based on gradient and shadow
            confidence = 0.6
            if gradient_strength > 50:
                confidence += 0.2
            if shadow_darkness:
                confidence += 0.15

            confidence = min(0.9, confidence)

            detections.append((BoundingBox(x=x, y=y, width=w, height=h), confidence))

        return detections

    def _detect_icon_buttons(
        self, img: np.ndarray, params: Dict[str, Any]
    ) -> List[Tuple[BoundingBox, float]]:
        """
        Detect icon-only or icon+text buttons

        Characteristics:
        - Often circular or square
        - Contains icon/symbol
        - Typically smaller than text buttons
        - May have background or be transparent
        """
        detections = []

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Detect blobs that might be icons
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        min_size, max_size = params["icon_size_range"]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Icon buttons are typically small and squarish
            if not (min_size <= w <= max_size and min_size <= h <= max_size):
                continue

            aspect_ratio = w / h if h > 0 else 0
            if not (0.7 <= aspect_ratio <= 1.5):  # Nearly square
                continue

            # Calculate circularity
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter**2)

            if circularity < params["icon_min_circularity"]:
                continue

            # Check for icon-like content (moderate complexity)
            region = edges[y : y + h, x : x + w]
            edge_density = np.sum(region > 0) / region.size

            if not (0.1 <= edge_density <= 0.5):
                continue

            confidence = min(0.85, 0.4 + circularity * 0.3 + edge_density * 0.2)

            # Add padding for button area
            padding = 4
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(img.shape[1] - x_pad, w + 2 * padding)
            h_pad = min(img.shape[0] - y_pad, h + 2 * padding)

            detections.append(
                (BoundingBox(x=x_pad, y=y_pad, width=w_pad, height=h_pad), confidence)
            )

        return detections

    def _detect_text_link_buttons(
        self, img: np.ndarray, params: Dict[str, Any]
    ) -> List[Tuple[BoundingBox, float]]:
        """
        Detect minimal text-link style buttons

        Characteristics:
        - Minimal styling (just text, maybe underline)
        - Horizontal text layout
        - Small height, variable width
        - Often no background color
        """
        detections = []

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Use MSER for text detection
        mser = cv2.MSER_create(
            _min_area=100,
            _max_area=5000,
        )

        regions, _ = mser.detectRegions(gray)

        min_h, max_h = params["text_height_range"]
        min_ar, max_ar = params["text_aspect_ratio_range"]

        for region in regions:
            if len(region) < 10:
                continue

            x, y, w, h = cv2.boundingRect(region)

            # Text link buttons are short and wide
            if not (min_h <= h <= max_h):
                continue

            aspect_ratio = w / h if h > 0 else 0
            if not (min_ar <= aspect_ratio <= max_ar):
                continue

            # Check for text-like patterns (horizontal strokes)
            region_img = gray[y : y + h, x : x + w]
            horizontal_profile = np.mean(region_img, axis=1)
            profile_variance = np.var(horizontal_profile)

            # Text has varying horizontal profile
            if profile_variance < 10:
                continue

            confidence = min(0.75, 0.4 + (profile_variance / 500) * 0.35)

            # Add padding
            padding = 6
            x_pad = max(0, x - padding)
            y_pad = max(0, y - padding)
            w_pad = min(img.shape[1] - x_pad, w + 2 * padding)
            h_pad = min(img.shape[0] - y_pad, h + 2 * padding)

            detections.append(
                (BoundingBox(x=x_pad, y=y_pad, width=w_pad, height=h_pad), confidence)
            )

        return detections

    def _apply_nms(
        self, detections: List[Tuple[BoundingBox, float, str]], params: Dict[str, Any]
    ) -> List[Tuple[BoundingBox, float, str]]:
        """
        Apply non-maximum suppression to remove overlapping detections
        Keep detection with highest confidence
        """
        if not detections:
            return []

        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: x[1], reverse=True)

        keep = []
        while detections:
            # Take highest confidence detection
            best = detections.pop(0)
            keep.append(best)

            # Remove overlapping detections
            detections = [
                det
                for det in detections
                if det[0].iou(best[0]) < params["nms_iou_threshold"]
            ]

        return keep
