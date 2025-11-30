"""
Button Color Detector

Detects buttons by analyzing color consistency:
- Segments image by color using k-means clustering
- Finds uniform color regions with borders
- Looks for common button colors (blue, green, red, etc.)
- Detects contrast between button and background
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


class ButtonColorDetector(BaseAnalyzer):
    """
    Detects buttons by analyzing color patterns and consistency

    Algorithm:
    1. Segment image by color using k-means clustering
    2. Find uniform color regions that stand out
    3. Look for common button colors (blue, green, red)
    4. Detect high contrast between button and background
    5. Filter by size and shape constraints
    """

    # Common button colors in HSV space
    # Format: (name, H_min, H_max, S_min, V_min)
    COMMON_BUTTON_COLORS = [
        ("blue", 100, 130, 50, 50),  # Blue buttons (primary actions)
        ("green", 40, 80, 50, 50),  # Green buttons (success, confirm)
        ("red", 0, 10, 50, 50),  # Red buttons (danger, delete)
        ("red2", 170, 180, 50, 50),  # Red (wraps around hue)
        ("orange", 10, 25, 50, 50),  # Orange buttons (warnings)
        ("purple", 130, 160, 50, 50),  # Purple buttons
    ]

    @property
    def analysis_type(self) -> AnalysisType:
        return AnalysisType.CUSTOM

    @property
    def name(self) -> str:
        return "button_color_detector"

    @property
    def supports_multi_screenshot(self) -> bool:
        return True

    @property
    def required_screenshots(self) -> int:
        return 1

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            # K-means clustering parameters
            "num_clusters": 8,  # Number of color clusters
            "kmeans_attempts": 3,
            # Color uniformity parameters
            "min_color_uniformity": 0.7,  # How uniform the color should be
            "check_common_colors": True,
            # Contrast parameters
            "min_contrast": 30,  # Minimum contrast with background
            # Size constraints
            "min_width": 60,
            "max_width": 400,
            "min_height": 25,
            "max_height": 80,
            "min_area": 1500,  # Minimum button area
            # Shape constraints
            "min_aspect_ratio": 1.5,  # Buttons are typically wider than tall
            "max_aspect_ratio": 6.0,
            # Confidence thresholds
            "min_confidence": 0.5,
        }

    async def analyze(self, input_data: AnalysisInput) -> AnalysisResult:
        """Perform color-based button detection"""
        logger.info(f"Running button color detection on {len(input_data.screenshots)} screenshots")

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

        logger.info(f"Detected {len(all_elements)} button candidates using color analysis")

        return AnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            elements=all_elements,
            confidence=0.76,  # Color-based detection is fairly reliable
            metadata={
                "num_screenshots": len(images_gray),
                "method": "color_detection",
                "parameters": params,
                "detector_type": "button_color",
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
        """Analyze a single screenshot for color-based buttons"""
        elements = []

        h, w = img_color.shape[:2]

        # Step 1: Perform k-means color clustering
        clustered_img, labels = self._kmeans_segmentation(img_color, params)

        # Step 2: For each cluster, find uniform regions
        for cluster_id in range(params["num_clusters"]):
            # Create mask for this cluster
            cluster_mask = (labels == cluster_id).astype(np.uint8) * 255

            # Find contours in this cluster
            contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Get bounding rectangle
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                area = w_rect * h_rect

                # Step 3: Filter by size
                if not (params["min_width"] <= w_rect <= params["max_width"]):
                    continue
                if not (params["min_height"] <= h_rect <= params["max_height"]):
                    continue
                if area < params["min_area"]:
                    continue

                # Step 4: Filter by aspect ratio
                aspect_ratio = w_rect / h_rect if h_rect > 0 else 0
                if not (params["min_aspect_ratio"] <= aspect_ratio <= params["max_aspect_ratio"]):
                    continue

                # Step 5: Analyze color uniformity within this region
                region = img_color[y : y + h_rect, x : x + w_rect]
                color_info = self._analyze_region_color(region, params)

                if color_info["uniformity"] < params["min_color_uniformity"]:
                    continue

                # Step 6: Check contrast with background
                contrast = self._calculate_contrast_with_background(
                    img_color, x, y, w_rect, h_rect, region
                )

                if contrast < params["min_contrast"]:
                    continue

                # Step 7: Calculate confidence
                confidence = self._calculate_confidence(
                    w_rect, h_rect, aspect_ratio, color_info, contrast, params
                )

                if confidence < params["min_confidence"]:
                    continue

                # Create detected element
                elements.append(
                    DetectedElement(
                        bounding_box=BoundingBox(
                            x=int(x), y=int(y), width=int(w_rect), height=int(h_rect)
                        ),
                        confidence=confidence,
                        label="Button",
                        element_type="button",
                        screenshot_index=screenshot_idx,
                        metadata={
                            "method": "color_detection",
                            "color_info": color_info,
                            "contrast": float(contrast),
                            "cluster_id": int(cluster_id),
                        },
                    )
                )

        return elements

    def _kmeans_segmentation(
        self, img_color: np.ndarray, params: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Segment image using k-means color clustering

        Returns:
            (clustered_image, labels)
        """
        # Reshape image to list of pixels
        h, w = img_color.shape[:2]
        pixels = img_color.reshape((-1, 3)).astype(np.float32)

        # Define k-means criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        # Perform k-means clustering
        _, labels, centers = cv2.kmeans(
            pixels,
            params["num_clusters"],
            None,
            criteria,
            params["kmeans_attempts"],
            cv2.KMEANS_PP_CENTERS,
        )  # type: ignore[call-overload]

        # Convert back to image
        centers = np.uint8(centers)
        clustered = centers[labels.flatten()]  # type: ignore[index]
        clustered_img = clustered.reshape((h, w, 3))

        # Reshape labels
        labels = labels.reshape((h, w))

        return clustered_img, labels

    def _analyze_region_color(self, region: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze color properties of a region

        Returns:
            Dict with:
            - uniformity: how uniform the color is (0-1)
            - dominant_color: (B, G, R) tuple
            - is_common_button_color: bool
            - color_name: str or None
        """
        if region.size == 0:
            return {
                "uniformity": 0.0,
                "dominant_color": (0, 0, 0),
                "is_common_button_color": False,
                "color_name": None,
            }

        # Calculate color uniformity using standard deviation
        std_dev = np.std(region, axis=(0, 1))
        avg_std = np.mean(std_dev)
        # Uniformity: lower std = higher uniformity
        uniformity = max(0, 1.0 - (avg_std / 50))  # 50 is empirical threshold

        # Get dominant color (mean)
        dominant_color = tuple(np.mean(region, axis=(0, 1)).astype(int))

        # Check if this matches common button colors
        is_common, color_name = self._check_common_button_color(region, params)

        return {
            "uniformity": float(uniformity),
            "dominant_color": dominant_color,
            "is_common_button_color": is_common,
            "color_name": color_name,
            "std_dev": float(avg_std),
        }

    def _check_common_button_color(
        self, region: np.ndarray, params: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """
        Check if region matches common button colors

        Returns:
            (is_common, color_name)
        """
        if not params["check_common_colors"]:
            return False, None

        # Convert to HSV for color analysis
        region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        avg_hsv = np.mean(region_hsv, axis=(0, 1))
        h, s, v = avg_hsv

        # Check against common button colors
        for color_name, h_min, h_max, s_min, v_min in self.COMMON_BUTTON_COLORS:
            if h_min <= h <= h_max and s >= s_min and v >= v_min:
                return True, color_name

        return False, None

    def _calculate_contrast_with_background(
        self,
        img_color: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        region: np.ndarray,
    ) -> float:
        """
        Calculate contrast between button region and surrounding background

        High contrast indicates the button stands out visually
        """
        img_h, img_w = img_color.shape[:2]

        # Define background region (border around button)
        border_width = 10
        x1 = max(0, x - border_width)
        y1 = max(0, y - border_width)
        x2 = min(img_w, x + w + border_width)
        y2 = min(img_h, y + h + border_width)

        # Extract background region (excluding button itself)
        background = img_color[y1:y2, x1:x2].copy()

        # Mask out the button region from background
        button_x_in_bg = x - x1
        button_y_in_bg = y - y1
        if (
            button_y_in_bg >= 0
            and button_x_in_bg >= 0
            and button_y_in_bg + h <= background.shape[0]
            and button_x_in_bg + w <= background.shape[1]
        ):
            # Set button region to zero (will be excluded from mean calculation)
            mask = np.ones(background.shape[:2], dtype=bool)
            mask[button_y_in_bg : button_y_in_bg + h, button_x_in_bg : button_x_in_bg + w] = False
            background_pixels = background[mask]
        else:
            background_pixels = background.reshape(-1, 3)

        if len(background_pixels) == 0:
            return 0.0

        # Calculate mean colors
        button_mean = np.mean(region, axis=(0, 1))
        background_mean = np.mean(background_pixels, axis=0)

        # Calculate Euclidean distance in color space
        contrast = np.linalg.norm(button_mean - background_mean)

        return float(contrast)

    def _calculate_confidence(
        self,
        width: int,
        height: int,
        aspect_ratio: float,
        color_info: dict[str, Any],
        contrast: float,
        params: dict[str, Any],
    ) -> float:
        """
        Calculate confidence score for color-based button detection

        Factors:
        - Color uniformity: buttons have consistent color
        - Common button color: bonus for typical button colors
        - Contrast: buttons stand out from background
        - Size and aspect ratio: typical button proportions
        """
        confidence = 0.0

        # Color uniformity (0.0 to 0.3)
        confidence += color_info["uniformity"] * 0.3

        # Common button color bonus (0.0 to 0.25)
        if color_info["is_common_button_color"]:
            confidence += 0.25

        # Contrast with background (0.0 to 0.25)
        # Normalize contrast (typical range 0-100)
        contrast_score = min(1.0, contrast / 80)
        confidence += contrast_score * 0.25

        # Aspect ratio score (0.0 to 0.15)
        # Optimal aspect ratio is around 3.0
        optimal_ratio = 3.0
        ratio_diff = abs(aspect_ratio - optimal_ratio)
        ratio_score = max(0, 1.0 - (ratio_diff / 3.0))
        confidence += ratio_score * 0.15

        # Size appropriateness (0.0 to 0.05)
        # Prefer buttons around 150px wide and 40px tall
        width_score = 1.0 - abs(width - 150) / 150.0
        height_score = 1.0 - abs(height - 40) / 40.0
        size_score = (max(0, width_score) + max(0, height_score)) / 2.0
        confidence += size_score * 0.05

        return min(1.0, max(0.0, confidence))  # type: ignore[no-any-return]
