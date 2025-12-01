"""
Modal Dialog Detector

Detects modal dialogs and popup windows.
Characteristics:
- Centered rectangular regions
- Semi-transparent overlays behind them
- Button pairs at bottom (OK/Cancel, Yes/No)
- Title bar at top
- Typically smaller than screen but prominent
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


class ModalDialogDetector(BaseAnalyzer):
    """
    Detects modal dialogs and popup windows

    Algorithm:
    1. Find centered rectangular regions
    2. Detect semi-transparent overlays (darkened background)
    3. Look for button pairs at bottom
    4. Detect title bar at top
    5. Check size constraints (not too small, not full screen)
    """

    @property
    def analysis_type(self) -> AnalysisType:
        return AnalysisType.SINGLE_SHOT

    @property
    def name(self) -> str:
        return "modal_dialog_detector"

    @property
    def supports_multi_screenshot(self) -> bool:
        return True

    @property
    def required_screenshots(self) -> int:
        return 1

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "min_width": 250,
            "max_width_ratio": 0.8,  # Max 80% of screen width
            "min_height": 150,
            "max_height_ratio": 0.8,
            "center_tolerance": 0.2,  # How centered it must be (0-1)
            "detect_overlay": True,
            "detect_buttons": True,
            "detect_title_bar": True,
            "edge_threshold_low": 50,
            "edge_threshold_high": 150,
        }

    async def analyze(self, input_data: AnalysisInput) -> AnalysisResult:
        """Detect modal dialogs in screenshots"""
        logger.info(
            f"Running modal dialog detection on "
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

        logger.info(f"Detected {len(all_elements)} modal dialogs")

        return AnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            elements=all_elements,
            confidence=0.79,
            metadata={
                "num_screenshots": len(images_color),
                "method": "modal_dialog_detection",
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
        """Analyze a single screenshot for modal dialogs"""
        elements = []

        height, width = img_gray.shape

        # Apply edge detection
        edges = cv2.Canny(
            img_gray, params["edge_threshold_low"], params["edge_threshold_high"]
        )

        # Dilate to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Size constraints
            max_width = int(width * params["max_width_ratio"])
            max_height = int(height * params["max_height_ratio"])

            if not (params["min_width"] <= w <= max_width):
                continue
            if not (params["min_height"] <= h <= max_height):
                continue

            # Modal should not be full screen
            if w > width * 0.95 or h > height * 0.95:
                continue

            # Check if centered
            center_x = x + w / 2
            center_y = y + h / 2
            screen_center_x = width / 2
            screen_center_y = height / 2

            x_offset = abs(center_x - screen_center_x) / width
            y_offset = abs(center_y - screen_center_y) / height

            # Modals are typically centered
            is_centered = (
                x_offset <= params["center_tolerance"]
                and y_offset <= params["center_tolerance"]
            )

            # Extract dialog region
            dialog_region = (
                img_gray[y : y + h, x : x + w]
                if y + h <= height and x + w <= width
                else None
            )
            if dialog_region is None or dialog_region.size == 0:
                continue

            # Detect features
            has_title_bar = False
            has_buttons = False

            if params["detect_title_bar"]:
                has_title_bar = self._detect_title_bar(dialog_region)

            if params["detect_buttons"]:
                has_buttons = self._detect_bottom_buttons(dialog_region)

            # Check for overlay (background should be darker around dialog)
            has_overlay = False
            if params["detect_overlay"]:
                has_overlay = self._detect_overlay(img_gray, x, y, w, h)

            # Calculate confidence
            confidence = self._calculate_confidence(
                is_centered,
                has_title_bar,
                has_buttons,
                has_overlay,
                w,
                h,
                width,
                height,
            )

            if confidence < 0.55:
                continue

            elements.append(
                DetectedElement(
                    bounding_box=BoundingBox(
                        x=int(x), y=int(y), width=int(w), height=int(h)
                    ),
                    confidence=confidence,
                    label="Modal Dialog",
                    element_type="dialog",
                    screenshot_index=screenshot_idx,
                    metadata={
                        "method": "modal_dialog_detection",
                        "is_centered": bool(is_centered),
                        "has_title_bar": bool(has_title_bar),
                        "has_buttons": bool(has_buttons),
                        "has_overlay": bool(has_overlay),
                        "center_offset_x": float(x_offset),
                        "center_offset_y": float(y_offset),
                    },
                )
            )

        return elements

    def _detect_title_bar(self, dialog_region: np.ndarray) -> bool:
        """Detect if dialog has a title bar at top"""
        h, w = dialog_region.shape

        # Check top portion (top 15% of dialog)
        title_height = min(50, int(h * 0.15))
        title_region = dialog_region[:title_height, :]

        # Title bar typically has different background or edge
        top_edge_density = np.sum(cv2.Canny(title_region, 50, 150))
        total_pixels = title_region.size

        # If there's significant edge activity at top, likely a title bar
        return bool((top_edge_density / total_pixels) > 0.02)

    def _detect_bottom_buttons(self, dialog_region: np.ndarray) -> bool:
        """Detect if dialog has buttons at bottom"""
        h, w = dialog_region.shape

        # Check bottom portion (bottom 20% of dialog)
        button_region_height = min(60, int(h * 0.20))
        button_region = dialog_region[-button_region_height:, :]

        # Apply edge detection
        edges = cv2.Canny(button_region, 50, 150)

        # Find contours (potential buttons)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Look for 1-3 rectangular regions (buttons)
        button_count = 0
        for contour in contours:
            x, y, w_btn, h_btn = cv2.boundingRect(contour)

            # Button-like dimensions
            if 50 <= w_btn <= 150 and 20 <= h_btn <= 50:
                aspect_ratio = w_btn / h_btn if h_btn > 0 else 0
                if 1.5 <= aspect_ratio <= 5.0:
                    button_count += 1

        # Dialogs typically have 1-3 buttons at bottom
        return 1 <= button_count <= 3

    def _detect_overlay(
        self,
        img_gray: np.ndarray,
        dialog_x: int,
        dialog_y: int,
        dialog_w: int,
        dialog_h: int,
    ) -> bool:
        """Detect if there's a semi-transparent overlay around dialog"""
        h, w = img_gray.shape

        # Sample background regions around dialog
        samples = []

        # Left region
        if dialog_x > 50:
            left_region = img_gray[:, :dialog_x]
            samples.append(np.mean(left_region))

        # Right region
        if dialog_x + dialog_w < w - 50:
            right_region = img_gray[:, dialog_x + dialog_w :]
            samples.append(np.mean(right_region))

        # Top region
        if dialog_y > 50:
            top_region = img_gray[:dialog_y, :]
            samples.append(np.mean(top_region))

        # Bottom region
        if dialog_y + dialog_h < h - 50:
            bottom_region = img_gray[dialog_y + dialog_h :, :]
            samples.append(np.mean(bottom_region))

        if not samples:
            return False

        # Check if background is darker than dialog (overlay effect)
        background_brightness = np.mean(samples)
        dialog_brightness = np.mean(
            img_gray[dialog_y : dialog_y + dialog_h, dialog_x : dialog_x + dialog_w]
        )

        # Background should be 10-30% darker if there's an overlay
        return dialog_brightness > background_brightness * 1.1  # type: ignore[no-any-return]

    def _calculate_confidence(
        self,
        is_centered: bool,
        has_title_bar: bool,
        has_buttons: bool,
        has_overlay: bool,
        w: int,
        h: int,
        screen_w: int,
        screen_h: int,
    ) -> float:
        """Calculate confidence score"""
        confidence = 0.5  # Base confidence

        # Centered position is strong indicator
        if is_centered:
            confidence += 0.2

        # Title bar
        if has_title_bar:
            confidence += 0.15

        # Bottom buttons
        if has_buttons:
            confidence += 0.15

        # Overlay (dimmed background)
        if has_overlay:
            confidence += 0.12

        # Typical modal size (300-600px wide, 200-500px tall)
        if 300 <= w <= 600 and 200 <= h <= 500:
            confidence += 0.1
        elif 250 <= w <= 700 and 150 <= h <= 600:
            confidence += 0.05

        return min(0.95, confidence)
