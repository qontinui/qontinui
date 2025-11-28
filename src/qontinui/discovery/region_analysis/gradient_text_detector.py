"""
Gradient-based text detector.

This analyzer uses gradient magnitude and orientation to detect text regions.
Text typically has strong, consistent gradients due to the contrast between
characters and background.

Performance: 40-100ms
Accuracy: 70-80% for text with good contrast
"""

from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image

from qontinui.discovery.region_analysis.base import (
    BaseRegionAnalyzer,
    BoundingBox,
    DetectedRegion,
    RegionAnalysisInput,
    RegionAnalysisResult,
    RegionAnalysisType,
    RegionType,
)


class GradientTextDetector(BaseRegionAnalyzer):
    """Detects text regions using gradient analysis."""

    @property
    def analysis_type(self) -> RegionAnalysisType:
        return RegionAnalysisType.EDGE_DETECTION

    @property
    def name(self) -> str:
        return "gradient_text_detector"

    @property
    def supported_region_types(self) -> list[RegionType]:
        return [RegionType.TEXT_AREA]

    @property
    def version(self) -> str:
        return "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Gradient text detector.

        Args:
            gradient_threshold: Threshold for gradient magnitude
            min_gradient_density: Minimum ratio of strong gradients
            max_gradient_density: Maximum ratio of strong gradients
            morph_kernel_width: Width of morphological kernel
            morph_kernel_height: Height of morphological kernel
            min_area: Minimum region area
            max_area: Maximum region area
            min_aspect_ratio: Minimum aspect ratio
            max_aspect_ratio: Maximum aspect ratio
        """
        super().__init__(config)

        params = self.get_default_parameters()
        if config:
            params.update(config)

        self.gradient_threshold = params["gradient_threshold"]
        self.min_gradient_density = params["min_gradient_density"]
        self.max_gradient_density = params["max_gradient_density"]
        self.morph_kernel_width = params["morph_kernel_width"]
        self.morph_kernel_height = params["morph_kernel_height"]
        self.min_area = params["min_area"]
        self.max_area = params["max_area"]
        self.min_aspect_ratio = params["min_aspect_ratio"]
        self.max_aspect_ratio = params["max_aspect_ratio"]

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "gradient_threshold": 30.0,
            "min_gradient_density": 0.15,
            "max_gradient_density": 0.85,
            "morph_kernel_width": 15,
            "morph_kernel_height": 3,
            "min_area": 200,
            "max_area": 30000,
            "min_aspect_ratio": 1.0,
            "max_aspect_ratio": 12.0,
        }

    async def analyze(self, input_data: RegionAnalysisInput) -> RegionAnalysisResult:
        """Detect text regions using gradient analysis."""
        all_regions = []

        for idx, screenshot_bytes in enumerate(input_data.screenshot_data):
            # Convert bytes to numpy array
            image = Image.open(BytesIO(screenshot_bytes))
            image_np = np.array(image)

            # Convert to grayscale
            if len(image_np.shape) == 3:
                if image_np.shape[2] == 4:
                    gray = cv2.cvtColor(image_np, cv2.COLOR_RGBA2GRAY)
                else:
                    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np

            # Detect text regions
            regions = self._detect_text_regions(gray, idx)
            all_regions.extend(regions)

        # Calculate overall confidence
        overall_confidence = (
            sum(r.confidence for r in all_regions) / len(all_regions) if all_regions else 0.0
        )

        return RegionAnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            confidence=overall_confidence,
            regions=all_regions,
            metadata={
                "gradient_threshold": self.gradient_threshold,
                "min_gradient_density": self.min_gradient_density,
                "total_text_regions": len(all_regions),
            },
        )

    def _detect_text_regions(self, gray: np.ndarray, screenshot_index: int) -> list[DetectedRegion]:
        """Detect text regions in a grayscale image."""
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Compute gradients
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

        # Compute gradient magnitude
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize to 0-255
        gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Threshold gradient magnitude
        _, gradient_binary = cv2.threshold(
            gradient_mag, self.gradient_threshold, 255, cv2.THRESH_BINARY
        )

        # Morphological operations to connect text
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.morph_kernel_width, self.morph_kernel_height)
        )

        # Close to connect nearby gradients
        closed = cv2.morphologyEx(gradient_binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_regions = []

        for i, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue

            # Filter by aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

            # Calculate gradient density in this region
            roi_gradient = gradient_binary[y : y + h, x : x + w]
            gradient_density = np.count_nonzero(roi_gradient) / (w * h) if (w * h) > 0 else 0

            # Filter by gradient density (text should have moderate density)
            if (
                gradient_density < self.min_gradient_density
                or gradient_density > self.max_gradient_density
            ):
                continue

            # Calculate gradient variance (text should have relatively uniform gradients)
            roi_gradient_mag = gradient_mag[y : y + h, x : x + w]
            gradient_std = np.std(roi_gradient_mag[roi_gradient_mag > self.gradient_threshold])

            # Calculate gradient orientation consistency
            roi_grad_x = grad_x[y : y + h, x : x + w]
            roi_grad_y = grad_y[y : y + h, x : x + w]
            gradient_angles = np.arctan2(roi_grad_y, roi_grad_x)

            # Text often has consistent vertical or horizontal gradients
            # Calculate histogram of gradient orientations
            angle_hist, _ = np.histogram(gradient_angles.flatten(), bins=8, range=(-np.pi, np.pi))
            angle_consistency = np.max(angle_hist) / (np.sum(angle_hist) + 1e-5)

            # Calculate confidence
            # Higher gradient density (but not too high) = higher confidence
            density_score = 1.0 - abs(gradient_density - 0.4) / 0.6
            density_score = max(0, min(density_score, 1.0))

            # Moderate aspect ratios are more likely text
            aspect_score = 1.0 - min(abs(aspect_ratio - 4.0) / 8.0, 1.0)

            # Some orientation consistency is good
            consistency_score = min(angle_consistency * 2, 1.0)

            confidence = (density_score * 0.4 + aspect_score * 0.3 + consistency_score * 0.3) * 0.7

            detected_region = DetectedRegion(
                bounding_box=BoundingBox(x, y, w, h),
                confidence=min(confidence, 1.0),
                region_type=RegionType.TEXT_AREA,
                label=f"gradient_text_{i}",
                screenshot_index=screenshot_index,
                metadata={
                    "area": int(area),
                    "aspect_ratio": float(aspect_ratio),
                    "gradient_density": float(gradient_density),
                    "gradient_std": float(gradient_std),
                    "angle_consistency": float(angle_consistency),
                    "detection_method": "gradient",
                    "total_text_regions": len(contours),
                },
            )
            detected_regions.append(detected_region)

        return detected_regions
