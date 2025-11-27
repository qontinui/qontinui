"""
Edge + Morphological text detector.

This analyzer uses edge detection combined with morphological operations to find
text regions. It's fast and effective for text with clear edges.

Performance: 20-80ms (very fast)
Accuracy: 70-80% for clear text with good edges
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


class EdgeMorphologyTextDetector(BaseRegionAnalyzer):
    """Detects text regions using edge detection and morphological operations."""

    @property
    def analysis_type(self) -> RegionAnalysisType:
        return RegionAnalysisType.EDGE_DETECTION

    @property
    def name(self) -> str:
        return "edge_morphology_text_detector"

    @property
    def supported_region_types(self) -> list[RegionType]:
        return [RegionType.TEXT_AREA]

    @property
    def version(self) -> str:
        return "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Edge + Morphology text detector.

        Args:
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            morph_kernel_width: Width of morphological kernel
            morph_kernel_height: Height of morphological kernel
            close_iterations: Iterations for morphological closing
            dilate_iterations: Iterations for dilation
            min_area: Minimum region area
            max_area: Maximum region area
            min_aspect_ratio: Minimum aspect ratio
            max_aspect_ratio: Maximum aspect ratio
        """
        super().__init__(config)

        params = self.get_default_parameters()
        if config:
            params.update(config)

        self.canny_low = params["canny_low"]
        self.canny_high = params["canny_high"]
        self.morph_kernel_width = params["morph_kernel_width"]
        self.morph_kernel_height = params["morph_kernel_height"]
        self.close_iterations = params["close_iterations"]
        self.dilate_iterations = params["dilate_iterations"]
        self.min_area = params["min_area"]
        self.max_area = params["max_area"]
        self.min_aspect_ratio = params["min_aspect_ratio"]
        self.max_aspect_ratio = params["max_aspect_ratio"]

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "canny_low": 50,
            "canny_high": 150,
            "morph_kernel_width": 20,
            "morph_kernel_height": 3,
            "close_iterations": 2,
            "dilate_iterations": 1,
            "min_area": 200,
            "max_area": 50000,
            "min_aspect_ratio": 1.5,
            "max_aspect_ratio": 15.0,
        }

    async def analyze(self, input_data: RegionAnalysisInput) -> RegionAnalysisResult:
        """Detect text regions using edge detection and morphology."""
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
                "canny_low": self.canny_low,
                "canny_high": self.canny_high,
                "morph_kernel": f"{self.morph_kernel_width}x{self.morph_kernel_height}",
            },
        )

    def _detect_text_regions(self, gray: np.ndarray, screenshot_index: int) -> list[DetectedRegion]:
        """Detect text regions in a grayscale image."""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # Create morphological kernel (wider than tall to connect horizontal text)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.morph_kernel_width, self.morph_kernel_height)
        )

        # Morphological closing to connect text regions
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=self.close_iterations)

        # Additional dilation to ensure text is connected
        if self.dilate_iterations > 0:
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.dilate(closed, dilate_kernel, iterations=self.dilate_iterations)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_regions = []

        for i, contour in enumerate(contours):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue

            # Filter by aspect ratio (text is usually wider than tall)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

            # Calculate additional features for confidence
            contour_area = cv2.contourArea(contour)
            rect_area = w * h
            extent = contour_area / rect_area if rect_area > 0 else 0

            # Text regions should fill the bounding box reasonably well
            if extent < 0.2:
                continue

            # Calculate edge density in this region
            roi = edges[y : y + h, x : x + w]
            edge_density = np.count_nonzero(roi) / (w * h) if (w * h) > 0 else 0

            # Text should have moderate edge density
            if edge_density < 0.05:
                continue

            # Calculate confidence based on features
            # Prefer aspect ratios around 4:1 to 8:1 for text lines
            aspect_score = 1.0 - min(abs(aspect_ratio - 6.0) / 10.0, 1.0)
            extent_score = min(extent * 2, 1.0)
            edge_score = min(edge_density * 10, 1.0)

            confidence = (aspect_score * 0.4 + extent_score * 0.3 + edge_score * 0.3) * 0.7

            detected_region = DetectedRegion(
                bounding_box=BoundingBox(x, y, w, h),
                confidence=min(confidence, 1.0),
                region_type=RegionType.TEXT_AREA,
                label=f"edge_text_{i}",
                screenshot_index=screenshot_index,
                metadata={
                    "area": int(area),
                    "aspect_ratio": float(aspect_ratio),
                    "extent": float(extent),
                    "edge_density": float(edge_density),
                    "detection_method": "edge_morphology",
                },
            )
            detected_regions.append(detected_region)

        return detected_regions
