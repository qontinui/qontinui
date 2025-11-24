"""
Connected Components text detector.

This analyzer uses connected component analysis with morphological operations to
detect text regions. It works by finding connected components that have text-like
properties (aspect ratio, size, spacing).

Performance: 30-100ms
Accuracy: 75-85% for clear text with good contrast
"""

from io import BytesIO
from typing import Any, Dict, List, Optional

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


class ConnectedComponentsTextDetector(BaseRegionAnalyzer):
    """Detects text regions using connected component analysis."""

    @property
    def analysis_type(self) -> RegionAnalysisType:
        return RegionAnalysisType.EDGE_DETECTION

    @property
    def name(self) -> str:
        return "connected_components_text_detector"

    @property
    def supported_region_types(self) -> List[RegionType]:
        return [RegionType.TEXT_AREA]

    @property
    def version(self) -> str:
        return "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Connected Components text detector.

        Args:
            config: Optional configuration parameters
        """
        super().__init__(config)

        params = self.get_default_parameters()
        if config:
            params.update(config)

        self.min_area = params["min_area"]
        self.max_area = params["max_area"]
        self.min_aspect_ratio = params["min_aspect_ratio"]
        self.max_aspect_ratio = params["max_aspect_ratio"]
        self.min_solidity = params["min_solidity"]
        self.morph_kernel_size = params["morph_kernel_size"]
        self.dilation_iterations = params["dilation_iterations"]
        self.use_adaptive_threshold = params["use_adaptive_threshold"]

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "min_area": 50,
            "max_area": 10000,
            "min_aspect_ratio": 0.1,
            "max_aspect_ratio": 10.0,
            "min_solidity": 0.3,
            "morph_kernel_size": 3,
            "dilation_iterations": 2,
            "use_adaptive_threshold": True,
        }

    async def analyze(self, input_data: RegionAnalysisInput) -> RegionAnalysisResult:
        """Detect text regions using connected components."""
        all_regions = []

        # Process each screenshot
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
            sum(r.confidence for r in all_regions) / len(all_regions)
            if all_regions
            else 0.0
        )

        return RegionAnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            regions=all_regions,
            confidence=overall_confidence,
            metadata={
                "min_area": self.min_area,
                "max_area": self.max_area,
                "use_adaptive_threshold": self.use_adaptive_threshold,
                "total_text_regions": len(all_regions),
            },
        )

    def _detect_text_regions(
        self, gray: np.ndarray, screenshot_index: int
    ) -> List[DetectedRegion]:
        """Detect text regions in a grayscale image."""
        # Threshold the image
        if self.use_adaptive_threshold:
            # Adaptive threshold for varying lighting
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
        else:
            # Simple Otsu threshold
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

        # Morphological operations to connect text characters
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.morph_kernel_size, self.morph_kernel_size)
        )

        # Dilate to connect nearby text
        dilated = cv2.dilate(binary, kernel, iterations=self.dilation_iterations)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            dilated, connectivity=8
        )

        detected_regions = []

        # Skip label 0 (background)
        for label in range(1, num_labels):
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]

            # Filter by area
            if area < self.min_area or area > self.max_area:
                continue

            # Filter by aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if (
                aspect_ratio < self.min_aspect_ratio
                or aspect_ratio > self.max_aspect_ratio
            ):
                continue

            # Calculate solidity (compactness)
            component_mask = (labels == label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                continue

            contour = contours[0]
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)

            if hull_area == 0:
                continue

            solidity = area / hull_area

            # Filter by solidity
            if solidity < self.min_solidity:
                continue

            # Calculate confidence based on properties
            # Higher solidity and moderate aspect ratio = higher confidence
            aspect_score = 1.0 - min(
                abs(aspect_ratio - 3.0) / 7.0, 1.0
            )  # Prefer ~3:1 ratio
            solidity_score = solidity
            confidence = (aspect_score * 0.4 + solidity_score * 0.6) * 0.75

            detected_region = DetectedRegion(
                bounding_box=BoundingBox(x, y, w, h),
                confidence=min(confidence, 1.0),
                region_type=RegionType.TEXT_AREA,
                label=f"cc_text_{label}",
                screenshot_index=screenshot_index,
                metadata={
                    "area": int(area),
                    "aspect_ratio": float(aspect_ratio),
                    "solidity": float(solidity),
                    "detection_method": "connected_components",
                },
            )
            detected_regions.append(detected_region)

        return detected_regions
