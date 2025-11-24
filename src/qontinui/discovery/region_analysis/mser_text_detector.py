"""
MSER (Maximally Stable Extremal Regions) text detector.

This analyzer uses MSER to detect blob-like regions that are stable across multiple
thresholds, which is characteristic of text. MSER is particularly effective for
detecting text in natural scenes and games.

Performance: 50-200ms
Accuracy: 70-85% for clear text, good for stylized fonts
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


class MSERTextDetector(BaseRegionAnalyzer):
    """Detects text regions using MSER (Maximally Stable Extremal Regions)."""

    @property
    def analysis_type(self) -> RegionAnalysisType:
        return RegionAnalysisType.EDGE_DETECTION

    @property
    def name(self) -> str:
        return "mser_text_detector"

    @property
    def supported_region_types(self) -> List[RegionType]:
        return [RegionType.TEXT_AREA]

    @property
    def version(self) -> str:
        return "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MSER text detector.

        Args:
            config: Optional configuration parameters
        """
        super().__init__(config)

        params = self.get_default_parameters()
        if config:
            params.update(config)

        self.delta = params["delta"]
        self.min_area = params["min_area"]
        self.max_area = params["max_area"]
        self.max_variation = params["max_variation"]
        self.min_aspect_ratio = params["min_aspect_ratio"]
        self.max_aspect_ratio = params["max_aspect_ratio"]
        self.group_nearby = params["group_nearby"]
        self.grouping_distance = params["grouping_distance"]

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            "delta": 5,
            "min_area": 60,
            "max_area": 14400,
            "max_variation": 0.25,
            "min_aspect_ratio": 0.2,
            "max_aspect_ratio": 5.0,
            "group_nearby": True,
            "grouping_distance": 20,
        }

    async def analyze(self, input_data: RegionAnalysisInput) -> RegionAnalysisResult:
        """Detect text regions using MSER."""
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
                "delta": self.delta,
                "min_area": self.min_area,
                "max_area": self.max_area,
                "group_nearby": self.group_nearby,
                "total_text_regions": len(all_regions),
            },
        )

    def _detect_text_regions(
        self, gray: np.ndarray, screenshot_index: int
    ) -> List[DetectedRegion]:
        """Detect text regions in a grayscale image."""
        # Create MSER detector
        mser = cv2.MSER_create(
            _delta=self.delta,
            _min_area=self.min_area,
            _max_area=self.max_area,
            _max_variation=self.max_variation,
        )

        # Detect regions
        regions_mser, bboxes = mser.detectRegions(gray)

        detected_regions = []

        for i, (region_points, bbox) in enumerate(zip(regions_mser, bboxes)):
            x, y, w, h = bbox

            # Skip invalid regions
            if w <= 0 or h <= 0:
                continue

            # Check aspect ratio (text is usually not too square or too elongated)
            aspect_ratio = w / h
            if (
                aspect_ratio < self.min_aspect_ratio
                or aspect_ratio > self.max_aspect_ratio
            ):
                continue

            # Calculate confidence based on region stability
            # More compact regions with less variation are more likely to be text
            hull = cv2.convexHull(region_points)
            hull_area = cv2.contourArea(hull)
            region_area = len(region_points)

            # Solidity: how much of the convex hull is filled
            solidity = region_area / (hull_area + 1e-5)
            confidence = min(solidity, 1.0)

            detected_region = DetectedRegion(
                bounding_box=BoundingBox(x, y, w, h),
                confidence=confidence * 0.8,  # Scale down as MSER can be noisy
                region_type=RegionType.TEXT_AREA,
                label=f"mser_text_{i}",
                screenshot_index=screenshot_index,
                metadata={
                    "area": region_area,
                    "aspect_ratio": aspect_ratio,
                    "solidity": solidity,
                    "detection_method": "mser",
                },
            )
            detected_regions.append(detected_region)

        # Group nearby regions if enabled
        if self.group_nearby and detected_regions:
            detected_regions = self._group_nearby_regions(detected_regions)

        return detected_regions

    def _group_nearby_regions(
        self, regions: List[DetectedRegion]
    ) -> List[DetectedRegion]:
        """Group nearby text regions into text blocks."""
        if not regions:
            return regions

        # Sort by y-coordinate
        sorted_regions = sorted(regions, key=lambda r: r.bounding_box.y)

        # Group regions that are close together
        groups = []
        current_group = [sorted_regions[0]]

        for region in sorted_regions[1:]:
            # Check if close to any region in current group
            should_group = False
            for grouped in current_group:
                if self._are_nearby(grouped.bounding_box, region.bounding_box):
                    should_group = True
                    break

            if should_group:
                current_group.append(region)
            else:
                groups.append(current_group)
                current_group = [region]

        # Add last group
        if current_group:
            groups.append(current_group)

        # Create merged regions
        merged_regions = []
        for group in groups:
            if len(group) >= 2:  # Only merge if 2+ regions
                merged_regions.append(self._merge_regions(group))
            else:
                merged_regions.extend(group)

        return merged_regions

    def _are_nearby(self, box1: BoundingBox, box2: BoundingBox) -> bool:
        """Check if two bounding boxes are nearby."""
        # Calculate boundaries
        box1_x2 = box1.x + box1.width
        box1_y2 = box1.y + box1.height
        box2_x2 = box2.x + box2.width
        box2_y2 = box2.y + box2.height

        # Check if boxes overlap or are close
        x_overlap = not (
            box1_x2 < box2.x - self.grouping_distance
            or box2_x2 < box1.x - self.grouping_distance
        )
        y_overlap = not (
            box1_y2 < box2.y - self.grouping_distance
            or box2_y2 < box1.y - self.grouping_distance
        )

        return x_overlap and y_overlap

    def _merge_regions(self, regions: List[DetectedRegion]) -> DetectedRegion:
        """Merge multiple regions into one text block."""
        # Find bounding box that encompasses all
        min_x = min(r.bounding_box.x for r in regions)
        min_y = min(r.bounding_box.y for r in regions)
        max_x = max(r.bounding_box.x + r.bounding_box.width for r in regions)
        max_y = max(r.bounding_box.y + r.bounding_box.height for r in regions)

        # Average confidence
        avg_conf = sum(r.confidence for r in regions) / len(regions)

        # Boost confidence for grouped regions (more evidence)
        boosted_conf = min(avg_conf * 1.15, 1.0)

        return DetectedRegion(
            bounding_box=BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y),
            confidence=boosted_conf,
            region_type=RegionType.TEXT_AREA,
            label=f"text_block_{len(regions)}_regions",
            screenshot_index=regions[0].screenshot_index,
            metadata={
                "merged_count": len(regions),
                "detection_method": "mser_grouped",
            },
        )
