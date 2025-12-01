"""
Contour-based text detector with heuristics.

This analyzer finds contours and applies text-specific heuristics to identify
text regions. It uses multiple features like aspect ratio, rectangularity,
and spatial arrangement.

Performance: 40-120ms
Accuracy: 75-85% for text with clear boundaries
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


class ContourTextDetector(BaseRegionAnalyzer):
    """Detects text regions using contour analysis with text heuristics."""

    @property
    def analysis_type(self) -> RegionAnalysisType:
        return RegionAnalysisType.EDGE_DETECTION

    @property
    def name(self) -> str:
        return "contour_text_detector"

    @property
    def supported_region_types(self) -> list[RegionType]:
        return [RegionType.TEXT_AREA]

    @property
    def version(self) -> str:
        return "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize Contour text detector.

        Args:
            min_contour_area: Minimum contour area
            max_contour_area: Maximum contour area
            min_aspect_ratio: Minimum aspect ratio
            max_aspect_ratio: Maximum aspect ratio
            min_rectangularity: Minimum rectangularity (how box-like)
            min_height: Minimum contour height
            max_height: Maximum contour height
            group_nearby: Whether to group nearby contours
            max_height_ratio: Max height ratio for grouping
            max_spacing_ratio: Max spacing ratio for grouping
        """
        super().__init__(config)

        params = self.get_default_parameters()
        if config:
            params.update(config)

        self.min_contour_area = params["min_contour_area"]
        self.max_contour_area = params["max_contour_area"]
        self.min_aspect_ratio = params["min_aspect_ratio"]
        self.max_aspect_ratio = params["max_aspect_ratio"]
        self.min_rectangularity = params["min_rectangularity"]
        self.min_height = params["min_height"]
        self.max_height = params["max_height"]
        self.group_nearby = params["group_nearby"]
        self.max_height_ratio = params["max_height_ratio"]
        self.max_spacing_ratio = params["max_spacing_ratio"]

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "min_contour_area": 100,
            "max_contour_area": 20000,
            "min_aspect_ratio": 0.2,
            "max_aspect_ratio": 8.0,
            "min_rectangularity": 0.4,
            "min_height": 10,
            "max_height": 200,
            "group_nearby": True,
            "max_height_ratio": 2.0,
            "max_spacing_ratio": 3.0,
        }

    async def analyze(self, input_data: RegionAnalysisInput) -> RegionAnalysisResult:
        """Detect text regions using contour analysis."""
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
                "min_contour_area": self.min_contour_area,
                "group_nearby": self.group_nearby,
                "total_text_regions": len(all_regions),
            },
        )

    def _detect_text_regions(self, gray: np.ndarray, screenshot_index: int) -> list[DetectedRegion]:
        """Detect text regions in a grayscale image."""
        # Apply bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours using text heuristics
        character_candidates = []

        for i, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            # Filter by area
            if area < self.min_contour_area or area > self.max_contour_area:
                continue

            # Filter by height
            if h < self.min_height or h > self.max_height:
                continue

            # Filter by aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

            # Calculate rectangularity (how well the contour fits its bounding box)
            rect_area = w * h
            rectangularity = area / rect_area if rect_area > 0 else 0

            if rectangularity < self.min_rectangularity:
                continue

            # Calculate solidity (ratio of contour area to convex hull area)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            # Text characters should have reasonable solidity
            if solidity < 0.3:
                continue

            # Get parent/child relationships (for filtering nested contours)
            has_parent = hierarchy[0][i][3] != -1 if hierarchy is not None else False

            character_candidates.append(
                {
                    "bbox": (x, y, w, h),
                    "area": area,
                    "aspect_ratio": aspect_ratio,
                    "rectangularity": rectangularity,
                    "solidity": solidity,
                    "has_parent": has_parent,
                }
            )

        # Group nearby characters into text regions if enabled
        if self.group_nearby and character_candidates:
            text_regions = self._group_characters(character_candidates)
        else:
            # Return individual characters as regions
            text_regions = [
                {
                    "bbox": c["bbox"],
                    "chars": [c],
                    "confidence": (c["rectangularity"] * 0.5 + c["solidity"] * 0.5) * 0.6,
                }
                for c in character_candidates
            ]

        # Convert to DetectedRegion objects
        detected_regions = []

        for region_data in text_regions:
            x, y, w, h = region_data["bbox"]
            confidence = region_data["confidence"]
            char_count = len(region_data["chars"])

            detected_region = DetectedRegion(
                bounding_box=BoundingBox(x, y, w, h),
                confidence=min(confidence, 1.0),
                region_type=RegionType.TEXT_AREA,
                label=f"contour_text_{char_count}_chars",
                screenshot_index=screenshot_index,
                metadata={
                    "character_count": char_count,
                    "avg_aspect_ratio": float(
                        np.mean([c["aspect_ratio"] for c in region_data["chars"]])
                    ),
                    "avg_rectangularity": float(
                        np.mean([c["rectangularity"] for c in region_data["chars"]])
                    ),
                    "detection_method": "contour_heuristics",
                    "total_text_regions": len(text_regions),
                },
            )
            detected_regions.append(detected_region)

        return detected_regions

    def _group_characters(self, characters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Group nearby characters into text regions."""
        if not characters:
            return []

        # Sort by x-coordinate (left to right)
        sorted_chars = sorted(characters, key=lambda c: c["bbox"][0])

        # Group characters into lines
        lines = []
        current_line = [sorted_chars[0]]

        for char in sorted_chars[1:]:
            # Check if this character belongs to the current line
            if self._can_group(current_line, char):
                current_line.append(char)
            else:
                if len(current_line) >= 2:  # Only keep lines with 2+ characters
                    lines.append(current_line)
                current_line = [char]

        # Add last line
        if len(current_line) >= 2:
            lines.append(current_line)

        # Create bounding boxes for lines
        text_regions = []

        for line in lines:
            # Calculate bounding box for the line
            min_x = min(c["bbox"][0] for c in line)
            min_y = min(c["bbox"][1] for c in line)
            max_x = max(c["bbox"][0] + c["bbox"][2] for c in line)
            max_y = max(c["bbox"][1] + c["bbox"][3] for c in line)

            # Calculate confidence based on number of characters and consistency
            avg_rect = np.mean([c["rectangularity"] for c in line])
            avg_solid = np.mean([c["solidity"] for c in line])
            char_score = min(len(line) / 5, 1.0)

            confidence = (avg_rect * 0.3 + avg_solid * 0.3 + char_score * 0.4) * 0.75

            text_regions.append(
                {
                    "bbox": (min_x, min_y, max_x - min_x, max_y - min_y),
                    "chars": line,
                    "confidence": confidence,
                }
            )

        return text_regions

    def _can_group(self, line: list[dict[str, Any]], char: dict[str, Any]) -> bool:
        """Check if a character can be grouped with a line."""
        if not line:
            return True

        # Get average height of current line
        avg_height = np.mean([c["bbox"][3] for c in line])

        # Check height similarity
        char_height = char["bbox"][3]
        height_ratio = max(char_height, avg_height) / (min(char_height, avg_height) + 1e-5)

        if height_ratio > self.max_height_ratio:
            return False

        # Check vertical alignment (y-coordinate should be similar)
        avg_y = np.mean([c["bbox"][1] for c in line])
        y_diff = abs(char["bbox"][1] - avg_y)

        if y_diff > avg_height * 0.5:
            return False

        # Check horizontal spacing
        rightmost = max(c["bbox"][0] + c["bbox"][2] for c in line)
        spacing = char["bbox"][0] - rightmost

        # Spacing should be reasonable (not too far)
        if spacing > avg_height * self.max_spacing_ratio:
            return False

        # Spacing should not be negative (overlapping - unless slight)
        if spacing < -avg_height * 0.2:
            return False

        return True
