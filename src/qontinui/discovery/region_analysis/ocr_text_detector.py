"""
OCR-based text region detector using Tesseract.

This analyzer uses Tesseract OCR to detect text in images and creates bounding boxes
around detected text regions. It can detect individual words, lines, or blocks of text.

Performance: 100-500ms depending on image size
Accuracy: 85-95% for clear text, lower for stylized/game fonts
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

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class OCRTextDetector(BaseRegionAnalyzer):
    """Detects text regions using Tesseract OCR."""

    @property
    def analysis_type(self) -> RegionAnalysisType:
        return RegionAnalysisType.ML_CLASSIFICATION

    @property
    def name(self) -> str:
        return "ocr_text_detector"

    @property
    def supported_region_types(self) -> list[RegionType]:
        return [RegionType.TEXT_AREA]

    @property
    def version(self) -> str:
        return "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize OCR text detector.

        Args:
            config: Optional configuration parameters
        """
        super().__init__(config)

        if not TESSERACT_AVAILABLE:
            raise ImportError("pytesseract is required for OCRTextDetector")

        params = self.get_default_parameters()
        if config:
            params.update(config)

        self.min_confidence = params["min_confidence"]
        self.level = params["level"]
        self.merge_nearby = params["merge_nearby"]
        self.merge_distance = params["merge_distance"]

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "min_confidence": 30.0,
            "level": "word",  # word, line, or block
            "merge_nearby": True,
            "merge_distance": 10,
        }

    async def analyze(self, input_data: RegionAnalysisInput) -> RegionAnalysisResult:
        """Detect text regions using OCR."""
        all_regions = []

        # Process each screenshot
        for idx, screenshot_bytes in enumerate(input_data.screenshot_data):
            # Convert bytes to numpy array
            image = Image.open(BytesIO(screenshot_bytes))
            image_np = np.array(image)

            # Convert RGBA to RGB if needed
            if len(image_np.shape) == 3 and image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

            # Detect text regions
            regions = self._detect_text_regions(image_np, idx)
            all_regions.extend(regions)

        # Calculate overall confidence
        overall_confidence = (
            sum(r.confidence for r in all_regions) / len(all_regions) if all_regions else 0.0
        )

        return RegionAnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            regions=all_regions,
            confidence=overall_confidence,
            metadata={
                "min_confidence": self.min_confidence,
                "level": self.level,
                "merge_nearby": self.merge_nearby,
                "total_text_regions": len(all_regions),
            },
        )

    def _detect_text_regions(
        self, image: np.ndarray, screenshot_index: int
    ) -> list[DetectedRegion]:
        """Detect text regions in a single image."""
        # Get OCR data
        data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT,
            config="--psm 11",  # Sparse text
        )

        regions = []
        n_boxes = len(data["text"])

        # Determine which level to use
        level_map = {
            "word": 5,  # Word level
            "line": 4,  # Line level
            "block": 2,  # Block level
        }
        target_level = level_map.get(self.level, 5)

        for i in range(n_boxes):
            # Skip if below target level or low confidence
            if data["level"][i] != target_level:
                continue

            conf = float(data["conf"][i])
            if conf < self.min_confidence:
                continue

            text = data["text"][i].strip()
            if not text:
                continue

            x, y, w, h = (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
            )

            # Skip very small regions
            if w < 5 or h < 5:
                continue

            region = DetectedRegion(
                bounding_box=BoundingBox(x, y, w, h),
                confidence=conf / 100.0,  # Normalize to 0-1
                region_type=RegionType.TEXT_AREA,
                label=f"text: {text[:30]}",
                screenshot_index=screenshot_index,
                metadata={
                    "text": text,
                    "ocr_confidence": conf,
                    "level": self.level,
                    "text_length": len(text),
                },
            )
            regions.append(region)

        # Merge nearby regions if enabled
        if self.merge_nearby and regions:
            regions = self._merge_nearby_regions(regions)

        return regions

    def _merge_nearby_regions(self, regions: list[DetectedRegion]) -> list[DetectedRegion]:
        """Merge nearby text regions into larger blocks."""
        if not regions:
            return regions

        # Sort by y-coordinate (top to bottom)
        sorted_regions = sorted(regions, key=lambda r: r.bounding_box.y)

        merged = []
        current_group = [sorted_regions[0]]

        for region in sorted_regions[1:]:
            # Check if close to any region in current group
            should_merge = False
            for grouped in current_group:
                if self._are_nearby(grouped.bounding_box, region.bounding_box):
                    should_merge = True
                    break

            if should_merge:
                current_group.append(region)
            else:
                # Merge current group and start new one
                merged.append(self._merge_group(current_group))
                current_group = [region]

        # Merge last group
        if current_group:
            merged.append(self._merge_group(current_group))

        return merged

    def _are_nearby(self, box1: BoundingBox, box2: BoundingBox) -> bool:
        """Check if two bounding boxes are nearby."""
        # Check vertical proximity
        box1_y2 = box1.y + box1.height
        box2_y2 = box2.y + box2.height
        vertical_gap = min(abs(box1_y2 - box2.y), abs(box2_y2 - box1.y))

        # Check horizontal proximity
        box1_x2 = box1.x + box1.width
        box2_x2 = box2.x + box2.width
        horizontal_gap = min(abs(box1_x2 - box2.x), abs(box2_x2 - box1.x))

        # Merge if close vertically or horizontally aligned and close
        return vertical_gap <= self.merge_distance or horizontal_gap <= self.merge_distance  # type: ignore[no-any-return]

    def _merge_group(self, group: list[DetectedRegion]) -> DetectedRegion:
        """Merge a group of regions into one."""
        if len(group) == 1:
            return group[0]

        # Find bounding box that encompasses all
        min_x = min(r.bounding_box.x for r in group)
        min_y = min(r.bounding_box.y for r in group)
        max_x = max(r.bounding_box.x + r.bounding_box.width for r in group)
        max_y = max(r.bounding_box.y + r.bounding_box.height for r in group)

        # Average confidence
        avg_conf = sum(r.confidence for r in group) / len(group)

        # Concatenate text
        texts = [r.metadata.get("text", "") for r in group]
        combined_text = " ".join(texts)

        return DetectedRegion(
            bounding_box=BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y),
            confidence=avg_conf,
            region_type=RegionType.TEXT_AREA,
            label=f"text: {combined_text[:30]}",
            screenshot_index=group[0].screenshot_index,
            metadata={
                "text": combined_text,
                "merged_count": len(group),
                "level": "merged",
                "text_length": len(combined_text),
            },
        )
