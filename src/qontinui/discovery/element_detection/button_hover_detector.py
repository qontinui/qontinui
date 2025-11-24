"""
Button Hover Detector

Detects buttons by comparing multiple screenshots to identify hover state changes:
- Compares multiple screenshots (e.g., normal state vs. hover state)
- Detects regions that change appearance (color/brightness)
- Identifies interactive element changes
- Requires 2+ screenshots to function

This is particularly effective for detecting interactive elements that change
appearance on hover, which is a strong indicator of clickability.
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


class ButtonHoverDetector(BaseAnalyzer):
    """
    Detects buttons by identifying hover state changes across screenshots

    Algorithm:
    1. Compare pairs of screenshots
    2. Detect regions with appearance changes (color, brightness)
    3. Filter by size and shape (button-like dimensions)
    4. Identify consistent changes (not random noise)
    5. High confidence for elements that change on hover
    """

    @property
    def analysis_type(self) -> AnalysisType:
        return AnalysisType.CUSTOM

    @property
    def name(self) -> str:
        return "button_hover_detector"

    @property
    def supports_multi_screenshot(self) -> bool:
        return True

    @property
    def required_screenshots(self) -> int:
        return 2  # Needs at least 2 screenshots to compare

    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            # Difference detection parameters
            "min_color_change": 15,  # Minimum color difference to detect change
            "min_brightness_change": 10,  # Minimum brightness change
            "change_area_threshold": 0.3,  # % of region that must change
            # Morphological operations
            "morph_kernel_size": 5,  # For cleaning up difference mask
            # Size constraints (typical button dimensions)
            "min_width": 60,
            "max_width": 400,
            "min_height": 25,
            "max_height": 80,
            "min_area": 1500,
            # Shape constraints
            "min_aspect_ratio": 1.5,
            "max_aspect_ratio": 6.0,
            # Confidence thresholds
            "min_confidence": 0.6,  # Hover detection is high confidence
            # Comparison strategy
            "compare_all_pairs": False,  # If true, compare all pairs; else consecutive only
        }

    async def analyze(self, input_data: AnalysisInput) -> AnalysisResult:
        """Perform hover state button detection"""
        logger.info(
            f"Running button hover detection on {len(input_data.screenshots)} screenshots"
        )

        params = {**self.get_default_parameters(), **input_data.parameters}

        # Check minimum screenshots
        if len(input_data.screenshots) < self.required_screenshots:
            logger.warning(
                f"Hover detector requires at least {self.required_screenshots} screenshots, "
                f"got {len(input_data.screenshots)}"
            )
            return AnalysisResult(
                analyzer_type=self.analysis_type,
                analyzer_name=self.name,
                elements=[],
                confidence=0.0,
                metadata={
                    "error": "insufficient_screenshots",
                    "required": self.required_screenshots,
                    "provided": len(input_data.screenshots),
                },
            )

        # Load images
        images_gray = self._load_images_grayscale(input_data.screenshot_data)
        images_color = self._load_images_color(input_data.screenshot_data)

        # Compare screenshots to find changes
        all_elements = []

        if params["compare_all_pairs"]:
            # Compare all pairs of screenshots
            for i in range(len(images_gray)):
                for j in range(i + 1, len(images_gray)):
                    elements = await self._compare_screenshots(
                        images_gray[i],
                        images_color[i],
                        images_gray[j],
                        images_color[j],
                        i,
                        j,
                        params,
                    )
                    all_elements.extend(elements)
        else:
            # Compare consecutive screenshots only
            for i in range(len(images_gray) - 1):
                elements = await self._compare_screenshots(
                    images_gray[i],
                    images_color[i],
                    images_gray[i + 1],
                    images_color[i + 1],
                    i,
                    i + 1,
                    params,
                )
                all_elements.extend(elements)

        # Merge overlapping detections from different comparisons
        merged_elements = self._merge_overlapping_elements(all_elements)

        logger.info(
            f"Detected {len(merged_elements)} button candidates using hover state analysis"
        )

        return AnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            elements=merged_elements,
            confidence=0.85,  # Hover detection is highly reliable
            metadata={
                "num_screenshots": len(images_gray),
                "method": "hover_detection",
                "parameters": params,
                "detector_type": "button_hover",
                "num_comparisons": (
                    len(images_gray) - 1
                    if not params["compare_all_pairs"]
                    else (len(images_gray) * (len(images_gray) - 1)) // 2
                ),
            },
        )

    def _load_images_grayscale(self, screenshot_data: List[bytes]) -> List[np.ndarray]:
        """Load screenshots as grayscale"""
        images = []
        for data in screenshot_data:
            img = Image.open(BytesIO(data)).convert("L")
            images.append(np.array(img, dtype=np.uint8))
        return images

    def _load_images_color(self, screenshot_data: List[bytes]) -> List[np.ndarray]:
        """Load screenshots in color (BGR for OpenCV)"""
        images = []
        for data in screenshot_data:
            img = Image.open(BytesIO(data)).convert("RGB")
            images.append(
                cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            )
        return images

    async def _compare_screenshots(
        self,
        img1_gray: np.ndarray,
        img1_color: np.ndarray,
        img2_gray: np.ndarray,
        img2_color: np.ndarray,
        idx1: int,
        idx2: int,
        params: Dict[str, Any],
    ) -> List[DetectedElement]:
        """
        Compare two screenshots to detect hover state changes

        Returns list of detected buttons that changed between screenshots
        """
        elements = []

        # Ensure images are the same size
        if img1_gray.shape != img2_gray.shape:
            logger.warning(
                f"Screenshots {idx1} and {idx2} have different sizes, skipping comparison"
            )
            return elements

        # Step 1: Calculate absolute difference
        diff_gray = cv2.absdiff(img1_gray, img2_gray)
        diff_color = cv2.absdiff(img1_color, img2_color)

        # Step 2: Threshold the difference to get changed regions
        _, diff_mask = cv2.threshold(
            diff_gray, params["min_brightness_change"], 255, cv2.THRESH_BINARY
        )

        # For color difference, check if any channel changed significantly
        color_diff_max = np.max(diff_color, axis=2)
        _, color_mask = cv2.threshold(
            color_diff_max, params["min_color_change"], 255, cv2.THRESH_BINARY
        )

        # Combine brightness and color change masks
        combined_mask = cv2.bitwise_or(diff_mask, color_mask)

        # Step 3: Clean up the mask with morphological operations
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (params["morph_kernel_size"], params["morph_kernel_size"])
        )

        # Close small gaps
        combined_mask = cv2.morphologyEx(
            combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2
        )

        # Remove small noise
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Step 4: Find contours of changed regions
        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        logger.debug(
            f"Found {len(contours)} changed regions between screenshots {idx1} and {idx2}"
        )

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Step 5: Filter by size
            if not (params["min_width"] <= w <= params["max_width"]):
                continue
            if not (params["min_height"] <= h <= params["max_height"]):
                continue
            if area < params["min_area"]:
                continue

            # Step 6: Filter by aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if not (
                params["min_aspect_ratio"] <= aspect_ratio <= params["max_aspect_ratio"]
            ):
                continue

            # Step 7: Analyze the change in this region
            change_info = self._analyze_region_change(
                img1_color[y : y + h, x : x + w],
                img2_color[y : y + h, x : x + w],
                combined_mask[y : y + h, x : x + w],
                params,
            )

            # Check if enough of the region changed (not just edges)
            change_percentage = change_info["change_percentage"]
            if change_percentage < params["change_area_threshold"]:
                continue

            # Step 8: Calculate confidence
            confidence = self._calculate_confidence(
                w, h, aspect_ratio, change_info, params
            )

            if confidence < params["min_confidence"]:
                continue

            # Create detected element
            # Use the screenshot index where the button is most prominent
            screenshot_idx = (
                idx1
                if change_info["avg_brightness_1"] > change_info["avg_brightness_2"]
                else idx2
            )

            elements.append(
                DetectedElement(
                    bounding_box=BoundingBox(
                        x=int(x), y=int(y), width=int(w), height=int(h)
                    ),
                    confidence=confidence,
                    label="Button (hover detected)",
                    element_type="button",
                    screenshot_index=screenshot_idx,
                    metadata={
                        "method": "hover_detection",
                        "compared_screenshots": [int(idx1), int(idx2)],
                        "change_info": change_info,
                    },
                )
            )

        return elements

    def _analyze_region_change(
        self,
        region1: np.ndarray,
        region2: np.ndarray,
        change_mask: np.ndarray,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Analyze how a region changed between two screenshots

        Returns:
            Dict with change statistics
        """
        if region1.size == 0 or region2.size == 0:
            return {
                "change_percentage": 0.0,
                "avg_brightness_1": 0.0,
                "avg_brightness_2": 0.0,
                "avg_color_change": 0.0,
            }

        # Calculate percentage of region that changed
        total_pixels = change_mask.size
        changed_pixels = np.count_nonzero(change_mask)
        change_percentage = changed_pixels / total_pixels if total_pixels > 0 else 0

        # Calculate average brightness in each region
        gray1 = cv2.cvtColor(region1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(region2, cv2.COLOR_BGR2GRAY)
        avg_brightness_1 = np.mean(gray1)
        avg_brightness_2 = np.mean(gray2)

        # Calculate average color change
        color_diff = cv2.absdiff(region1, region2)
        avg_color_change = np.mean(color_diff)

        # Determine change type
        brightness_diff = avg_brightness_2 - avg_brightness_1
        if brightness_diff > 5:
            change_type = "brightened"  # Typical for hover state
        elif brightness_diff < -5:
            change_type = "darkened"
        else:
            change_type = "color_changed"

        return {
            "change_percentage": float(change_percentage),
            "avg_brightness_1": float(avg_brightness_1),
            "avg_brightness_2": float(avg_brightness_2),
            "brightness_difference": float(brightness_diff),
            "avg_color_change": float(avg_color_change),
            "change_type": change_type,
        }

    def _calculate_confidence(
        self,
        width: int,
        height: int,
        aspect_ratio: float,
        change_info: Dict[str, Any],
        params: Dict[str, Any],
    ) -> float:
        """
        Calculate confidence score for hover-based button detection

        Factors:
        - Change percentage: how much of the region changed
        - Change magnitude: how much it changed
        - Size and aspect ratio: typical button proportions
        """
        confidence = 0.5  # Base confidence (hover detection starts high)

        # Change percentage bonus (0.0 to 0.25)
        change_pct = change_info["change_percentage"]
        # Ideal: 30-70% of button changes (not everything, not just border)
        if 0.3 <= change_pct <= 0.7:
            confidence += 0.25
        elif change_pct > 0.7:
            confidence += 0.15  # Might be too much change
        else:
            confidence += change_pct * 0.3  # Partial credit

        # Change magnitude bonus (0.0 to 0.15)
        color_change = change_info["avg_color_change"]
        change_score = min(1.0, color_change / 50)  # Normalize to 0-1
        confidence += change_score * 0.15

        # Aspect ratio bonus (0.0 to 0.1)
        # Optimal aspect ratio is around 3.0
        optimal_ratio = 3.0
        ratio_diff = abs(aspect_ratio - optimal_ratio)
        ratio_score = max(0, 1.0 - (ratio_diff / 3.0))
        confidence += ratio_score * 0.1

        return min(1.0, max(0.0, confidence))

    def _merge_overlapping_elements(
        self, elements: List[DetectedElement]
    ) -> List[DetectedElement]:
        """
        Merge overlapping button detections from different screenshot comparisons

        If the same button was detected in multiple comparisons, merge them
        """
        if not elements:
            return []

        merged = []

        for element in elements:
            # Check if this overlaps with any existing merged element
            overlapped = False

            for i, existing in enumerate(merged):
                # Only merge if from same screenshot
                if element.screenshot_index != existing.screenshot_index:
                    continue

                if element.bounding_box.iou(existing.bounding_box) > 0.5:
                    # Merge - use higher confidence and average bounding box
                    if element.confidence > existing.confidence:
                        best_element = element
                    else:
                        best_element = existing

                    # Average bounding box
                    avg_bbox = BoundingBox(
                        x=(element.bounding_box.x + existing.bounding_box.x) // 2,
                        y=(element.bounding_box.y + existing.bounding_box.y) // 2,
                        width=(element.bounding_box.width + existing.bounding_box.width)
                        // 2,
                        height=(
                            element.bounding_box.height + existing.bounding_box.height
                        )
                        // 2,
                    )

                    # Create merged element with combined metadata
                    compared_screenshots = set(
                        element.metadata.get("compared_screenshots", [])
                        + existing.metadata.get("compared_screenshots", [])
                    )

                    merged_element = DetectedElement(
                        bounding_box=avg_bbox,
                        confidence=max(element.confidence, existing.confidence),
                        label=best_element.label,
                        element_type=best_element.element_type,
                        screenshot_index=best_element.screenshot_index,
                        metadata={
                            **best_element.metadata,
                            "compared_screenshots": sorted(compared_screenshots),
                            "num_detections": existing.metadata.get("num_detections", 1)
                            + 1,
                        },
                    )

                    merged[i] = merged_element
                    overlapped = True
                    break

            if not overlapped:
                merged.append(element)

        return merged
