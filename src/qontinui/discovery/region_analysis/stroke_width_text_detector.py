"""
Stroke Width Transform (SWT) text detector.

This analyzer implements the Stroke Width Transform algorithm, which is specifically
designed for detecting text in natural images. It works by analyzing stroke widths
and grouping strokes with similar widths.

Performance: 200-500ms (more computationally intensive)
Accuracy: 80-90% for text with consistent stroke width
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


class StrokeWidthTextDetector(BaseRegionAnalyzer):
    """Detects text regions using Stroke Width Transform."""

    @property
    def analysis_type(self) -> RegionAnalysisType:
        return RegionAnalysisType.EDGE_DETECTION

    @property
    def name(self) -> str:
        return "stroke_width_text_detector"

    @property
    def supported_region_types(self) -> list[RegionType]:
        return [RegionType.TEXT_AREA]

    @property
    def version(self) -> str:
        return "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize SWT text detector.

        Args:
            dark_on_light: True for dark text on light background, False for inverse
            max_stroke_width: Maximum allowed stroke width
            min_component_area: Minimum area for text components
            max_component_area: Maximum area for text components
            stroke_width_variance_threshold: Max variance ratio for stroke width
            min_letters_in_chain: Minimum letters needed to form a text region
        """
        super().__init__(config)

        params = self.get_default_parameters()
        if config:
            params.update(config)

        self.dark_on_light = params["dark_on_light"]
        self.max_stroke_width = params["max_stroke_width"]
        self.min_component_area = params["min_component_area"]
        self.max_component_area = params["max_component_area"]
        self.stroke_width_variance_threshold = params["stroke_width_variance_threshold"]
        self.min_letters_in_chain = params["min_letters_in_chain"]

    def get_default_parameters(self) -> dict[str, Any]:
        return {
            "dark_on_light": True,
            "max_stroke_width": 50,
            "min_component_area": 100,
            "max_component_area": 15000,
            "stroke_width_variance_threshold": 0.5,
            "min_letters_in_chain": 3,
        }

    async def analyze(self, input_data: RegionAnalysisInput) -> RegionAnalysisResult:
        """Detect text regions using SWT."""
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
            sum(r.confidence for r in all_regions) / len(all_regions)
            if all_regions
            else 0.0
        )

        return RegionAnalysisResult(
            analyzer_type=self.analysis_type,
            analyzer_name=self.name,
            confidence=overall_confidence,
            regions=all_regions,
            metadata={
                "dark_on_light": self.dark_on_light,
                "max_stroke_width": self.max_stroke_width,
                "total_text_regions": len(all_regions),
            },
        )

    def _detect_text_regions(
        self, gray: np.ndarray, screenshot_index: int
    ) -> list[DetectedRegion]:
        """Detect text regions using SWT."""
        # Compute SWT image
        swt_image = self._compute_swt(gray)

        # Find connected components in SWT image
        components = self._find_letter_candidates(swt_image)

        # Chain letters into words
        text_regions = self._chain_letters_to_words(components)

        # Convert to DetectedRegion objects
        detected_regions = []
        for i, region_data in enumerate(text_regions):
            bbox, confidence, metadata = region_data

            detected_region = DetectedRegion(
                bounding_box=BoundingBox(
                    bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                ),
                confidence=confidence,
                region_type=RegionType.TEXT_AREA,
                label=f"swt_text_{i}",
                screenshot_index=screenshot_index,
                metadata=metadata,
            )
            detected_regions.append(detected_region)

        return detected_regions

    def _compute_swt(self, gray: np.ndarray) -> np.ndarray:
        """Compute Stroke Width Transform of the image."""
        # Compute gradients
        if self.dark_on_light:
            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
        else:
            # Invert for light on dark
            inverted = 255 - gray
            edges = cv2.Canny(inverted, 50, 150)

        # Compute gradient direction
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Gradient magnitude and direction
        gradient_mag = np.sqrt(gx**2 + gy**2)
        gradient_dir = np.arctan2(gy, gx)

        # Initialize SWT image with infinity
        swt = np.full(gray.shape, np.inf, dtype=np.float64)

        # Get edge pixel coordinates
        edge_points = np.column_stack(np.where(edges > 0))

        # For each edge pixel, trace ray in gradient direction
        for y, x in edge_points:
            if gradient_mag[y, x] == 0:
                continue

            # Normalize gradient direction
            direction = gradient_dir[y, x]

            # Ray parameters
            step_x = np.cos(direction)
            step_y = np.sin(direction)

            # Trace ray
            ray_points = []
            curr_x, curr_y = float(x), float(y)
            max_steps = self.max_stroke_width

            for _step in range(max_steps):
                curr_x += step_x
                curr_y += step_y

                # Check bounds
                ix, iy = int(round(curr_x)), int(round(curr_y))
                if not (0 <= ix < edges.shape[1] and 0 <= iy < edges.shape[0]):
                    break

                ray_points.append((iy, ix))

                # Check if we hit another edge
                if edges[iy, ix] > 0:
                    # Check if gradients are opposite (text stroke)
                    opposite_dir = gradient_dir[iy, ix]
                    angle_diff = abs(direction - opposite_dir)

                    # Normalize angle difference to [0, pi]
                    while angle_diff > np.pi:
                        angle_diff -= 2 * np.pi
                    angle_diff = abs(angle_diff)

                    # If gradients roughly opposite (within pi/6), this is a stroke
                    if abs(angle_diff - np.pi) < np.pi / 6:
                        stroke_width = len(ray_points)

                        # Update SWT for all points along ray
                        for py, px in ray_points:
                            swt[py, px] = min(swt[py, px], stroke_width)

                    break

        # Replace infinity with 0
        swt[swt == np.inf] = 0

        return swt

    def _find_letter_candidates(self, swt: np.ndarray) -> list[dict[str, Any]]:
        """Find connected components that could be letters."""
        # Threshold SWT to get binary image
        swt_binary = ((swt > 0) & (swt < self.max_stroke_width)).astype(np.uint8) * 255

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            swt_binary, connectivity=8
        )

        letter_candidates = []

        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]

            # Filter by area
            if area < self.min_component_area or area > self.max_component_area:
                continue

            # Get stroke widths for this component
            component_mask = labels == label
            stroke_widths = swt[component_mask]
            stroke_widths = stroke_widths[stroke_widths > 0]

            if len(stroke_widths) == 0:
                continue

            # Calculate stroke width statistics
            mean_sw = np.mean(stroke_widths)
            std_sw = np.std(stroke_widths)
            variance_ratio = std_sw / (mean_sw + 1e-5)

            # Text should have consistent stroke width
            if variance_ratio > self.stroke_width_variance_threshold:
                continue

            # Get bounding box
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            w = stats[label, cv2.CC_STAT_WIDTH]
            h = stats[label, cv2.CC_STAT_HEIGHT]

            # Check aspect ratio (letters shouldn't be too wide or too tall)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue

            letter_candidates.append(
                {
                    "bbox": (x, y, x + w, y + h),
                    "centroid": centroids[label],
                    "mean_stroke_width": mean_sw,
                    "area": area,
                    "aspect_ratio": aspect_ratio,
                    "variance_ratio": variance_ratio,
                }
            )

        return letter_candidates

    def _chain_letters_to_words(self, letters: list[dict[str, Any]]) -> list[tuple]:
        """Chain letter candidates into word regions."""
        if len(letters) < self.min_letters_in_chain:
            return []

        # Group letters by similar stroke width and spatial proximity
        chains = []

        for i, letter in enumerate(letters):
            # Find nearby letters with similar properties
            chain = [letter]

            for j, other in enumerate(letters):
                if i == j:
                    continue

                # Check stroke width similarity
                sw_ratio = letter["mean_stroke_width"] / (
                    other["mean_stroke_width"] + 1e-5
                )
                if sw_ratio < 0.5 or sw_ratio > 2.0:
                    continue

                # Check spatial proximity
                dist = np.linalg.norm(letter["centroid"] - other["centroid"])
                avg_height = (
                    letter["bbox"][3]
                    - letter["bbox"][1]
                    + other["bbox"][3]
                    - other["bbox"][1]
                ) / 2

                # Should be nearby (within 3x height)
                if dist > avg_height * 3:
                    continue

                # Check vertical alignment
                y_diff = abs(letter["centroid"][1] - other["centroid"][1])
                if y_diff > avg_height * 0.5:
                    continue

                chain.append(other)

            # Only keep chains with enough letters
            if len(chain) >= self.min_letters_in_chain:
                chains.append(chain)

        # Merge overlapping chains and create bounding boxes
        text_regions = []
        used = set()

        for chain in sorted(chains, key=len, reverse=True):
            chain_ids = [id(letter) for letter in chain]

            # Skip if letters already used
            if any(cid in used for cid in chain_ids):
                continue

            # Mark as used
            used.update(chain_ids)

            # Calculate bounding box
            min_x = min(letter["bbox"][0] for letter in chain)
            min_y = min(letter["bbox"][1] for letter in chain)
            max_x = max(letter["bbox"][2] for letter in chain)
            max_y = max(letter["bbox"][3] for letter in chain)

            # Calculate confidence based on chain length and consistency
            avg_variance = np.mean([letter["variance_ratio"] for letter in chain])
            length_score = min(len(chain) / 10, 1.0)
            consistency_score = 1.0 - min(avg_variance, 1.0)
            confidence = (length_score * 0.5 + consistency_score * 0.5) * 0.85

            metadata = {
                "letter_count": len(chain),
                "avg_stroke_width": float(
                    np.mean([letter["mean_stroke_width"] for letter in chain])
                ),
                "avg_variance_ratio": float(avg_variance),
                "detection_method": "swt",
            }

            text_regions.append(((min_x, min_y, max_x, max_y), confidence, metadata))

        return text_regions
