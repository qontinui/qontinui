"""Extract stable regions from pixel stability maps."""

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StableRegionExtractor:
    """Extracts and refines stable regions from stability maps."""

    def __init__(
        self,
        min_area: int = 400,  # 20x20
        max_area: int = 250000,  # 500x500
        min_stability: float = 0.95,
    ) -> None:
        """
        Initialize extractor.

        Args:
            min_area: Minimum area for valid regions
            max_area: Maximum area for valid regions
            min_stability: Minimum stability score
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_stability = min_stability

    def extract_regions(
        self,
        stability_map: np.ndarray[Any, Any],
        reference_image: np.ndarray[Any, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Extract stable regions from stability map.

        Args:
            stability_map: Binary or float stability map
            reference_image: Optional reference image for pixel data

        Returns:
            List of region dictionaries
        """
        # Convert to binary if needed
        if stability_map.dtype != np.uint8:
            binary_map = (stability_map > self.min_stability).astype(np.uint8)
        else:
            binary_map = stability_map

        # Apply morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Find connected components
        regions = self._find_connected_components(cleaned, reference_image)

        # Filter regions
        regions = self._filter_regions(regions)

        # Merge overlapping regions
        regions = self._merge_overlapping(regions)

        return regions

    def _find_connected_components(
        self,
        binary_map: np.ndarray[Any, Any],
        reference_image: np.ndarray[Any, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Find connected components in binary map."""
        regions = []

        # Find contours
        contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate area
            area = cv2.contourArea(contour)

            # Skip if outside size bounds
            if area < self.min_area or area > self.max_area:
                continue

            # Create region mask
            mask = np.zeros(binary_map.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)  # type: ignore[call-overload]

            region = {
                "x": x,
                "y": y,
                "x2": x + w - 1,
                "y2": y + h - 1,
                "width": w,
                "height": h,
                "area": area,
                "contour": contour,
                "mask": mask[y : y + h, x : x + w],
            }

            # Extract pixel data if reference image provided
            if reference_image is not None:
                region["pixel_data"] = reference_image[y : y + h, x : x + w].copy()

            regions.append(region)

        return regions

    def _filter_regions(self, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filter regions based on various criteria."""
        filtered = []

        for region in regions:
            # Check aspect ratio
            aspect_ratio = region["width"] / region["height"]
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue

            # Check fill ratio (actual pixels vs bounding box)
            fill_ratio = region["area"] / (region["width"] * region["height"])
            if fill_ratio < 0.3:  # Less than 30% filled
                continue

            filtered.append(region)

        return filtered

    def _merge_overlapping(
        self, regions: list[dict[str, Any]], overlap_threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        """Merge significantly overlapping regions."""
        if not regions:
            return regions

        merged = []
        used = set()

        for i, region1 in enumerate(regions):
            if i in used:
                continue

            # Start a group with this region
            group = [region1]
            used.add(i)

            for j, region2 in enumerate(regions[i + 1 :], i + 1):
                if j in used:
                    continue

                # Check overlap
                overlap = self._calculate_overlap(region1, region2)
                if overlap > overlap_threshold:
                    group.append(region2)
                    used.add(j)

            # Merge group into single region
            if len(group) == 1:
                merged.append(group[0])
            else:
                merged.append(self._merge_group(group))

        return merged

    def _calculate_overlap(self, region1: dict[str, Any], region2: dict[str, Any]) -> float:
        """Calculate overlap ratio between two regions."""
        # Calculate intersection
        x1 = max(region1["x"], region2["x"])
        y1 = max(region1["y"], region2["y"])
        x2 = min(region1["x2"], region2["x2"])
        y2 = min(region1["y2"], region2["y2"])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Calculate union
        area1 = region1["width"] * region1["height"]
        area2 = region2["width"] * region2["height"]
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _merge_group(self, group: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge a group of regions into one."""
        # Find bounding box
        x_min = min(r["x"] for r in group)
        y_min = min(r["y"] for r in group)
        x_max = max(r["x2"] for r in group)
        y_max = max(r["y2"] for r in group)

        # Combine masks
        height = y_max - y_min + 1
        width = x_max - x_min + 1
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        for region in group:
            rel_x = region["x"] - x_min
            rel_y = region["y"] - y_min
            h, w = region["mask"].shape
            combined_mask[rel_y : rel_y + h, rel_x : rel_x + w] = np.maximum(
                combined_mask[rel_y : rel_y + h, rel_x : rel_x + w], region["mask"]
            )

        return {
            "x": x_min,
            "y": y_min,
            "x2": x_max,
            "y2": y_max,
            "width": width,
            "height": height,
            "area": np.sum(combined_mask > 0),
            "mask": combined_mask,
        }

    def find_stable_text_regions(
        self, stability_map: np.ndarray[Any, Any], reference_image: np.ndarray[Any, Any]
    ) -> list[dict[str, Any]]:
        """
        Find regions likely to contain stable text.

        Args:
            stability_map: Stability map
            reference_image: Reference image

        Returns:
            List of text region candidates
        """
        # Convert to grayscale if needed
        if len(reference_image.shape) == 3:
            gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = reference_image

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Combine with stability map
        text_candidates = cv2.bitwise_and(thresh, thresh, mask=stability_map)

        # Find text regions
        regions = self._find_connected_components(text_candidates, reference_image)

        # Filter for text-like properties
        text_regions = []
        for region in regions:
            # Check aspect ratio for text
            if 0.1 < region["width"] / region["height"] < 20:
                text_regions.append(region)

        return text_regions
