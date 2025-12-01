"""Intelligent region detection for State Discovery."""

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class IntelligentRegionDetector:
    """Intelligent detection of UI regions using perceptual grouping and heuristics."""

    def __init__(self, min_region_size=(20, 20), max_region_size=(500, 500)) -> None:
        self.min_region_size = min_region_size
        self.max_region_size = max_region_size

    def merge_nearby_components(
        self,
        stability_map: np.ndarray[Any, Any],
        max_gap: int = 8,
        min_pixels: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Merge nearby stable components to form meaningful regions.

        This connects:
        - Individual letters into words
        - Icon fragments into complete icons
        - Border segments into frames

        Args:
            stability_map: Binary map where 1 = stable pixel
            max_gap: Maximum pixel gap to bridge between components
            min_pixels: Minimum pixels for a valid region

        Returns:
            List of merged region masks
        """
        # Step 1: Dilate to connect nearby pixels
        kernel = np.ones((max_gap, max_gap), np.uint8)
        dilated = cv2.dilate(stability_map.astype(np.uint8), kernel, iterations=1)

        # Step 2: Find connected components on dilated map
        num_labels, labels = cv2.connectedComponents(dilated)

        logger.info(f"Found {num_labels - 1} merged components (from dilation)")

        # Step 3: Extract regions using original pixels (not dilated)
        merged_regions = []
        for label_id in range(1, num_labels):
            # Get original stable pixels for this component
            original_mask = (labels == label_id) & (stability_map > 0)

            # Check if region meets size requirements
            if np.sum(original_mask) >= min_pixels:
                # Get bounding box
                coords = np.column_stack(np.where(original_mask))
                if len(coords) > 0:
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)

                    width = x_max - x_min + 1
                    height = y_max - y_min + 1

                    # Check size constraints
                    if (
                        width >= self.min_region_size[0]
                        and height >= self.min_region_size[1]
                        and width <= self.max_region_size[0]
                        and height <= self.max_region_size[1]
                    ):

                        merged_regions.append(
                            {
                                "mask": original_mask[
                                    y_min : y_max + 1, x_min : x_max + 1
                                ],
                                "bbox": (x_min, y_min, x_max, y_max),
                                "pixel_count": np.sum(original_mask),
                            }
                        )

        logger.info(f"After filtering: {len(merged_regions)} valid merged regions")
        return merged_regions

    def pairwise_stability_analysis(
        self,
        screenshots: list[np.ndarray[Any, Any]],
        similarity_threshold: float = 0.95,
    ) -> dict[str, dict[str, Any]]:
        """
        Compare each pair of screenshots to find stable regions.

        Args:
            screenshots: List of screenshot arrays
            similarity_threshold: Threshold for considering regions similar

        Returns:
            Dictionary of regions with their exact screenshot presence
        """
        n = len(screenshots)
        pairwise_regions: dict[str, dict[str, Any]] = {}

        # Compare each pair of screenshots
        for i in range(n):
            for j in range(i + 1, n):
                logger.debug(f"Comparing screenshots {i} and {j}")

                # Find stable regions between this pair
                stable_regions = self._find_stable_between_pair(
                    screenshots[i], screenshots[j], similarity_threshold
                )

                for region in stable_regions:
                    region_hash = self._hash_region(region["pixels"])

                    if region_hash not in pairwise_regions:
                        pairwise_regions[region_hash] = {
                            "region": region,
                            "screenshot_pairs": set(),
                            "exact_presence": set(),
                        }

                    pairwise_regions[region_hash]["screenshot_pairs"].add((i, j))

        # Now check each region against ALL screenshots for exact presence
        for _region_hash, region_data in pairwise_regions.items():
            region_dict: dict[str, Any] = region_data["region"]  # type: ignore[assignment]

            for idx, screenshot in enumerate(screenshots):
                if self._is_region_present(
                    region_dict, screenshot, similarity_threshold
                ):
                    region_data["exact_presence"].add(idx)  # type: ignore[union-attr]

            # Convert set to sorted list for consistency
            region_data["exact_presence"] = sorted(region_data["exact_presence"])  # type: ignore[arg-type]

        logger.info(
            f"Found {len(pairwise_regions)} unique regions from pairwise analysis"
        )
        return pairwise_regions

    def detect_ui_elements(
        self, screenshot: np.ndarray[Any, Any], stability_map: np.ndarray[Any, Any]
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Detect specific UI elements: icons, buttons, windows.

        Args:
            screenshot: Screenshot image
            stability_map: Binary stability map

        Returns:
            Dictionary with detected elements by type
        """
        elements = {
            "icons": self._detect_icons(screenshot, stability_map),
            "buttons": self._detect_buttons(screenshot, stability_map),
            "windows": self._detect_windows(screenshot, stability_map),
        }

        total = sum(len(v) for v in elements.values())
        logger.info(
            f"Detected {total} UI elements: {len(elements['icons'])} icons, "
            f"{len(elements['buttons'])} buttons, {len(elements['windows'])} windows"
        )

        return elements

    def _detect_icons(
        self, screenshot: np.ndarray[Any, Any], stability_map: np.ndarray[Any, Any]
    ) -> list[dict[str, Any]]:
        """Detect icon-like square regions."""
        edges = cv2.Canny(screenshot, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        icons = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            # Icon-like dimensions (square-ish, reasonable size)
            if 0.8 < aspect_ratio < 1.2 and 16 <= w <= 128 and 16 <= h <= 128:
                # Check stability in region
                region_stability = stability_map[y : y + h, x : x + w]

                # At least 30% stable pixels (allows for dynamic background)
                if region_stability.size > 0 and np.mean(region_stability) > 0.3:
                    icons.append(
                        {
                            "bbox": (x, y, w, h),
                            "type": "icon",
                            "stability": np.mean(region_stability),
                        }
                    )

        return icons

    def _detect_buttons(
        self, screenshot: np.ndarray[Any, Any], stability_map: np.ndarray[Any, Any]
    ) -> list[dict[str, Any]]:
        """Detect button-like rectangular regions."""
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        # Use adaptive threshold to find rectangular regions
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        buttons = []
        for contour in contours:
            # Approximate to polygon
            approx = cv2.approxPolyDP(
                contour, 0.02 * cv2.arcLength(contour, True), True
            )

            # Rectangles have 4 vertices
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)

                # Button-like dimensions
                if 50 <= w <= 400 and 20 <= h <= 80:
                    # Check if mostly stable
                    region_stability = stability_map[y : y + h, x : x + w]

                    if region_stability.size > 0 and np.mean(region_stability) > 0.5:
                        buttons.append(
                            {
                                "bbox": (x, y, w, h),
                                "type": "button",
                                "stability": np.mean(region_stability),
                            }
                        )

        return buttons

    def _detect_windows(
        self, screenshot: np.ndarray[Any, Any], stability_map: np.ndarray[Any, Any]
    ) -> list[dict[str, Any]]:
        """Detect window frames and panels."""
        edges = cv2.Canny(screenshot, 30, 100)

        # Detect horizontal lines (title bars, borders)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)

        # Detect vertical lines (side borders)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)

        # Combine to find rectangular window shapes
        window_structure = cv2.bitwise_or(horizontal_lines, vertical_lines)

        contours, _ = cv2.findContours(
            window_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        windows = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Window-like dimensions
            if w > 200 and h > 100:
                windows.append(
                    {
                        "bbox": (x, y, w, h),
                        "type": "window",
                        "title_bar": (x, y, w, min(30, h)),  # Top portion is title
                    }
                )

        return windows

    def _find_stable_between_pair(
        self, img1: np.ndarray[Any, Any], img2: np.ndarray[Any, Any], threshold: float
    ) -> list[dict[str, Any]]:
        """Find stable regions between two screenshots."""
        # Simple difference-based stability
        diff = cv2.absdiff(img1, img2)
        gray_diff = (
            cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if len(diff.shape) == 3 else diff
        )

        # Regions with low difference are stable
        stable_mask = gray_diff < (255 * (1 - threshold))

        # Find connected components in stable regions
        num_labels, labels = cv2.connectedComponents(stable_mask.astype(np.uint8))

        regions = []
        for label_id in range(1, num_labels):
            mask = labels == label_id

            if np.sum(mask) >= 50:  # Minimum size
                coords = np.column_stack(np.where(mask))
                if len(coords) > 0:
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)

                    regions.append(
                        {
                            "bbox": (x_min, y_min, x_max, y_max),
                            "pixels": img1[y_min : y_max + 1, x_min : x_max + 1].copy(),
                            "mask": mask[y_min : y_max + 1, x_min : x_max + 1],
                        }
                    )

        return regions

    def _is_region_present(
        self, region: dict[str, Any], screenshot: np.ndarray[Any, Any], threshold: float
    ) -> bool:
        """Check if a region is present in a screenshot."""
        x_min, y_min, x_max, y_max = region["bbox"]

        # Extract corresponding region from screenshot
        roi = screenshot[y_min : y_max + 1, x_min : x_max + 1]

        if roi.shape != region["pixels"].shape:
            return False

        # Calculate similarity
        diff = cv2.absdiff(roi, region["pixels"])
        mean_diff = np.mean(diff)

        similarity = 1.0 - (mean_diff / 255.0)
        return bool(similarity >= threshold)

    def _hash_region(self, pixels: np.ndarray[Any, Any]) -> str:
        """Create a hash for a region for deduplication."""
        import hashlib

        return hashlib.sha256(pixels.tobytes()).hexdigest()

    def hierarchical_detection(
        self, screenshot: np.ndarray[Any, Any], stability_map: np.ndarray[Any, Any]
    ) -> list[dict[str, Any]]:
        """
        Detect regions hierarchically to avoid processing overlapping regions.

        Process order:
        1. Large windows/panels
        2. Medium buttons/toolbars
        3. Small icons/text
        """
        all_regions = []
        processed_mask = np.zeros_like(stability_map, dtype=bool)

        # Level 1: Large elements
        windows = self._detect_windows(screenshot, stability_map)
        for window in windows:
            x, y, w, h = window["bbox"]
            processed_mask[y : y + h, x : x + w] = True
            all_regions.append({**window, "level": 1})

        # Level 2: Medium elements (skip processed areas)
        remaining_stability = stability_map & ~processed_mask
        buttons = self._detect_buttons(screenshot, remaining_stability)
        for button in buttons:
            x, y, w, h = button["bbox"]
            processed_mask[y : y + h, x : x + w] = True
            all_regions.append({**button, "level": 2})

        # Level 3: Small elements
        remaining_stability = stability_map & ~processed_mask
        icons = self._detect_icons(screenshot, remaining_stability)
        for icon in icons:
            all_regions.append({**icon, "level": 3})

        logger.info(
            f"Hierarchical detection found {len(all_regions)} regions "
            f"(Level 1: {len(windows)}, Level 2: {len(buttons)}, Level 3: {len(icons)})"
        )

        return all_regions
