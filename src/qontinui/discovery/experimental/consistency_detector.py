"""
EXPERIMENTAL: Consistency Detector - Finds unchanging elements across multiple screenshots.

WARNING: This module is experimental and unstable. APIs may change without notice.
Do not use in production code.

This detector analyzes multiple screenshots to identify UI elements that
remain consistent across them. Useful for finding static navigation elements,
headers, footers, and persistent UI components.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base_detector import MultiScreenshotDetector
from .types import BBox, MultiScreenshotDataset


class ConsistencyDetector(MultiScreenshotDetector):
    """Detects GUI elements by finding pixels that remain consistent across multiple screenshots.

    EXPERIMENTAL: This detector is experimental and analyzes multiple screenshots
    to find consistent UI elements.

    The core idea: Static UI elements (like navigation bars, buttons, headers) will have
    unchanging pixels across different screenshots, while dynamic content will vary.

    This approach is useful for:
    - Identifying persistent navigation elements
    - Finding static headers and footers
    - Detecting consistent UI chrome
    """

    def __init__(self):
        super().__init__("ConsistencyDetector")

    def detect_multi(
        self, dataset: MultiScreenshotDataset, **params
    ) -> Dict[int, List[BBox]]:
        """
        Detect elements by finding consistent regions across screenshots

        Args:
            dataset: MultiScreenshotDataset with multiple screenshots
            **params: Detection parameters
                - consistency_threshold (float): Minimum consistency score (0-1) for a pixel, default 0.9
                - edge_weight (float): Weight for edge detection in consistency, default 0.3
                - alignment_method (str): 'none', 'simple', or 'feature', default 'simple'
                - min_area (int): Minimum area for detected boxes, default 100
                - max_area (int): Maximum area for detected boxes, default None
                - min_width (int): Minimum width for detected boxes, default 5
                - min_height (int): Minimum height for detected boxes, default 5

        Returns:
            Dict mapping screenshot_id to list of detected boxes
        """
        consistency_threshold = params.get("consistency_threshold", 0.9)
        edge_weight = params.get("edge_weight", 0.3)
        alignment_method = params.get("alignment_method", "simple")
        min_area = params.get("min_area", 100)
        max_area = params.get("max_area", None)
        min_width = params.get("min_width", 5)
        min_height = params.get("min_height", 5)

        # Load all screenshots
        screenshots = []
        for screenshot_info in dataset.screenshots:
            img = cv2.imread(screenshot_info.path)
            if img is None:
                raise ValueError(f"Could not load screenshot: {screenshot_info.path}")
            screenshots.append((screenshot_info.screenshot_id, img))

        if len(screenshots) < 2:
            # Single screenshot - return empty results
            return {sid: [] for sid, _ in screenshots}

        # Align screenshots if requested
        aligned_screenshots = self._align_screenshots(screenshots, alignment_method)

        # Compute consistency map
        consistency_map = self._compute_consistency_map(
            aligned_screenshots, consistency_threshold, edge_weight
        )

        # Extract bounding boxes from consistency mask
        boxes = self._extract_boxes_from_mask(
            consistency_map,
            min_area=min_area,
            max_area=max_area,
            min_width=min_width,
            min_height=min_height,
        )

        # Return the same boxes for all screenshots
        # (consistent elements appear in all screenshots by definition)
        results = {}
        for screenshot_id, _ in screenshots:
            results[screenshot_id] = boxes[:]

        return results

    def _align_screenshots(
        self, screenshots: List[Tuple[int, np.ndarray]], method: str = "simple"
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Align screenshots to compensate for small shifts

        Args:
            screenshots: List of (screenshot_id, image) tuples
            method: Alignment method - 'none', 'simple', or 'feature'

        Returns:
            List of aligned (screenshot_id, image) tuples
        """
        if method == "none" or len(screenshots) < 2:
            return screenshots

        # Use first screenshot as reference
        reference_id, reference_img = screenshots[0]
        aligned = [(reference_id, reference_img)]

        if method == "simple":
            # Simple alignment: just ensure all images have the same size
            ref_height, ref_width = reference_img.shape[:2]

            for screenshot_id, img in screenshots[1:]:
                height, width = img.shape[:2]

                if height != ref_height or width != ref_width:
                    # Resize to match reference
                    img_aligned = cv2.resize(
                        img, (ref_width, ref_height), interpolation=cv2.INTER_LINEAR
                    )
                    aligned.append((screenshot_id, img_aligned))
                else:
                    aligned.append((screenshot_id, img))

        elif method == "feature":
            # Feature-based alignment using ORB or SIFT (more advanced)
            # For now, use simple template matching for small shifts
            ref_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
            ref_height, ref_width = reference_img.shape[:2]

            for screenshot_id, img in screenshots[1:]:
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                height, width = img.shape[:2]

                if height != ref_height or width != ref_width:
                    # Resize first
                    img = cv2.resize(
                        img, (ref_width, ref_height), interpolation=cv2.INTER_LINEAR
                    )
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Use phase correlation to find shift
                try:
                    # Convert to float for phase correlation
                    ref_float = np.float32(ref_gray)
                    img_float = np.float32(img_gray)

                    # Compute phase correlation
                    shift, _ = cv2.phaseCorrelate(ref_float, img_float)

                    # Apply shift if significant
                    shift_x, shift_y = int(shift[0]), int(shift[1])
                    if abs(shift_x) > 1 or abs(shift_y) > 1:
                        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                        img_aligned = cv2.warpAffine(img, M, (ref_width, ref_height))
                        aligned.append((screenshot_id, img_aligned))
                    else:
                        aligned.append((screenshot_id, img))
                except:
                    # If phase correlation fails, use unaligned image
                    aligned.append((screenshot_id, img))

        else:
            # Unknown method, return original
            return screenshots

        return aligned

    def _compute_consistency_map(
        self,
        screenshots: List[Tuple[int, np.ndarray]],
        consistency_threshold: float = 0.9,
        edge_weight: float = 0.3,
    ) -> np.ndarray:
        """
        Compute a consistency map showing which pixels remain consistent across screenshots

        Args:
            screenshots: List of (screenshot_id, image) tuples
            consistency_threshold: Threshold for considering a pixel consistent (0-1)
            edge_weight: Weight for edge information in consistency calculation

        Returns:
            Binary mask (uint8) where 255 = consistent pixel, 0 = inconsistent
        """
        if len(screenshots) < 2:
            # Single screenshot - all pixels are "consistent"
            _, img = screenshots[0]
            return np.ones(img.shape[:2], dtype=np.uint8) * 255

        # Get reference image dimensions
        _, reference_img = screenshots[0]
        height, width = reference_img.shape[:2]

        # Convert all screenshots to grayscale
        gray_screenshots = []
        edge_screenshots = []

        for _, img in screenshots:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_screenshots.append(gray)

            # Compute edges using Canny
            edges = cv2.Canny(gray, 50, 150)
            edge_screenshots.append(edges)

        # Compute pixel-wise standard deviation across screenshots
        # Low std = consistent pixel
        gray_stack = np.stack(gray_screenshots, axis=0)
        pixel_std = np.std(gray_stack, axis=0)

        # Normalize std to 0-1 range
        pixel_std_norm = pixel_std / 255.0

        # Compute consistency score: 1 - std (high consistency = low variance)
        pixel_consistency = 1.0 - pixel_std_norm

        # Compute edge consistency
        edge_stack = np.stack(edge_screenshots, axis=0)
        # Count how many screenshots have an edge at each pixel
        edge_count = np.sum(edge_stack > 0, axis=0) / len(screenshots)

        # Combine pixel consistency and edge consistency
        combined_consistency = (
            1.0 - edge_weight
        ) * pixel_consistency + edge_weight * edge_count

        # Threshold to create binary mask
        consistency_mask = (combined_consistency >= consistency_threshold).astype(
            np.uint8
        ) * 255

        # Clean up the mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        consistency_mask = cv2.morphologyEx(consistency_mask, cv2.MORPH_CLOSE, kernel)
        consistency_mask = cv2.morphologyEx(consistency_mask, cv2.MORPH_OPEN, kernel)

        return consistency_mask

    def _extract_boxes_from_mask(
        self,
        mask: np.ndarray,
        min_area: int = 100,
        max_area: Optional[int] = None,
        min_width: int = 5,
        min_height: int = 5,
    ) -> List[BBox]:
        """
        Extract bounding boxes from a binary consistency mask

        Args:
            mask: Binary mask (uint8) where 255 = consistent region
            min_area: Minimum area for a box
            max_area: Maximum area for a box (None = no limit)
            min_width: Minimum width for a box
            min_height: Minimum height for a box

        Returns:
            List of detected bounding boxes
        """
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size constraints
            if w < min_width or h < min_height:
                continue

            area = w * h
            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue

            # Create bounding box
            boxes.append(
                BBox(x1=x, y1=y, x2=x + w, y2=y + h, label="consistent", confidence=1.0)
            )

        return boxes

    def get_param_grid(self) -> List[Dict]:
        """Return parameter grid for hyperparameter search"""
        return [
            # Conservative detection (high consistency required)
            {
                "consistency_threshold": 0.95,
                "edge_weight": 0.2,
                "alignment_method": "simple",
                "min_area": 200,
            },
            # Balanced detection
            {
                "consistency_threshold": 0.9,
                "edge_weight": 0.3,
                "alignment_method": "simple",
                "min_area": 100,
            },
            # Aggressive detection (lower consistency threshold)
            {
                "consistency_threshold": 0.85,
                "edge_weight": 0.4,
                "alignment_method": "simple",
                "min_area": 50,
            },
            # Feature-based alignment
            {
                "consistency_threshold": 0.9,
                "edge_weight": 0.3,
                "alignment_method": "feature",
                "min_area": 100,
            },
            # Edge-focused detection
            {
                "consistency_threshold": 0.85,
                "edge_weight": 0.5,
                "alignment_method": "simple",
                "min_area": 100,
            },
        ]
