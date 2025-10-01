"""Decompose complex shapes into rectangles."""

import logging
from typing import Any, cast

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class RectangleDecomposer:
    """Decomposes complex shapes into rectangular regions."""

    def __init__(self, min_rect_size: int = 20):
        """
        Initialize decomposer.

        Args:
            min_rect_size: Minimum size for valid rectangles
        """
        self.min_rect_size = min_rect_size

    def decompose_window_frame(
        self, region: dict[str, Any], frame_thickness_estimate: int = 5
    ) -> list[dict[str, Any]]:
        """
        Decompose a window frame into 4 border rectangles.

        Args:
            region: Region that might be a window frame
            frame_thickness_estimate: Estimated frame thickness

        Returns:
            List of 4 rectangles (top, bottom, left, right borders)
        """
        x, y = region["x"], region["y"]
        x2, y2 = region["x2"], region["y2"]
        width = region["width"]
        height = region["height"]

        # Check if this could be a frame (hollow rectangle)
        if not self._is_frame_like(region):
            return [region]

        # Estimate actual frame thickness
        thickness = self._estimate_frame_thickness(region["mask"])

        if thickness < self.min_rect_size:
            return [region]

        rectangles = []

        # Top border
        rectangles.append(
            {
                "x": x,
                "y": y,
                "x2": x2,
                "y2": y + thickness - 1,
                "width": width,
                "height": thickness,
                "type": "frame_top",
            }
        )

        # Bottom border
        rectangles.append(
            {
                "x": x,
                "y": y2 - thickness + 1,
                "x2": x2,
                "y2": y2,
                "width": width,
                "height": thickness,
                "type": "frame_bottom",
            }
        )

        # Left border
        rectangles.append(
            {
                "x": x,
                "y": y + thickness,
                "x2": x + thickness - 1,
                "y2": y2 - thickness,
                "width": thickness,
                "height": height - 2 * thickness,
                "type": "frame_left",
            }
        )

        # Right border
        rectangles.append(
            {
                "x": x2 - thickness + 1,
                "y": y + thickness,
                "x2": x2,
                "y2": y2 - thickness,
                "width": thickness,
                "height": height - 2 * thickness,
                "type": "frame_right",
            }
        )

        # Copy pixel data if available
        if "pixel_data" in region:
            for rect in rectangles:
                rel_x = rect["x"] - x
                rel_y = rect["y"] - y
                rel_x2 = rect["x2"] - x
                rel_y2 = rect["y2"] - y
                rect["pixel_data"] = region["pixel_data"][
                    rel_y : rel_y2 + 1, rel_x : rel_x2 + 1
                ].copy()

        return rectangles

    def decompose_complex_shape(self, region: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Decompose complex shape into minimal rectangles.

        Args:
            region: Region with complex shape

        Returns:
            List of rectangular regions
        """
        if "mask" not in region:
            return [region]

        mask = region["mask"]

        # Try different decomposition strategies
        rectangles = self._greedy_rectangle_decomposition(mask)

        # Convert to region format
        x_offset, y_offset = region["x"], region["y"]
        result = []

        for rect in rectangles:
            rx, ry, rw, rh = rect
            result.append(
                {
                    "x": x_offset + rx,
                    "y": y_offset + ry,
                    "x2": x_offset + rx + rw - 1,
                    "y2": y_offset + ry + rh - 1,
                    "width": rw,
                    "height": rh,
                    "type": "decomposed",
                }
            )

        return result if result else [region]

    def _is_frame_like(self, region: dict[str, Any]) -> bool:
        """Check if region looks like a frame (hollow rectangle)."""
        if "mask" not in region:
            return False

        mask = region["mask"]
        h, w = mask.shape

        # Check if center is mostly empty
        center_region = mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        center_fill = np.sum(center_region > 0) / center_region.size

        # Check if edges are mostly filled
        edge_mask = mask.copy()
        edge_mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 0
        edge_fill = np.sum(edge_mask > 0) / edge_mask.size

        # Frame-like if center is empty and edges are filled
        return cast(bool, center_fill < 0.2 and edge_fill > 0.5)

    def _estimate_frame_thickness(self, mask: np.ndarray[Any, Any]) -> int:
        """Estimate the thickness of a frame."""
        h, w = mask.shape

        # Measure thickness from top edge
        top_thickness = 0
        for y in range(h // 3):
            if np.mean(mask[y, :]) > 0.5:
                top_thickness += 1
            else:
                break

        # Measure thickness from left edge
        left_thickness = 0
        for x in range(w // 3):
            if np.mean(mask[:, x]) > 0.5:
                left_thickness += 1
            else:
                break

        # Return average
        return max(top_thickness, left_thickness)

    def _greedy_rectangle_decomposition(
        self, mask: np.ndarray[Any, Any]
    ) -> list[tuple[int, int, int, int]]:
        """
        Greedy algorithm to decompose mask into rectangles.

        Args:
            mask: Binary mask

        Returns:
            List of rectangles as (x, y, width, height) tuples
        """
        rectangles = []
        remaining = mask.copy()

        while np.any(remaining):
            # Find largest rectangle
            rect = self._find_largest_rectangle(remaining)

            if rect is None:
                break

            x, y, w, h = rect

            # Check minimum size
            if w < self.min_rect_size or h < self.min_rect_size:
                break

            rectangles.append(rect)

            # Remove from remaining
            remaining[y : y + h, x : x + w] = 0

            # Stop if we have enough rectangles
            if len(rectangles) > 10:
                break

        return rectangles

    def _find_largest_rectangle(
        self, mask: np.ndarray[Any, Any]
    ) -> tuple[int, int, int, int] | None:
        """
        Find the largest rectangle in a binary mask.

        Args:
            mask: Binary mask

        Returns:
            Rectangle as (x, y, width, height) or None
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Find largest contour
        largest = max(contours, key=cv2.contourArea)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest)

        return (x, y, w, h) if w > 0 and h > 0 else None

    def merge_adjacent_rectangles(
        self, rectangles: list[dict[str, Any]], gap_tolerance: int = 5
    ) -> list[dict[str, Any]]:
        """
        Merge adjacent rectangles that are close together.

        Args:
            rectangles: List of rectangles
            gap_tolerance: Maximum gap to consider rectangles adjacent

        Returns:
            Merged list of rectangles
        """
        if not rectangles:
            return rectangles

        merged = []
        used = set()

        for i, rect1 in enumerate(rectangles):
            if i in used:
                continue

            # Check for adjacent rectangles
            group = [rect1]
            used.add(i)

            for j, rect2 in enumerate(rectangles[i + 1 :], i + 1):
                if j in used:
                    continue

                if self._are_adjacent(rect1, rect2, gap_tolerance):
                    group.append(rect2)
                    used.add(j)

            # Merge group
            if len(group) == 1:
                merged.append(group[0])
            else:
                merged.append(self._merge_rectangle_group(group))

        return merged

    def _are_adjacent(
        self, rect1: dict[str, Any], rect2: dict[str, Any], gap_tolerance: int
    ) -> bool:
        """Check if two rectangles are adjacent."""
        # Check horizontal adjacency
        if (
            abs(rect1["x2"] - rect2["x"]) <= gap_tolerance
            or abs(rect2["x2"] - rect1["x"]) <= gap_tolerance
        ):
            # Check vertical overlap
            if not (rect1["y2"] < rect2["y"] or rect2["y2"] < rect1["y"]):
                return True

        # Check vertical adjacency
        if (
            abs(rect1["y2"] - rect2["y"]) <= gap_tolerance
            or abs(rect2["y2"] - rect1["y"]) <= gap_tolerance
        ):
            # Check horizontal overlap
            if not (rect1["x2"] < rect2["x"] or rect2["x2"] < rect1["x"]):
                return True

        return False

    def _merge_rectangle_group(self, group: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge a group of rectangles."""
        x_min = min(r["x"] for r in group)
        y_min = min(r["y"] for r in group)
        x_max = max(r["x2"] for r in group)
        y_max = max(r["y2"] for r in group)

        return {
            "x": x_min,
            "y": y_min,
            "x2": x_max,
            "y2": y_max,
            "width": x_max - x_min + 1,
            "height": y_max - y_min + 1,
            "type": "merged",
        }
