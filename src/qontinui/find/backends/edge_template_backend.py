"""Edge-aware template matching detection backend.

Converts both needle and haystack to Canny edge maps before running
cv2.matchTemplate. This makes matching invariant to color/theme changes
(dark/light mode, hover states, disabled opacity) while preserving
structural shape.

Slots into the cascade between raw template matching (~20ms) and
feature matching (~100ms) at ~40ms estimated cost.
"""

import logging
from typing import Any

import cv2
import numpy as np

from .base import DetectionBackend, DetectionResult

logger = logging.getLogger(__name__)

# Default Canny thresholds (matching EdgeDetector defaults)
_DEFAULT_CANNY_LOW = 50
_DEFAULT_CANNY_HIGH = 150


class EdgeTemplateMatchBackend(DetectionBackend):
    """Detection backend using edge-map template matching.

    Extracts Canny edges from both needle and haystack, then runs
    ``cv2.matchTemplate`` on the edge maps. More robust to theme
    and color changes than pixel-based template matching.

    Args:
        canny_low: Canny edge detection low threshold.
        canny_high: Canny edge detection high threshold.
        min_confidence: Default minimum confidence threshold.
    """

    def __init__(
        self,
        canny_low: int = _DEFAULT_CANNY_LOW,
        canny_high: int = _DEFAULT_CANNY_HIGH,
        min_confidence: float = 0.7,
    ) -> None:
        self._canny_low = canny_low
        self._canny_high = canny_high
        self._min_confidence = min_confidence

    def _to_edges(self, image: np.ndarray) -> np.ndarray:
        """Convert image to Canny edge map.

        Args:
            image: BGR or grayscale numpy array.

        Returns:
            Single-channel edge map (uint8).
        """
        if len(image.shape) == 3 and image.shape[2] >= 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, self._canny_low, self._canny_high)

    def _extract_image(self, obj: Any) -> np.ndarray | None:
        """Extract a numpy array from a Pattern, PIL Image, or raw ndarray."""
        if isinstance(obj, np.ndarray):
            return obj

        # Handle Pattern objects (pixel_data attribute)
        if hasattr(obj, "pixel_data") and obj.pixel_data is not None:
            data = obj.pixel_data
            if isinstance(data, np.ndarray):
                return data

        # Handle PIL Images
        try:
            from PIL import Image as PILImage

            if isinstance(obj, PILImage.Image):
                arr = np.array(obj)
                if len(arr.shape) == 3 and arr.shape[2] >= 3:
                    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                return arr
        except ImportError:
            pass

        return None

    def find(self, needle: Any, haystack: Any, config: dict[str, Any]) -> list[DetectionResult]:
        """Find needle in haystack using edge-map template matching.

        Args:
            needle: Pattern object, numpy array, or PIL Image of the template.
            haystack: Screenshot as numpy array or PIL Image.
            config: Keys used: ``min_confidence``, ``search_region``.

        Returns:
            List of DetectionResult sorted by confidence.
        """
        needle_img = self._extract_image(needle)
        haystack_img = self._extract_image(haystack)

        if needle_img is None or haystack_img is None:
            return []

        min_confidence = config.get("min_confidence", self._min_confidence)
        search_region = config.get("search_region")

        # Crop to search region if specified
        offset_x, offset_y = 0, 0
        if search_region is not None:
            rx, ry, rw, rh = search_region
            haystack_img = haystack_img[ry : ry + rh, rx : rx + rw]
            offset_x, offset_y = rx, ry

        # Convert to edge maps
        needle_edges = self._to_edges(needle_img)
        haystack_edges = self._to_edges(haystack_img)

        th, tw = needle_edges.shape[:2]

        # Template must fit in search area
        if th > haystack_edges.shape[0] or tw > haystack_edges.shape[1]:
            return []

        # Minimum edge density check — skip if template has no meaningful edges
        edge_density = np.count_nonzero(needle_edges) / max(needle_edges.size, 1)
        if edge_density < 0.01:
            logger.debug(
                "EdgeTemplateMatchBackend: template has too few edges (%.3f), skipping",
                edge_density,
            )
            return []

        # Run template matching on edge maps
        result = cv2.matchTemplate(haystack_edges, needle_edges, cv2.TM_CCOEFF_NORMED)

        # Find locations above threshold
        locations = np.where(result >= min_confidence)
        if len(locations[0]) == 0:
            return []

        # Apply non-maximum suppression
        return self._nms(result, locations, tw, th, min_confidence, offset_x, offset_y)

    def _nms(
        self,
        result: np.ndarray,
        locations: tuple[np.ndarray, np.ndarray],
        width: int,
        height: int,
        threshold: float,
        offset_x: int,
        offset_y: int,
    ) -> list[DetectionResult]:
        """Apply non-maximum suppression and return DetectionResults."""
        candidates = [
            (locations[0][i], locations[1][i], float(result[locations[0][i], locations[1][i]]))
            for i in range(len(locations[0]))
        ]
        candidates.sort(key=lambda c: c[2], reverse=True)

        results: list[DetectionResult] = []
        used: list[tuple[int, int]] = []  # center points

        for y, x, confidence in candidates:
            cx, cy = x + width // 2, y + height // 2

            # Check overlap with existing matches
            is_dup = False
            for ux, uy in used:
                if abs(cx - ux) < width // 2 and abs(cy - uy) < height // 2:
                    is_dup = True
                    break

            if not is_dup:
                results.append(
                    DetectionResult(
                        x=int(x) + offset_x,
                        y=int(y) + offset_y,
                        width=width,
                        height=height,
                        confidence=confidence,
                        backend_name=self.name,
                        metadata={"matching_mode": "edge"},
                    )
                )
                used.append((cx, cy))

        return results

    def supports(self, needle_type: str) -> bool:
        return needle_type == "template"

    def estimated_cost_ms(self) -> float:
        return 40.0

    @property
    def name(self) -> str:
        return "edge_template"
