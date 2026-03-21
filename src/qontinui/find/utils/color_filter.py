"""Color-difference filter for post-filtering template matches.

Compares mean RGB of matched regions against a reference color to
distinguish visually similar elements with different color states
(e.g., enabled=blue vs disabled=gray buttons).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ColorDifferenceFilter:
    """Post-filter that rejects matches whose mean color diverges from a reference.

    Attributes:
        reference_color: Expected RGB color of the target element.
        tolerance: Maximum Euclidean distance in RGB space (0-441).
    """

    reference_color: tuple[int, int, int]
    tolerance: float = 50.0

    def mean_rgb(self, region_bgr: np.ndarray[Any, Any]) -> tuple[float, float, float]:
        """Compute mean RGB of a BGR image region.

        Args:
            region_bgr: Image region in BGR format (OpenCV convention).

        Returns:
            Mean (R, G, B) as floats.
        """
        if region_bgr.size == 0:
            return (0.0, 0.0, 0.0)
        mean_bgr = region_bgr.mean(axis=(0, 1))
        # BGR -> RGB
        return (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0]))

    def color_distance(self, color: tuple[float, float, float]) -> float:
        """Euclidean distance between *color* and the reference.

        Args:
            color: (R, G, B) values.

        Returns:
            Distance in RGB space.
        """
        r, g, b = color
        rr, rg, rb = self.reference_color
        return math.sqrt((r - rr) ** 2 + (g - rg) ** 2 + (b - rb) ** 2)

    def passes(self, region_bgr: np.ndarray[Any, Any]) -> bool:
        """Check if the region's mean color is within tolerance.

        Args:
            region_bgr: Image region in BGR format.

        Returns:
            True if the color distance is within tolerance.
        """
        return self.color_distance(self.mean_rgb(region_bgr)) <= self.tolerance

    def filter_results(
        self,
        results: list[Any],
        haystack_bgr: np.ndarray[Any, Any],
    ) -> list[Any]:
        """Remove results whose matched region color exceeds tolerance.

        Each result must have x, y, width, height attributes (e.g.
        ``DetectionResult`` or HAL ``Match``).

        Args:
            results: List of detection results with bounding-box attributes.
            haystack_bgr: Full screenshot in BGR format.

        Returns:
            Filtered list (order preserved).
        """
        kept: list[Any] = []
        for r in results:
            x, y, w, h = r.x, r.y, r.width, r.height
            region = haystack_bgr[y : y + h, x : x + w]
            if region.size > 0 and self.passes(region):
                kept.append(r)
        return kept
