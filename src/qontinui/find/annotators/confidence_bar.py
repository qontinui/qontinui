"""Confidence bar annotator — draws a horizontal bar inside each detection box."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from .base import Annotator, Color

if TYPE_CHECKING:
    from ..detections import Detections


class ConfidenceBarAnnotator(Annotator):
    """Draw a confidence bar at the bottom of each detection bounding box.

    The bar fills from left to right proportional to the confidence score.
    Colour interpolates from ``color_low`` to ``color_high``.

    Args:
        color_low: BGR colour at 0 % confidence.
        color_high: BGR colour at 100 % confidence.
        bar_height: Height of the bar in pixels.
        opacity: Blend opacity for the bar overlay (0.0–1.0).
    """

    def __init__(
        self,
        color_low: Color = (0, 0, 255),
        color_high: Color = (0, 255, 0),
        bar_height: int = 6,
        opacity: float = 0.7,
    ) -> None:
        self.color_low = color_low
        self.color_high = color_high
        self.bar_height = bar_height
        self.opacity = max(0.0, min(1.0, opacity))

    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        h_img, w_img = scene.shape[:2]

        for i in range(len(detections)):
            x1, y1, x2, y2 = (int(v) for v in detections.xyxy[i])
            conf = float(detections.confidence[i])

            # Bar sits at the bottom of the bounding box
            bar_top = max(0, y2 - self.bar_height)
            bar_bottom = min(y2, h_img)
            bar_left = max(0, x1)
            bar_right_full = min(x2, w_img)
            bar_width = bar_right_full - bar_left
            if bar_width <= 0 or bar_bottom <= bar_top:
                continue

            fill_right = bar_left + int(bar_width * conf)

            # Interpolate colour
            color = tuple(
                int(lo + (hi - lo) * conf)
                for lo, hi in zip(self.color_low, self.color_high, strict=True)
            )

            # Draw with alpha blending
            original = scene[bar_top:bar_bottom, bar_left:fill_right].copy()
            cv2.rectangle(
                scene,
                (bar_left, bar_top),
                (fill_right, bar_bottom),
                color,
                -1,
            )
            blended = cv2.addWeighted(
                scene[bar_top:bar_bottom, bar_left:fill_right],
                self.opacity,
                original,
                1.0 - self.opacity,
                0,
            )
            scene[bar_top:bar_bottom, bar_left:fill_right] = blended
        return scene
