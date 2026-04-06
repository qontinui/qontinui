"""Bounding box annotator — draws rectangles around detections."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from .base import Annotator, Color

if TYPE_CHECKING:
    from ..detections import Detections


class BoundingBoxAnnotator(Annotator):
    """Draw bounding box rectangles for each detection.

    Boxes are coloured per-detection based on a confidence threshold:
    detections at or above ``confidence_threshold`` get ``color_match``,
    those below get ``color_non_match``.  Set ``confidence_threshold=0``
    to colour everything with ``color_match``.

    Args:
        color_match: BGR colour for detections above threshold.
        color_non_match: BGR colour for detections below threshold.
        confidence_threshold: Confidence dividing match/non-match colours.
        thickness: Line thickness in pixels.
    """

    def __init__(
        self,
        color_match: Color = (0, 255, 0),
        color_non_match: Color = (0, 0, 255),
        confidence_threshold: float = 0.0,
        thickness: int = 2,
    ) -> None:
        self.color_match = color_match
        self.color_non_match = color_non_match
        self.confidence_threshold = confidence_threshold
        self.thickness = thickness

    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        for i in range(len(detections)):
            x1, y1, x2, y2 = detections.xyxy[i]
            color = (
                self.color_match
                if detections.confidence[i] >= self.confidence_threshold
                else self.color_non_match
            )
            cv2.rectangle(
                scene,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                self.thickness,
            )
        return scene
