"""Region annotator — draws semi-transparent filled overlays for zone visualisation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from .base import Annotator, Color

if TYPE_CHECKING:
    from ..detections import Detections


class RegionAnnotator(Annotator):
    """Draw semi-transparent filled rectangles over detection regions.

    Useful for highlighting zones, regions of interest, or heat-map-style
    overlays where the filled area communicates importance.

    Args:
        color: BGR fill colour.
        opacity: Fill opacity (0.0 fully transparent – 1.0 fully opaque).
        border_thickness: Optional border line thickness (0 = no border).
        border_color: BGR border colour (defaults to same as *color*).
    """

    def __init__(
        self,
        color: Color = (255, 200, 0),
        opacity: float = 0.25,
        border_thickness: int = 1,
        border_color: Color | None = None,
    ) -> None:
        self.color = color
        self.opacity = max(0.0, min(1.0, opacity))
        self.border_thickness = border_thickness
        self.border_color = border_color if border_color is not None else color

    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        overlay = scene.copy()

        for i in range(len(detections)):
            x1, y1, x2, y2 = (int(v) for v in detections.xyxy[i])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), self.color, -1)

        # Blend the filled overlay with the original scene
        cv2.addWeighted(overlay, self.opacity, scene, 1.0 - self.opacity, 0, dst=scene)

        # Draw borders on top (not blended)
        if self.border_thickness > 0:
            for i in range(len(detections)):
                x1, y1, x2, y2 = (int(v) for v in detections.xyxy[i])
                cv2.rectangle(
                    scene, (x1, y1), (x2, y2), self.border_color, self.border_thickness
                )

        return scene
