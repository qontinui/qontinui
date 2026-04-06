"""Label annotator — draws text labels above each detection."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .base import Annotator, Color

if TYPE_CHECKING:
    from ..detections import Detections


def _default_label(detections: Detections, index: int) -> str:
    """Default label: ``backend_name confidence%``."""
    name = str(detections.backend_name[index])
    conf = detections.confidence[index]
    return f"{name} {conf:.0%}"


class LabelAnnotator(Annotator):
    """Draw text labels with background for each detection.

    By default the label shows ``backend_name confidence%``.  Supply a
    custom ``label_fn(detections, index) -> str`` to override.

    Args:
        color_text: BGR text colour.
        color_background: BGR label background colour.
        label_fn: Callable ``(Detections, int) -> str`` producing per-detection text.
        font_scale: OpenCV font scale.
        thickness: Font line thickness.
        padding: Pixel padding around text.
    """

    def __init__(
        self,
        color_text: Color = (255, 255, 255),
        color_background: Color = (0, 0, 0),
        label_fn: Callable[[Detections, int], str] | None = None,
        font_scale: float = 0.5,
        thickness: int = 1,
        padding: int = 5,
    ) -> None:
        self.color_text = color_text
        self.color_background = color_background
        self.label_fn = label_fn or _default_label
        self.font_scale = font_scale
        self.thickness = thickness
        self.padding = padding
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        for i in range(len(detections)):
            x1, y1 = int(detections.xyxy[i, 0]), int(detections.xyxy[i, 1])
            text = self.label_fn(detections, i)

            (tw, th), _ = cv2.getTextSize(text, self._font, self.font_scale, self.thickness)

            # Position label above the box
            label_y = y1 - 5
            bg_top = label_y - th - self.padding
            bg_bottom = label_y
            bg_left = x1
            bg_right = x1 + tw + self.padding

            cv2.rectangle(
                scene,
                (bg_left, bg_top),
                (bg_right, bg_bottom),
                self.color_background,
                -1,
            )
            cv2.putText(
                scene,
                text,
                (x1 + 2, label_y - 2),
                self._font,
                self.font_scale,
                self.color_text,
                self.thickness,
                cv2.LINE_AA,
            )
        return scene
