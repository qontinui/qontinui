"""Base class for composable detection annotators.

All annotators follow the same interface: they take a scene (image) and
a Detections container, draw onto the scene, and return it.  This enables
chaining::

    scene = box_ann.annotate(scene, dets)
    scene = label_ann.annotate(scene, dets)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..detections import Detections

# BGR colour type used by OpenCV
Color = tuple[int, int, int]


class Annotator(ABC):
    """Abstract base for detection annotators.

    Subclasses implement ``annotate`` to draw overlays on a BGR image
    given a ``Detections`` container.
    """

    @abstractmethod
    def annotate(self, scene: np.ndarray, detections: Detections) -> np.ndarray:
        """Draw annotations onto *scene* and return the modified image.

        Implementations should draw **in-place** on *scene* (or a copy)
        and return the same array so callers can chain calls.

        Args:
            scene: BGR image (H, W, 3) uint8 numpy array.
            detections: Detections container with bounding boxes, confidence, etc.

        Returns:
            The annotated image (same object as *scene* unless a copy was made).
        """
        ...
