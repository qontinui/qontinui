"""Composable annotators for detection visualisation.

Each annotator follows the ``annotate(scene, detections) -> scene`` pattern
and can be chained to build complex debug visualisations::

    scene = box_annotator.annotate(scene, detections)
    scene = label_annotator.annotate(scene, detections)
    scene = bar_annotator.annotate(scene, detections)
"""

from .base import Annotator, Color
from .bounding_box import BoundingBoxAnnotator
from .confidence_bar import ConfidenceBarAnnotator
from .label import LabelAnnotator
from .region import RegionAnnotator

__all__ = [
    "Annotator",
    "BoundingBoxAnnotator",
    "Color",
    "ConfidenceBarAnnotator",
    "LabelAnnotator",
    "RegionAnnotator",
]
