"""Click Analysis module for inferring bounding boxes from click locations.

This module provides sophisticated analysis of click interactions to infer
accurate bounding boxes for clicked GUI elements. Unlike simple fixed-size
boxes, this module uses multiple detection strategies to find the actual
element boundaries.

Key Features:
    - Multi-strategy element boundary detection
    - Support for various element types (buttons, icons, text, images)
    - Mask support for non-rectangular elements
    - Integration with existing StateImage infrastructure
    - Context-aware element type classification

Example:
    >>> from qontinui.discovery.click_analysis import ClickBoundingBoxInferrer
    >>>
    >>> inferrer = ClickBoundingBoxInferrer()
    >>> result = inferrer.infer_bounding_box(
    ...     screenshot=screenshot_array,
    ...     click_location=(350, 250)
    ... )
    >>> print(f"Found element: {result.bbox} with type {result.element_type}")

Classes:
    ClickBoundingBoxInferrer: Main class for inferring bounding boxes
    ElementBoundaryFinder: Finds element boundaries using multiple strategies
    ClickContextAnalyzer: Determines the type of element clicked
    InferredBoundingBox: Result dataclass with bbox, mask, and metadata
"""

from .models import (
    InferredBoundingBox,
    InferenceConfig,
    InferenceResult,
    ElementType,
    DetectionStrategy,
)
from .inferrer import ClickBoundingBoxInferrer, infer_bbox_from_click
from .boundary_finder import ElementBoundaryFinder
from .context_analyzer import ClickContextAnalyzer

__all__ = [
    # Main classes
    "ClickBoundingBoxInferrer",
    "ElementBoundaryFinder",
    "ClickContextAnalyzer",
    # Data models
    "InferredBoundingBox",
    "InferenceConfig",
    "InferenceResult",
    "ElementType",
    "DetectionStrategy",
    # Convenience function
    "infer_bbox_from_click",
]
