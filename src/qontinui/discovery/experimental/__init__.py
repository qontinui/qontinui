"""Experimental detection features and research implementations.

This module contains experimental and research-level implementations of detection
algorithms, new approaches to state discovery, and cutting-edge techniques that
are being evaluated for inclusion in the main detection pipeline.

Warning:
    Code in this module is experimental and may change or be removed without notice.
    APIs are not stable and should not be relied upon in production code.

Key Areas:
    - Advanced machine learning models (SAM, YOLO, transformers)
    - Novel detection algorithms
    - Performance optimization experiments
    - Alternative state representation approaches
    - Experimental API designs
    - Research prototypes

Available Detectors:
    - SAM3Detector: Segment Anything Model 3 for image segmentation
    - HybridDetector: Combines multiple detection strategies with consensus voting
    - ConsistencyDetector: Finds elements consistent across multiple screenshots
    - EdgeBasedDetector: Edge detection using Canny algorithm
    - ContourDetector: Adaptive thresholding and contour detection
    - MSERDetector: Maximally Stable Extremal Regions detection
    - ColorClusterDetector: Color-based k-means clustering detection
    - TemplateDetector: Corner detection and rectangular pattern matching

Example:
    >>> from qontinui.discovery.experimental import EdgeBasedDetector
    >>> # EXPERIMENTAL - Use at your own risk, API may change
    >>> detector = EdgeBasedDetector()
    >>> results = detector.detect("screenshot.png", canny_low=50, canny_high=150)
"""

from typing import List

# Import type definitions
from .types import BBox, MultiScreenshotDataset, ScreenshotInfo

# Import base classes
from .base_detector import BaseDetector, MultiScreenshotDetector

# Import experimental detectors
from .sam3_detector import SAM3Detector
from .hybrid_detector import HybridDetector
from .consistency_detector import ConsistencyDetector
from .edge_detector import EdgeBasedDetector
from .contour_detector import ContourDetector
from .mser_detector import MSERDetector
from .color_detector import ColorClusterDetector
from .template_detector import TemplateDetector

__all__: List[str] = [
    # Type definitions
    "BBox",
    "MultiScreenshotDataset",
    "ScreenshotInfo",
    # Base classes
    "BaseDetector",
    "MultiScreenshotDetector",
    # Experimental detectors
    "SAM3Detector",
    "HybridDetector",
    "ConsistencyDetector",
    "EdgeBasedDetector",
    "ContourDetector",
    "MSERDetector",
    "ColorClusterDetector",
    "TemplateDetector",
]

__version__ = "0.1.0-experimental"
