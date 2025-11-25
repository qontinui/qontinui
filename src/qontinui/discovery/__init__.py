"""State Discovery system for automated state and StateImage detection.

This module provides comprehensive functionality for discovering, detecting, and
constructing application states through automated analysis of screenshots and
UI elements.

Submodules:
    - element_detection: UI element identification and classification
    - region_analysis: Screen region segmentation and analysis
    - state_detection: State identification and matching
    - state_construction: State object creation from detected elements
    - experimental: Research and experimental detection features
    - pixel_analysis: Low-level pixel stability analysis

The discovery system forms the foundation for automated testing and exploration
by enabling the system to understand and navigate application states without
manual state definition.

Example:
    >>> from qontinui.discovery.state_detection import DifferentialConsistencyDetector
    >>> from qontinui.discovery.state_construction import StateBuilder
    >>>
    >>> # Detect consistent regions across screenshots
    >>> detector = DifferentialConsistencyDetector()
    >>> result = detector.analyze_screenshots(screenshots)
    >>>
    >>> # Build state objects from detected elements
    >>> builder = StateBuilder()
    >>> state = builder.build_state(elements, screenshot)
"""

# Base classes
from .base_detector import BaseDetector
from .models import AnalysisResult, DiscoveredState, StateImage
from .multi_screenshot_detector import MultiScreenshotDetector

# Pixel analysis
from .pixel_analysis.analyzers import PixelStabilityAnalyzer
from .pixel_analysis.extractor import StableRegionExtractor
from .pixel_stability_matrix_analyzer import PixelStabilityMatrixAnalyzer

# State detection - import key classes for convenience
from .state_detection import (
    DifferentialConsistencyDetector,
    StateDetector,
    TransitionDetector,
)

# State construction - import key classes for convenience
from .state_construction import (
    ElementIdentifier,
    OCRNameGenerator,
    StateBuilder,
    TransitionInfo,
)

# Click analysis - import key classes for convenience
from .click_analysis import (
    ClickBoundingBoxInferrer,
    ElementBoundaryFinder,
    ClickContextAnalyzer,
    InferredBoundingBox,
    InferenceConfig,
    InferenceResult,
    ElementType,
    DetectionStrategy,
    infer_bbox_from_click,
)

# Submodules available for import
# from qontinui.discovery import element_detection
# from qontinui.discovery import region_analysis
# from qontinui.discovery import state_detection
# from qontinui.discovery import state_construction
# from qontinui.discovery import experimental
# from qontinui.discovery import click_analysis

__all__ = [
    # Base classes
    "BaseDetector",
    "MultiScreenshotDetector",
    # Pixel analysis
    "PixelStabilityAnalyzer",
    "PixelStabilityMatrixAnalyzer",
    "StableRegionExtractor",
    # Models
    "StateImage",
    "DiscoveredState",
    "AnalysisResult",
    # State detection (convenience exports)
    "DifferentialConsistencyDetector",
    "StateDetector",
    "TransitionDetector",
    # State construction (convenience exports)
    "StateBuilder",
    "ElementIdentifier",
    "OCRNameGenerator",
    "TransitionInfo",
    # Click analysis (convenience exports)
    "ClickBoundingBoxInferrer",
    "ElementBoundaryFinder",
    "ClickContextAnalyzer",
    "InferredBoundingBox",
    "InferenceConfig",
    "InferenceResult",
    "ElementType",
    "DetectionStrategy",
    "infer_bbox_from_click",
    # Submodules
    "element_detection",
    "region_analysis",
    "state_detection",
    "state_construction",
    "experimental",
    "click_analysis",
]
