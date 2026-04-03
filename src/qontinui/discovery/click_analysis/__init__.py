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
    - Capture session processing for click-to-template workflow
    - Application-specific profile tuning
    - Parallel batch processing

Example:
    >>> from qontinui.discovery.click_analysis import ClickBoundingBoxInferrer
    >>>
    >>> inferrer = ClickBoundingBoxInferrer()
    >>> result = inferrer.infer_bounding_box(
    ...     screenshot=screenshot_array,
    ...     click_location=(350, 250)
    ... )
    >>> print(f"Found element: {result.bbox} with type {result.element_type}")

Click-to-Template Workflow:
    >>> from qontinui.discovery.click_analysis import CaptureProcessor
    >>>
    >>> processor = CaptureProcessor()
    >>> candidates = processor.process_capture_session(
    ...     video_path=Path("session.mp4"),
    ...     events_file=Path("events.jsonl"),
    ... )
    >>> for candidate in candidates:
    ...     print(f"Found: {candidate.element_type} at ({candidate.click_x}, {candidate.click_y})")

Classes:
    ClickBoundingBoxInferrer: Main class for inferring bounding boxes
    ElementBoundaryFinder: Finds element boundaries using multiple strategies
    ClickContextAnalyzer: Determines the type of element clicked
    InferredBoundingBox: Result dataclass with bbox, mask, and metadata
    CaptureProcessor: Processes capture sessions to extract template candidates
    BatchDetector: Parallel boundary detection for multiple frames
    ApplicationTuner: Learns optimal detection parameters for applications
    ClickTemplateCandidate: Template candidate extracted from click event
    ApplicationProfile: Detection profile for a specific application
"""

from .application_profile import ApplicationProfile, TuningMetrics, TuningResult
from .application_tuner import ApplicationTuner
from .approved_template import ApprovedTemplate
from .batch_detector import BatchDetectionItem, BatchDetectionResult, BatchDetector
from .boundary_finder import ElementBoundaryFinder
from .capture_processor import CaptureProcessor
from .co_occurrence_analyzer import CoOccurrenceAnalyzer, CoOccurrenceResult
from .context_analyzer import ClickContextAnalyzer
from .inferrer import ClickBoundingBoxInferrer, infer_bbox_from_click
from .models import (
    DetectionStrategy,
    ElementType,
    InferenceConfig,
    InferenceResult,
    InferredBoundingBox,
)
from .state_grouper import GroupingResult, StateGroup, StateGrouper
from .state_machine_builder import (
    ClickToStateMachineBuilder,
    StateDef,
    StateImageDef,
    StateMachineResult,
    TransitionDef,
)
from .template_candidate import ClickTemplateCandidate

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
    # Click-to-Template classes
    "CaptureProcessor",
    "BatchDetector",
    "BatchDetectionItem",
    "BatchDetectionResult",
    "ApplicationTuner",
    "ClickTemplateCandidate",
    "ApplicationProfile",
    "TuningMetrics",
    "TuningResult",
    # State machine building
    "ApprovedTemplate",
    "CoOccurrenceAnalyzer",
    "CoOccurrenceResult",
    "StateGrouper",
    "StateGroup",
    "GroupingResult",
    "ClickToStateMachineBuilder",
    "StateMachineResult",
    "StateDef",
    "StateImageDef",
    "TransitionDef",
]
