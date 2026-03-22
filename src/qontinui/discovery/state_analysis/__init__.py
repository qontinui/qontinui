"""State Analysis Module for qontinui.

This module provides comprehensive state analysis capabilities for GUI automation,
including:
- State boundary detection using visual clustering
- Image extraction from detected states
- Transition analysis correlating states with input events
- Complete analysis pipeline orchestration

The analysis pipeline processes video frames and input events to automatically
identify unique GUI states and transitions between them.
"""

from qontinui.discovery.state_analysis.image_extractor import (
    ImageExtractionConfig,
    StateImageExtractor,
)
from qontinui.discovery.state_analysis.pipeline import (
    AnalysisPipeline,
    AnalysisResult,
)
from qontinui.discovery.state_analysis.pipeline import CaptureSession as PipelineCaptureSession
from qontinui.discovery.state_analysis.pipeline import (
    PipelineConfig,
    load_session_from_video,
    save_analysis_result,
)
from qontinui.discovery.state_analysis.state_boundary_detector import (
    FrameFeatures,
    StateBoundaryConfig,
    StateBoundaryDetector,
    TransitionPoint,
)
from qontinui.discovery.state_analysis.transition_analyzer import (
    AutoTransitionBuilder,
    CaptureSession,
    TransitionAnalyzer,
)

__all__ = [
    # State Boundary Detection
    "StateBoundaryDetector",
    "StateBoundaryConfig",
    "FrameFeatures",
    "TransitionPoint",
    # Image Extraction
    "StateImageExtractor",
    "ImageExtractionConfig",
    # Transition Analysis
    "TransitionAnalyzer",
    "AutoTransitionBuilder",
    "CaptureSession",
    # Pipeline
    "AnalysisPipeline",
    "AnalysisResult",
    "PipelineCaptureSession",
    "PipelineConfig",
    "load_session_from_video",
    "save_analysis_result",
]
