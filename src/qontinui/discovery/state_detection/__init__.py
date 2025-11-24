"""State detection module for identifying application states from screenshots.

This module provides functionality for detecting and classifying application states
by analyzing the visual characteristics of screenshots, including the presence of
specific elements, layout patterns, and visual features.

State detection is the core intelligence that enables automated state discovery,
allowing the system to recognize when the application is in different states and
understand the transitions between them.

Key Components:
    - State signature extraction
    - State matching and classification
    - Transition detection
    - State stability analysis
    - Probabilistic state inference
    - Multi-frame state validation

Example:
    >>> from qontinui.discovery.state_detection import DifferentialConsistencyDetector
    >>> detector = DifferentialConsistencyDetector()
    >>> result = detector.analyze_screenshots(screenshots)
    >>> print(f"Detected {len(result.consistent_regions)} consistent regions")
"""

from .detector import (
    DetectionMethod,
    MultiFrameValidator,
    SignatureBasedDetector,
    StateDetectionResult,
    StateDetector,
    StateSignature,
    TransitionDetector,
)
from .differential_consistency_detector import (
    DifferentialConsistencyDetector,
    StateRegion,
)

__all__ = [
    # Core detectors
    "DifferentialConsistencyDetector",
    "SignatureBasedDetector",
    "TransitionDetector",
    # Base classes
    "StateDetector",
    # Data models
    "StateRegion",
    "StateSignature",
    "StateDetectionResult",
    "DetectionMethod",
    # Validators
    "MultiFrameValidator",
]

__version__ = "0.1.0"
