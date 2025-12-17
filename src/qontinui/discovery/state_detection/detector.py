"""State detection implementation.

This module provides functionality for detecting and matching application states
from screenshots using various detection strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np


class DetectionMethod(Enum):
    """State detection methods."""

    SIGNATURE = "signature"  # Match against state signatures
    TEMPLATE = "template"  # Template matching
    FEATURE = "feature"  # Feature-based matching
    ML = "ml"  # Machine learning classification
    HYBRID = "hybrid"  # Combination of methods


@dataclass
class StateSignature:
    """Represents a visual signature of a state.

    A signature defines the unique visual characteristics that identify a state,
    including required elements, layout patterns, and visual features.

    Attributes:
        state_id: Unique identifier for the state
        required_elements: Element IDs that must be present
        optional_elements: Element IDs that may be present
        negative_elements: Element IDs that should NOT be present
        layout_constraints: Spatial relationships between elements
        visual_features: Additional visual features (colors, patterns, etc.)
    """

    state_id: str
    required_elements: set[str]
    optional_elements: set[str] | None = None
    negative_elements: set[str] | None = None
    layout_constraints: dict | None = None
    visual_features: dict | None = None

    def __repr__(self) -> str:
        """String representation of signature."""
        return (
            f"StateSignature(id={self.state_id}, "
            f"required={len(self.required_elements)})"
        )


@dataclass
class StateDetectionResult:
    """Result of state detection.

    Attributes:
        state_id: ID of detected state (None if no match)
        confidence: Detection confidence (0.0 to 1.0)
        matching_elements: Elements that matched the signature
        missing_elements: Required elements that were not found
        method: Detection method used
        metadata: Additional detection metadata
    """

    state_id: str | None
    confidence: float
    matching_elements: list[str]
    missing_elements: list[str] | None = None
    method: DetectionMethod = DetectionMethod.SIGNATURE
    metadata: dict | None = None

    def __repr__(self) -> str:
        """String representation of detection result."""
        return (
            f"StateDetectionResult(state={self.state_id}, "
            f"confidence={self.confidence:.3f}, method={self.method.value})"
        )


class StateDetector(ABC):
    """Abstract base class for state detection."""

    @abstractmethod
    def detect(self, screenshot: np.ndarray) -> StateDetectionResult:
        """Detect the current state from a screenshot.

        Args:
            screenshot: Screenshot image

        Returns:
            Detection result with state ID and confidence
        """
        pass

    @abstractmethod
    def register_state(self, state_id: str, signature: StateSignature) -> None:
        """Register a state signature for detection.

        Args:
            state_id: Unique state identifier
            signature: State signature to register
        """
        pass


class SignatureBasedDetector(StateDetector):
    """State detector using signature matching.

    Matches screenshots against registered state signatures by checking
    for the presence of required elements and validating constraints.
    """

    def __init__(self):
        """Initialize signature-based detector."""
        self.signatures: dict[str, StateSignature] = {}
        self.confidence_threshold = 0.7
        self.element_detector = None  # Placeholder for element detector

    def detect(self, screenshot: np.ndarray) -> StateDetectionResult:
        """Detect state using signature matching.

        TODO: Integrate with element detection to find elements and match signatures.

        Args:
            screenshot: Screenshot image

        Returns:
            Detection result
        """
        # Placeholder implementation
        # In real implementation:
        # 1. Detect elements in screenshot
        # 2. Match detected elements against each signature
        # 3. Calculate confidence based on matches
        # 4. Return best matching state

        return StateDetectionResult(
            state_id=None,
            confidence=0.0,
            matching_elements=[],
            method=DetectionMethod.SIGNATURE,
        )

    def register_state(self, state_id: str, signature: StateSignature) -> None:
        """Register a state signature.

        Args:
            state_id: State identifier
            signature: State signature to register
        """
        self.signatures[state_id] = signature

    def calculate_match_score(
        self, detected_elements: set[str], signature: StateSignature
    ) -> float:
        """Calculate how well detected elements match a signature.

        Args:
            detected_elements: Set of detected element IDs
            signature: State signature to match against

        Returns:
            Match score between 0.0 and 1.0
        """
        # Check required elements
        required_found = len(
            signature.required_elements.intersection(detected_elements)
        )
        required_total = len(signature.required_elements)

        if required_total == 0:
            required_score = 1.0
        else:
            required_score = required_found / required_total

        # If not all required elements found, low score
        if required_score < 1.0:
            return required_score * 0.5  # Penalize missing required elements

        # Check optional elements
        optional_elements = signature.optional_elements or set()
        if optional_elements:
            optional_found = len(optional_elements.intersection(detected_elements))
            optional_total = len(optional_elements)
            optional_score = optional_found / optional_total
        else:
            optional_score = 1.0

        # Check negative elements (should NOT be present)
        negative_elements = signature.negative_elements or set()
        if negative_elements:
            negative_found = len(negative_elements.intersection(detected_elements))
            if negative_found > 0:
                # Penalize presence of negative elements
                return required_score * 0.3
            negative_score = 1.0
        else:
            negative_score = 1.0

        # Combine scores
        final_score = required_score * 0.6 + optional_score * 0.2 + negative_score * 0.2

        return final_score


class TransitionDetector:
    """Detects state transitions between frames.

    Analyzes sequences of frames to identify when the application
    transitions from one state to another.
    """

    def __init__(self):
        """Initialize transition detector."""
        self.previous_state: str | None = None
        self.state_detector: StateDetector | None = None
        self.stability_threshold = 3  # Frames to confirm stable state

    def detect_transition(
        self, current_frame: np.ndarray, previous_frame: np.ndarray | None = None
    ) -> tuple[str, str] | None:
        """Detect if a state transition occurred.

        Args:
            current_frame: Current screenshot
            previous_frame: Previous screenshot (optional)

        Returns:
            Tuple of (from_state, to_state) if transition detected, None otherwise
        """
        if self.state_detector is None or previous_frame is None:
            return None

        # Detect state in both frames
        previous_result = self.state_detector.detect(previous_frame)
        current_result = self.state_detector.detect(current_frame)

        # Check if we have valid detections with sufficient confidence
        if (
            previous_result.state_id is None
            or current_result.state_id is None
            or previous_result.confidence < 0.5
            or current_result.confidence < 0.5
        ):
            return None

        # Check if state changed
        if previous_result.state_id != current_result.state_id:
            # Update tracked previous state
            from_state = self.previous_state or previous_result.state_id
            to_state = current_result.state_id
            self.previous_state = to_state
            return (from_state, to_state)

        # Update previous state even if no transition
        self.previous_state = current_result.state_id
        return None

    def is_stable_state(self, frames: list[np.ndarray]) -> bool:
        """Check if the state is stable across multiple frames.

        Args:
            frames: List of recent frames

        Returns:
            True if state is stable
        """
        if len(frames) < self.stability_threshold:
            return False

        if self.state_detector is None:
            return False

        # Detect state in all frames
        detected_states: list[str | None] = []
        for frame in frames:
            result = self.state_detector.detect(frame)
            # Only consider high-confidence detections
            if result.confidence >= 0.7:
                detected_states.append(result.state_id)
            else:
                detected_states.append(None)

        # Check if all detections are valid and identical
        if None in detected_states:
            return False

        # All frames should detect the same state
        unique_states = set(detected_states)
        return len(unique_states) == 1 and None not in unique_states


class MultiFrameValidator:
    """Validates state detection across multiple frames.

    Reduces false positives by requiring consistent detection
    across multiple consecutive frames.
    """

    def __init__(self, required_frames: int = 3):
        """Initialize multi-frame validator.

        Args:
            required_frames: Number of consecutive frames required for confirmation
        """
        self.required_frames = required_frames
        self.frame_buffer: list[StateDetectionResult] = []

    def validate(self, detection: StateDetectionResult) -> str | None:
        """Validate state detection with multi-frame confirmation.

        Args:
            detection: Current frame detection result

        Returns:
            Confirmed state ID if validation passes, None otherwise
        """
        self.frame_buffer.append(detection)

        # Keep only recent frames
        if len(self.frame_buffer) > self.required_frames:
            self.frame_buffer.pop(0)

        # Check if we have enough frames
        if len(self.frame_buffer) < self.required_frames:
            return None

        # Check if all recent frames agree
        state_ids = [d.state_id for d in self.frame_buffer]
        if len(set(state_ids)) == 1 and state_ids[0] is not None:
            return state_ids[0]

        return None

    def reset(self) -> None:
        """Reset the frame buffer."""
        self.frame_buffer.clear()
