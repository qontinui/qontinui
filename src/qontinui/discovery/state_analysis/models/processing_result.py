"""ProcessingResult data models for video capture architecture.

This module defines the ProcessingResult, ProcessingLog, and ProcessingStep
models used to capture the complete output of video analysis processing.

A ProcessingResult contains:
- All detected states and transitions
- Processing logs and metadata
- References to source data
- Performance and confidence metrics
"""

from dataclasses import dataclass, field
from typing import Any

from qontinui.discovery.state_analysis.models.state_models import DetectedState
from qontinui.discovery.state_analysis.models.transition import Transition


@dataclass
class ProcessingStep:
    """Represents a single step in the video processing pipeline.

    Each step corresponds to a distinct phase of processing (e.g., frame extraction,
    clustering, state detection, transition analysis) and tracks its inputs, outputs,
    timing, and success status.

    Attributes:
        name: Name of the processing step (e.g., "frame_extraction", "clustering")
        start_time: Unix timestamp when step started
        end_time: Unix timestamp when step completed
        input_count: Number of input items processed (e.g., frames, clusters)
        output_count: Number of output items generated (e.g., states, transitions)
        parameters: Dictionary of parameters used for this step
        success: Whether the step completed successfully
        error: Error message if step failed (None if successful)
        metadata: Additional step-specific data
    """

    name: str
    start_time: float
    end_time: float
    input_count: int
    output_count: int
    parameters: dict[str, Any]
    success: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Calculate the duration of this step in seconds.

        Returns:
            Duration in seconds
        """
        return self.end_time - self.start_time

    @property
    def duration_ms(self) -> float:
        """Calculate the duration of this step in milliseconds.

        Returns:
            Duration in milliseconds
        """
        return self.duration * 1000

    @property
    def throughput(self) -> float:
        """Calculate throughput (items processed per second).

        Returns:
            Throughput in items/second, or 0 if duration is zero
        """
        if self.duration == 0:
            return 0.0
        return self.input_count / self.duration

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary containing all serializable fields
        """
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "duration_ms": self.duration_ms,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "throughput": self.throughput,
            "parameters": self.parameters,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessingStep":
        """Create ProcessingStep from dictionary.

        Args:
            data: Dictionary containing step data

        Returns:
            New ProcessingStep instance
        """
        # Remove computed properties
        data_copy = {
            k: v
            for k, v in data.items()
            if k not in ["duration", "duration_ms", "throughput"]
        }

        return cls(
            name=data_copy["name"],
            start_time=data_copy["start_time"],
            end_time=data_copy["end_time"],
            input_count=data_copy["input_count"],
            output_count=data_copy["output_count"],
            parameters=data_copy["parameters"],
            success=data_copy["success"],
            error=data_copy.get("error"),
            metadata=data_copy.get("metadata", {}),
        )


@dataclass
class ProcessingLog:
    """Tracks the complete processing pipeline execution.

    The ProcessingLog provides a detailed record of all processing steps,
    including timing, parameters, and outcomes. This enables debugging,
    performance analysis, and quality assessment.

    Attributes:
        steps: List of ProcessingStep objects in execution order
        parameters_used: Global parameters used for the entire processing run
        confidence_scores: Dictionary of confidence metrics (e.g., avg_state_confidence)
        start_time: Unix timestamp when processing started
        end_time: Unix timestamp when processing completed
        errors: List of error messages encountered during processing
        metadata: Additional processing metadata
    """

    steps: list[ProcessingStep]
    parameters_used: dict[str, Any]
    confidence_scores: dict[str, float]
    start_time: float
    end_time: float
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_duration(self) -> float:
        """Calculate total processing duration in seconds.

        Returns:
            Total duration in seconds
        """
        return self.end_time - self.start_time

    @property
    def total_duration_ms(self) -> float:
        """Calculate total processing duration in milliseconds.

        Returns:
            Total duration in milliseconds
        """
        return self.total_duration * 1000

    @property
    def success(self) -> bool:
        """Check if all processing steps succeeded.

        Returns:
            True if all steps succeeded and no errors occurred
        """
        return all(step.success for step in self.steps) and len(self.errors) == 0

    @property
    def num_steps(self) -> int:
        """Get the number of processing steps.

        Returns:
            Count of steps
        """
        return len(self.steps)

    def get_step_by_name(self, name: str) -> ProcessingStep | None:
        """Find a processing step by name.

        Args:
            name: Name of the step to find

        Returns:
            ProcessingStep if found, None otherwise
        """
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary containing all serializable fields
        """
        return {
            "steps": [step.to_dict() for step in self.steps],
            "parameters_used": self.parameters_used,
            "confidence_scores": self.confidence_scores,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_duration": self.total_duration,
            "total_duration_ms": self.total_duration_ms,
            "success": self.success,
            "num_steps": self.num_steps,
            "errors": self.errors,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessingLog":
        """Create ProcessingLog from dictionary.

        Args:
            data: Dictionary containing log data

        Returns:
            New ProcessingLog instance
        """
        return cls(
            steps=[ProcessingStep.from_dict(step) for step in data["steps"]],
            parameters_used=data["parameters_used"],
            confidence_scores=data["confidence_scores"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            errors=data.get("errors", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ProcessingResult:
    """Complete result of video capture processing.

    The ProcessingResult is the top-level output of the video analysis pipeline.
    It contains all detected states, transitions, processing logs, and references
    to source data.

    This serves as the primary data structure for:
    - Saving processing results to disk
    - Transmitting results to other components
    - Generating reports and visualizations

    Attributes:
        session_id: Unique identifier for the capture session
        states: List of DetectedState objects found in the video
        transitions: List of Transition objects between states
        processing_log: ProcessingLog tracking the execution pipeline
        source_screenshots: List of file paths to source screenshot/frame files
        metadata: Additional result metadata
    """

    session_id: str
    states: list[DetectedState]
    transitions: list[Transition]
    processing_log: ProcessingLog
    source_screenshots: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_states(self) -> int:
        """Get the number of detected states.

        Returns:
            Count of states
        """
        return len(self.states)

    @property
    def num_transitions(self) -> int:
        """Get the number of detected transitions.

        Returns:
            Count of transitions
        """
        return len(self.transitions)

    @property
    def num_screenshots(self) -> int:
        """Get the number of source screenshots.

        Returns:
            Count of screenshots
        """
        return len(self.source_screenshots)

    @property
    def processing_success(self) -> bool:
        """Check if processing completed successfully.

        Returns:
            True if processing log indicates success
        """
        return self.processing_log.success

    def get_state_by_id(self, state_id: str) -> DetectedState | None:
        """Find a state by its ID.

        Args:
            state_id: ID of the state to find

        Returns:
            DetectedState if found, None otherwise
        """
        for state in self.states:
            if state.id == state_id:
                return state
        return None

    def get_transition_by_id(self, transition_id: str) -> Transition | None:
        """Find a transition by its ID.

        Args:
            transition_id: ID of the transition to find

        Returns:
            Transition if found, None otherwise
        """
        for transition in self.transitions:
            if transition.id == transition_id:
                return transition
        return None

    def get_transitions_from_state(self, state_id: str) -> list[Transition]:
        """Get all transitions originating from a specific state.

        Args:
            state_id: ID of the source state

        Returns:
            List of transitions (may be empty)
        """
        return [t for t in self.transitions if state_id in t.source_states]

    def get_transitions_to_state(self, state_id: str) -> list[Transition]:
        """Get all transitions leading to a specific state.

        Args:
            state_id: ID of the target state

        Returns:
            List of transitions (may be empty)
        """
        return [t for t in self.transitions if state_id in t.target_states]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary containing all serializable fields
        """
        return {
            "session_id": self.session_id,
            "states": [state.to_dict() for state in self.states],
            "transitions": [transition.to_dict() for transition in self.transitions],
            "processing_log": self.processing_log.to_dict(),
            "source_screenshots": self.source_screenshots,
            "num_states": self.num_states,
            "num_transitions": self.num_transitions,
            "num_screenshots": self.num_screenshots,
            "processing_success": self.processing_success,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessingResult":
        """Create ProcessingResult from dictionary.

        Args:
            data: Dictionary containing result data

        Returns:
            New ProcessingResult instance
        """
        return cls(
            session_id=data["session_id"],
            states=[DetectedState.from_dict(state) for state in data["states"]],
            transitions=[Transition.from_dict(t) for t in data["transitions"]],
            processing_log=ProcessingLog.from_dict(data["processing_log"]),
            source_screenshots=data.get("source_screenshots", []),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ProcessingResult(session_id={self.session_id}, "
            f"states={self.num_states}, transitions={self.num_transitions}, "
            f"screenshots={self.num_screenshots}, success={self.processing_success})"
        )
