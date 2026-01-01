"""Analysis Pipeline Orchestrator for qontinui-runner.

This module provides a unified pipeline that orchestrates all analysis components:
1. State Boundary Detection - Identifies unique screen states from frames
2. Image Extraction - Extracts StateImages from detected states
3. Element Detection - Identifies UI elements within states
4. Transition Analysis - Correlates states with input events

The pipeline generates detailed processing logs for review and supports
re-running with different parameters for optimization.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from qontinui.discovery.state_analysis.boundary_detector import BoundaryDetector
from qontinui.discovery.state_analysis.frame_analyzer import FrameAnalyzer
from qontinui.discovery.state_analysis.image_extractor import ImageExtractionConfig
from qontinui.discovery.state_analysis.metadata_generator import MetadataGenerator
from qontinui.discovery.state_analysis.models import (
    DetectedState,
    Frame,
    InputEvent,
    ProcessingLog,
    ProcessingResult,
    ProcessingStep,
    Transition,
)
from qontinui.discovery.state_analysis.state_boundary_detector import (
    StateBoundaryConfig,
)
from qontinui.discovery.state_analysis.state_extractor import StateExtractor
from qontinui.discovery.state_analysis.transition_analyzer_stage import (
    TransitionAnalyzerStage,
)

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Classes
# ============================================================================


@dataclass
class PipelineConfig:
    """Configuration for the complete analysis pipeline.

    This configuration allows tuning of all pipeline stages. Each stage has
    its own configuration section.

    Attributes:
        # State Detection Configuration
        state_similarity_threshold: SSIM threshold for frame similarity (0.0-1.0)
        min_state_duration_ms: Minimum duration for a state to be valid
        clustering_algorithm: Algorithm to use ("dbscan", "hierarchical", "kmeans")
        feature_extractor: Feature extraction method ("orb", "sift", "surf")
        dbscan_eps: DBSCAN epsilon parameter
        dbscan_min_samples: DBSCAN minimum samples parameter

        # Image Extraction Configuration
        extract_at_clicks: Extract images at click locations
        click_region_padding: Padding around click locations in pixels
        min_image_size: Minimum image size (width, height)
        max_image_size: Maximum image size (width, height)
        extract_from_contours: Extract images from contour detection
        max_contours: Maximum number of contours to extract

        # Element Detection Configuration
        detect_buttons: Enable button detection
        detect_text_fields: Enable text field detection
        element_confidence_threshold: Confidence threshold for element detection

        # Transition Analysis Configuration
        event_correlation_window_ms: Time window to correlate events with state changes
        click_proximity_threshold: Max distance to match click with StateImage
        min_visual_change_score: Minimum visual change score for state transitions
    """

    # State detection
    state_similarity_threshold: float = 0.92
    min_state_duration_ms: int = 500
    clustering_algorithm: str = "dbscan"
    feature_extractor: str = "orb"
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 3

    # Image extraction
    extract_at_clicks: bool = True
    click_region_padding: int = 20
    min_image_size: tuple[int, int] = (20, 20)
    max_image_size: tuple[int, int] = (500, 500)
    extract_from_contours: bool = True
    max_contours: int = 50

    # Element detection
    detect_buttons: bool = True
    detect_text_fields: bool = True
    element_confidence_threshold: float = 0.7

    # Transition analysis
    event_correlation_window_ms: int = 1000
    click_proximity_threshold: int = 50
    min_visual_change_score: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        return {
            "state_similarity_threshold": self.state_similarity_threshold,
            "min_state_duration_ms": self.min_state_duration_ms,
            "clustering_algorithm": self.clustering_algorithm,
            "feature_extractor": self.feature_extractor,
            "dbscan_eps": self.dbscan_eps,
            "dbscan_min_samples": self.dbscan_min_samples,
            "extract_at_clicks": self.extract_at_clicks,
            "click_region_padding": self.click_region_padding,
            "min_image_size": self.min_image_size,
            "max_image_size": self.max_image_size,
            "extract_from_contours": self.extract_from_contours,
            "max_contours": self.max_contours,
            "detect_buttons": self.detect_buttons,
            "detect_text_fields": self.detect_text_fields,
            "element_confidence_threshold": self.element_confidence_threshold,
            "event_correlation_window_ms": self.event_correlation_window_ms,
            "click_proximity_threshold": self.click_proximity_threshold,
            "min_visual_change_score": self.min_visual_change_score,
        }


@dataclass
class CaptureSession:
    """Represents a complete capture session with all necessary data.

    Attributes:
        session_id: Unique identifier for this session
        video_path: Path to the video file (if applicable)
        events_path: Path to the events file (if applicable)
        fps: Frames per second of the video
        frames: List of extracted frames
        events: List of input events
        metadata: Additional session metadata
    """

    session_id: str
    video_path: str
    events_path: str
    fps: int = 30
    frames: list[Frame] = field(default_factory=list)
    events: list[InputEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize computed fields."""
        if not self.session_id:
            self.session_id = str(uuid.uuid4())


# ============================================================================
# Analysis Result Classes
# ============================================================================


@dataclass
class AnalysisResult:
    """Complete result of pipeline analysis.

    This extends ProcessingResult with additional pipeline-specific information.

    Attributes:
        processing_result: The standard ProcessingResult
        session: The CaptureSession that was analyzed
        config: The PipelineConfig used for analysis
    """

    processing_result: ProcessingResult
    session: CaptureSession
    config: PipelineConfig

    @property
    def states(self) -> list[DetectedState]:
        """Get detected states."""
        return self.processing_result.states

    @property
    def transitions(self) -> list[Transition]:
        """Get detected transitions."""
        return self.processing_result.transitions

    @property
    def processing_log(self) -> ProcessingLog:
        """Get processing log."""
        return self.processing_result.processing_log

    @property
    def success(self) -> bool:
        """Check if analysis was successful."""
        return self.processing_result.processing_success

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "processing_result": self.processing_result.to_dict(),
            "session_id": self.session.session_id,
            "config": self.config.to_dict(),
            "success": self.success,
        }


# ============================================================================
# Main Pipeline Orchestrator
# ============================================================================


class AnalysisPipeline:
    """Orchestrates the complete analysis pipeline.

    This class coordinates all analysis components and provides:
    - End-to-end pipeline execution
    - Detailed logging of each processing stage
    - Configuration management
    - Result comparison for parameter tuning
    """

    def __init__(self, config: PipelineConfig | None = None):
        """Initialize the analysis pipeline.

        Args:
            config: Pipeline configuration (uses defaults if None)
        """
        self.config = config or PipelineConfig()
        logger.info("Initializing AnalysisPipeline with config: %s", self.config)

        # Initialize component configurations
        self.state_config = self._build_state_config()
        self.image_config = self._build_image_config()

        # Initialize pipeline stages
        self.frame_analyzer = FrameAnalyzer()
        self.boundary_detector = BoundaryDetector(config=self.state_config)
        self.state_extractor = StateExtractor(config=self.image_config)
        self.transition_analyzer = TransitionAnalyzerStage(
            event_correlation_window_ms=self.config.event_correlation_window_ms,
            click_proximity_threshold=self.config.click_proximity_threshold,
            min_visual_change_score=self.config.min_visual_change_score,
        )
        self.metadata_generator = MetadataGenerator()

        # Initialize processing log (will be reset for each analysis)
        self.processing_log: ProcessingLog | None = None

    def analyze_session(self, session: CaptureSession) -> AnalysisResult:
        """Run the complete analysis pipeline on a capture session.

        This is the main entry point for pipeline execution. It runs all phases:
        1. Frame validation and preparation
        2. State boundary detection
        3. StateImage extraction
        4. UI element detection (if enabled)
        5. Transition analysis
        6. Result compilation and logging

        Args:
            session: The capture session to analyze

        Returns:
            Complete AnalysisResult with states, transitions, and logs

        Raises:
            ValueError: If session is invalid or missing required data
        """
        logger.info("=" * 80)
        logger.info("Starting analysis pipeline for session: %s", session.session_id)
        logger.info("=" * 80)

        # Initialize processing log
        start_time = time.time()
        steps: list[ProcessingStep] = []
        errors: list[str] = []

        try:
            # Phase 1: Validate session
            logger.info("Phase 1: Validating session data...")
            validation_step = self._validate_session(session)
            steps.append(validation_step)

            if not validation_step.success:
                raise ValueError(f"Session validation failed: {validation_step.error}")

            # Phase 2: State boundary detection
            logger.info("Phase 2: Detecting state boundaries...")
            states, detection_step = self._run_state_detection(session.frames)
            steps.append(detection_step)

            if not detection_step.success:
                errors.append(f"State detection failed: {detection_step.error}")
                states = []

            # Phase 3: Image extraction
            logger.info("Phase 3: Extracting StateImages from detected states...")
            extraction_step = self._run_image_extraction(states, session.frames, session.events)
            steps.append(extraction_step)

            if not extraction_step.success:
                errors.append(f"Image extraction failed: {extraction_step.error}")

            # Phase 4: Element detection (optional)
            if self.config.detect_buttons or self.config.detect_text_fields:
                logger.info("Phase 4: Detecting UI elements...")
                element_step = self._run_element_detection(states, session.frames)
                steps.append(element_step)

                if not element_step.success:
                    errors.append(f"Element detection failed: {element_step.error}")

            # Phase 5: Transition analysis
            logger.info("Phase 5: Analyzing state transitions...")
            transitions, transition_step = self._run_transition_analysis(
                states, session.events, session.frames
            )
            steps.append(transition_step)

            if not transition_step.success:
                errors.append(f"Transition analysis failed: {transition_step.error}")
                transitions = []

            # Phase 6: Calculate confidence scores
            logger.info("Phase 6: Calculating confidence scores...")
            confidence_scores = self._calculate_confidence_scores(states, transitions)

            # Build processing log
            end_time = time.time()
            self.processing_log = ProcessingLog(
                steps=steps,
                parameters_used=self.config.to_dict(),
                confidence_scores=confidence_scores,
                start_time=start_time,
                end_time=end_time,
                errors=errors,
                metadata={
                    "session_id": session.session_id,
                    "fps": session.fps,
                    "total_frames": len(session.frames),
                    "total_events": len(session.events),
                },
            )

            # Build processing result
            processing_result = ProcessingResult(
                session_id=session.session_id,
                states=states,
                transitions=transitions,
                processing_log=self.processing_log,
                source_screenshots=[],  # Could be populated with frame paths
                metadata={
                    "video_path": session.video_path,
                    "events_path": session.events_path,
                    "fps": session.fps,
                },
            )

            # Build analysis result
            result = AnalysisResult(
                processing_result=processing_result,
                session=session,
                config=self.config,
            )

            logger.info("=" * 80)
            logger.info("Pipeline analysis complete!")
            logger.info(
                "Results: %d states, %d transitions, %d errors",
                len(states),
                len(transitions),
                len(errors),
            )
            logger.info("Total duration: %.2fs", self.processing_log.total_duration)
            logger.info("=" * 80)

            return result

        except Exception as e:
            logger.error("Pipeline analysis failed with exception: %s", e, exc_info=True)
            errors.append(str(e))

            # Create error processing log
            end_time = time.time()
            self.processing_log = ProcessingLog(
                steps=steps,
                parameters_used=self.config.to_dict(),
                confidence_scores={},
                start_time=start_time,
                end_time=end_time,
                errors=errors,
                metadata={"session_id": session.session_id, "failed": True},
            )

            # Return empty result with error log
            processing_result = ProcessingResult(
                session_id=session.session_id,
                states=[],
                transitions=[],
                processing_log=self.processing_log,
            )

            return AnalysisResult(
                processing_result=processing_result,
                session=session,
                config=self.config,
            )

    def run_phase(self, phase: str, session: CaptureSession, **kwargs) -> Any:
        """Run a single phase of the pipeline.

        This allows running individual pipeline phases for debugging or
        incremental processing.

        Args:
            phase: Name of the phase to run ("state_detection", "image_extraction",
                   "element_detection", "transition_analysis")
            session: The capture session to process
            **kwargs: Additional phase-specific parameters

        Returns:
            Phase-specific results

        Raises:
            ValueError: If phase name is invalid
        """
        logger.info("Running single phase: %s", phase)

        if phase == "state_detection":
            states, step = self._run_state_detection(session.frames)
            return states

        elif phase == "image_extraction":
            states = kwargs.get("states")  # type: ignore[assignment]
            if not states:
                raise ValueError("image_extraction phase requires 'states' parameter")
            step = self._run_image_extraction(states, session.frames, session.events)
            return states  # Modified in place

        elif phase == "element_detection":
            states = kwargs.get("states")  # type: ignore[assignment]
            if not states:
                raise ValueError("element_detection phase requires 'states' parameter")
            step = self._run_element_detection(states, session.frames)
            return states  # Modified in place

        elif phase == "transition_analysis":
            states = kwargs.get("states")  # type: ignore[assignment]
            if not states:
                raise ValueError("transition_analysis phase requires 'states' parameter")
            transitions, step = self._run_transition_analysis(
                states, session.events, session.frames
            )
            return transitions

        else:
            raise ValueError(
                f"Unknown phase: {phase}. Valid phases: state_detection, "
                "image_extraction, element_detection, transition_analysis"
            )

    def rerun_with_config(
        self, session: CaptureSession, new_config: PipelineConfig
    ) -> AnalysisResult:
        """Re-run analysis with different configuration parameters.

        This is useful for parameter tuning and comparison. The pipeline is
        re-initialized with the new configuration and analysis is run from scratch.

        Args:
            session: The capture session to re-analyze
            new_config: New pipeline configuration to use

        Returns:
            New AnalysisResult with updated configuration
        """
        logger.info("Re-running pipeline with new configuration")
        logger.info("Configuration changes:")

        # Log differences
        old_dict = self.config.to_dict()
        new_dict = new_config.to_dict()
        for key in new_dict:
            if old_dict.get(key) != new_dict.get(key):
                logger.info("  %s: %s -> %s", key, old_dict.get(key), new_dict.get(key))

        # Update configuration
        self.config = new_config
        self.state_config = self._build_state_config()
        self.image_config = self._build_image_config()

        # Re-initialize pipeline stages
        self.frame_analyzer = FrameAnalyzer()
        self.boundary_detector = BoundaryDetector(config=self.state_config)
        self.state_extractor = StateExtractor(config=self.image_config)
        self.transition_analyzer = TransitionAnalyzerStage(
            event_correlation_window_ms=self.config.event_correlation_window_ms,
            click_proximity_threshold=self.config.click_proximity_threshold,
            min_visual_change_score=self.config.min_visual_change_score,
        )
        self.metadata_generator = MetadataGenerator()

        # Re-run analysis
        return self.analyze_session(session)

    def get_processing_log(self) -> ProcessingLog | None:
        """Get the detailed processing log from the last analysis.

        Returns:
            ProcessingLog if analysis has been run, None otherwise
        """
        return self.processing_log

    def compare_results(self, result1: AnalysisResult, result2: AnalysisResult) -> dict[str, Any]:
        """Compare two analysis results for parameter tuning.

        This provides metrics to help evaluate which configuration produces
        better results.

        Args:
            result1: First analysis result
            result2: Second analysis result

        Returns:
            Dictionary with comparison metrics
        """
        logger.info("Comparing analysis results...")

        comparison = {
            "result1": {
                "num_states": len(result1.states),
                "num_transitions": len(result1.transitions),
                "num_state_images": sum(len(s.state_images) for s in result1.states),  # type: ignore[misc,attr-defined]
                "processing_duration_ms": result1.processing_log.total_duration_ms,
                "success": result1.success,
                "errors": len(result1.processing_log.errors),
                "config": result1.config.to_dict(),
            },
            "result2": {
                "num_states": len(result2.states),
                "num_transitions": len(result2.transitions),
                "num_state_images": sum(len(s.state_images) for s in result2.states),  # type: ignore[misc,attr-defined]
                "processing_duration_ms": result2.processing_log.total_duration_ms,
                "success": result2.success,
                "errors": len(result2.processing_log.errors),
                "config": result2.config.to_dict(),
            },
            "differences": {
                "states_delta": len(result2.states) - len(result1.states),
                "transitions_delta": len(result2.transitions) - len(result1.transitions),
                "state_images_delta": sum(len(s.state_images) for s in result2.states)  # type: ignore[misc,attr-defined]
                - sum(len(s.state_images) for s in result1.states),  # type: ignore[misc,attr-defined]
                "duration_delta_ms": result2.processing_log.total_duration_ms
                - result1.processing_log.total_duration_ms,
            },
            "recommendation": self._generate_recommendation(result1, result2),
        }

        logger.info("Comparison complete:")
        logger.info(
            "  Result 1: %d states, %d transitions",
            len(result1.states),
            len(result1.transitions),
        )
        logger.info(
            "  Result 2: %d states, %d transitions",
            len(result2.states),
            len(result2.transitions),
        )
        logger.info("  Recommendation: %s", comparison["recommendation"])

        return comparison

    # ========================================================================
    # Private Methods - Configuration Building
    # ========================================================================

    def _build_state_config(self) -> StateBoundaryConfig:
        """Build StateBoundaryConfig from pipeline config.

        Returns:
            StateBoundaryConfig instance
        """
        return StateBoundaryConfig(
            similarity_threshold=self.config.state_similarity_threshold,
            min_state_duration_ms=self.config.min_state_duration_ms,
            clustering_algorithm=self.config.clustering_algorithm,
            feature_extractor=self.config.feature_extractor,
            dbscan_eps=self.config.dbscan_eps,
            dbscan_min_samples=self.config.dbscan_min_samples,
        )

    def _build_image_config(self) -> ImageExtractionConfig:
        """Build ImageExtractionConfig from pipeline config.

        Returns:
            ImageExtractionConfig instance
        """
        return ImageExtractionConfig(
            min_size=self.config.min_image_size,
            max_size=self.config.max_image_size,
            extract_at_click_locations=self.config.extract_at_clicks,
            click_region_padding=self.config.click_region_padding,
            max_contours=self.config.max_contours,
        )

    # ========================================================================
    # Private Methods - Phase Execution
    # ========================================================================

    def _validate_session(self, session: CaptureSession) -> ProcessingStep:
        """Validate the capture session data.

        Args:
            session: Session to validate

        Returns:
            ProcessingStep with validation results
        """
        # Validate frames using frame_analyzer
        validation_step = self.frame_analyzer.validate_frames(
            session.frames, session.session_id, len(session.events)
        )

        if not session.events:
            logger.warning("Session has no input events")

        if validation_step.success:
            logger.info(
                "Session validation successful: %d frames, %d events",
                len(session.frames),
                len(session.events),
            )

        # Rename step to match expected name
        return ProcessingStep(
            name="session_validation",
            start_time=validation_step.start_time,
            end_time=validation_step.end_time,
            input_count=validation_step.input_count,
            output_count=validation_step.output_count,
            parameters=validation_step.parameters,
            success=validation_step.success,
            error=validation_step.error,
            metadata=validation_step.metadata,
        )

    def _run_state_detection(
        self, frames: list[Frame]
    ) -> tuple[list[DetectedState], ProcessingStep]:
        """Run state boundary detection phase.

        Args:
            frames: List of frames to analyze

        Returns:
            Tuple of (detected states, processing step)
        """
        return self.boundary_detector.detect_states(frames)

    def _run_image_extraction(
        self, states: list[DetectedState], frames: list[Frame], events: list[InputEvent]
    ) -> ProcessingStep:
        """Run StateImage extraction phase.

        Args:
            states: Detected states to extract images from
            frames: All frames
            events: Input events

        Returns:
            ProcessingStep with extraction results
        """
        return self.state_extractor.extract_images(states, frames, events)

    def _run_element_detection(
        self, states: list[DetectedState], frames: list[Frame]
    ) -> ProcessingStep:
        """Run UI element detection phase.

        Note: This is a placeholder for future element detection implementation.

        Args:
            states: States to detect elements in
            frames: All frames

        Returns:
            ProcessingStep with detection results
        """
        start_time = time.time()
        total_elements = 0
        error = None
        success = False

        try:
            # Placeholder: Element detection not yet implemented
            # Future implementation would use computer vision to detect buttons,
            # text fields, and other UI elements

            success = True
            logger.info("Element detection complete: %d elements detected", total_elements)

        except Exception as e:
            error = str(e)
            logger.error("Element detection failed: %s", error, exc_info=True)

        end_time = time.time()

        return ProcessingStep(
            name="element_detection",
            start_time=start_time,
            end_time=end_time,
            input_count=len(states),
            output_count=total_elements,
            parameters={
                "detect_buttons": self.config.detect_buttons,
                "detect_text_fields": self.config.detect_text_fields,
                "confidence_threshold": self.config.element_confidence_threshold,
            },
            success=success,
            error=error,
            metadata={"placeholder": True},
        )

    def _run_transition_analysis(
        self, states: list[DetectedState], events: list[InputEvent], frames: list[Frame]
    ) -> tuple[list[Transition], ProcessingStep]:
        """Run transition analysis phase.

        Args:
            states: Detected states
            events: Input events
            frames: All frames

        Returns:
            Tuple of (transitions, processing step)
        """
        return self.transition_analyzer.analyze_transitions(states, events, frames)

    # ========================================================================
    # Private Methods - Analysis and Reporting
    # ========================================================================

    def _calculate_confidence_scores(
        self, states: list[DetectedState], transitions: list[Transition]
    ) -> dict[str, float]:
        """Calculate confidence scores for the analysis.

        Args:
            states: Detected states
            transitions: Detected transitions

        Returns:
            Dictionary of confidence metrics
        """
        return self.metadata_generator.calculate_confidence_scores(states, transitions)

    def _generate_recommendation(self, result1: AnalysisResult, result2: AnalysisResult) -> str:
        """Generate a recommendation on which result is better.

        Args:
            result1: First result
            result2: Second result

        Returns:
            Recommendation string
        """
        result1_data = {
            "num_states": len(result1.states),
            "num_transitions": len(result1.transitions),
            "errors": len(result1.processing_log.errors),
        }
        result2_data = {
            "num_states": len(result2.states),
            "num_transitions": len(result2.transitions),
            "errors": len(result2.processing_log.errors),
        }
        return self.metadata_generator.generate_recommendation(result1_data, result2_data)


# ============================================================================
# Utility Functions
# ============================================================================


def load_session_from_video(video_path: str, events_path: str, fps: int = 30) -> CaptureSession:
    """Load a capture session from video and events files.

    Args:
        video_path: Path to the video file
        events_path: Path to the events file (JSON)
        fps: Frames per second to extract

    Returns:
        CaptureSession with loaded data

    Raises:
        FileNotFoundError: If video or events file not found
    """
    import json

    logger.info("Loading session from video: %s", video_path)

    # Load video frames
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_index / fps
        frames.append(
            Frame(
                image=frame,
                timestamp=timestamp,
                frame_index=frame_index,
                file_path=None,
            )
        )
        frame_index += 1

    cap.release()
    logger.info("Loaded %d frames from video", len(frames))

    # Load events
    events = []
    try:
        with open(events_path) as f:
            events_data = json.load(f)
            for event_dict in events_data:
                events.append(
                    InputEvent(
                        timestamp=event_dict["timestamp"],
                        event_type=event_dict["event_type"],
                        x=event_dict.get("x"),
                        y=event_dict.get("y"),
                        button=event_dict.get("button"),
                        key=event_dict.get("key"),
                        metadata=event_dict.get("metadata"),
                    )
                )
        logger.info("Loaded %d events from file", len(events))
    except FileNotFoundError:
        logger.warning("Events file not found: %s", events_path)

    # Create session
    session = CaptureSession(
        session_id=str(uuid.uuid4()),
        video_path=video_path,
        events_path=events_path,
        fps=fps,
        frames=frames,
        events=events,
    )

    return session


def save_analysis_result(result: AnalysisResult, output_dir: str) -> Path:
    """Save analysis result to disk.

    Args:
        result: The AnalysisResult to save
        output_dir: Directory to save results to

    Returns:
        Path to the saved result file
    """
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save processing result as JSON
    result_file = output_path / f"{result.session.session_id}_result.json"
    with open(result_file, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    logger.info("Saved analysis result to: %s", result_file)
    return result_file
