"""Transition Analysis Stage for the Analysis Pipeline.

This module provides the transition analysis stage that wraps the
TransitionAnalyzer and provides pipeline-specific functionality.
"""

import logging
import time

from qontinui.discovery.state_analysis.models import (
    DetectedState,
    Frame,
    InputEvent,
    ProcessingStep,
    Transition,
)
from qontinui.discovery.state_analysis.transition_analyzer import TransitionAnalyzer

logger = logging.getLogger(__name__)


class TransitionAnalyzerStage:
    """Handles transition analysis stage in the pipeline.

    This class wraps the TransitionAnalyzer and provides additional
    pipeline-specific functionality like logging and error handling.
    """

    def __init__(
        self,
        event_correlation_window_ms: int = 1000,
        click_proximity_threshold: int = 50,
        min_visual_change_score: float = 0.1,
    ):
        """Initialize the TransitionAnalyzerStage.

        Args:
            event_correlation_window_ms: Maximum time window to correlate events
            click_proximity_threshold: Maximum distance to match click with StateImage
            min_visual_change_score: Minimum visual change score for transitions
        """
        self.event_correlation_window_ms = event_correlation_window_ms
        self.click_proximity_threshold = click_proximity_threshold
        self.min_visual_change_score = min_visual_change_score
        self.analyzer = TransitionAnalyzer(
            event_correlation_window_ms=event_correlation_window_ms,
            click_proximity_threshold=click_proximity_threshold,
            min_visual_change_score=min_visual_change_score,
        )

    def analyze_transitions(
        self, states: list[DetectedState], events: list[InputEvent], frames: list[Frame]
    ) -> tuple[list[Transition], ProcessingStep]:
        """Run transition analysis.

        Args:
            states: Detected states
            events: Input events
            frames: All frames

        Returns:
            Tuple of (transitions, processing step)
        """
        start_time = time.time()
        transitions = []
        error = None
        success = False

        try:
            transitions = self.analyzer.analyze_transitions(states, events, frames)  # type: ignore[arg-type]
            success = True

            logger.info("Transition analysis complete: %d transitions found", len(transitions))
            for transition in transitions:
                logger.debug(
                    "  - %s: %s -> %s (%s)",
                    transition.id,
                    transition.get_primary_source_state(),
                    transition.get_primary_target_state(),
                    transition.action_type,
                )

        except Exception as e:
            error = str(e)
            logger.error("Transition analysis failed: %s", error, exc_info=True)

        end_time = time.time()

        return transitions, ProcessingStep(
            name="transition_analysis",
            start_time=start_time,
            end_time=end_time,
            input_count=len(states) + len(events),
            output_count=len(transitions),
            parameters={
                "event_correlation_window_ms": self.event_correlation_window_ms,
                "click_proximity_threshold": self.click_proximity_threshold,
                "min_visual_change_score": self.min_visual_change_score,
            },
            success=success,
            error=error,
            metadata={
                "num_states": len(states),
                "num_events": len(events),
            },
        )
