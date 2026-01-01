"""State Boundary Detection Stage for the Analysis Pipeline.

This module provides the state boundary detection stage that wraps the
StateBoundaryDetector and provides pipeline-specific functionality.
"""

import logging
import time

from qontinui.discovery.state_analysis.models import (
    DetectedState,
    Frame,
    ProcessingStep,
)
from qontinui.discovery.state_analysis.state_boundary_detector import (
    StateBoundaryConfig,
    StateBoundaryDetector,
)

logger = logging.getLogger(__name__)


class BoundaryDetector:
    """Handles state boundary detection stage in the pipeline.

    This class wraps the StateBoundaryDetector and provides additional
    pipeline-specific functionality like logging and error handling.
    """

    def __init__(self, config: StateBoundaryConfig):
        """Initialize the BoundaryDetector.

        Args:
            config: Configuration for state boundary detection
        """
        self.config = config
        self.detector = StateBoundaryDetector(config=config)

    def detect_states(self, frames: list[Frame]) -> tuple[list[DetectedState], ProcessingStep]:
        """Run state boundary detection.

        Args:
            frames: List of frames to analyze

        Returns:
            Tuple of (detected states, processing step)
        """
        start_time = time.time()
        states = []
        error = None
        success = False

        try:
            states = self.detector.detect_states(frames)
            success = True

            logger.info("State detection complete: %d states detected", len(states))
            for state in states:
                logger.debug(
                    "  - %s: %d frames (indices %d-%d)",
                    state.name,
                    len(state.frame_indices),
                    state.start_frame_index,
                    state.end_frame_index,
                )

        except Exception as e:
            error = str(e)
            logger.error("State detection failed: %s", error, exc_info=True)

        end_time = time.time()

        return states, ProcessingStep(  # type: ignore[return-value]
            name="state_detection",
            start_time=start_time,
            end_time=end_time,
            input_count=len(frames),
            output_count=len(states),
            parameters=self.config.__dict__,
            success=success,
            error=error,
            metadata={
                "clustering_algorithm": self.config.clustering_algorithm,
                "similarity_threshold": self.config.similarity_threshold,
            },
        )
