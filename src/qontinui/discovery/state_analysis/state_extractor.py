"""State Image Extraction Stage for the Analysis Pipeline.

This module provides the state image extraction stage that wraps the
StateImageExtractor and provides pipeline-specific functionality.
"""

import logging
import time

from qontinui.discovery.state_analysis.image_extractor import (
    ImageExtractionConfig,
    StateImageExtractor,
)
from qontinui.discovery.state_analysis.models import (
    DetectedState,
    Frame,
    InputEvent,
    ProcessingStep,
)

logger = logging.getLogger(__name__)


class StateExtractor:
    """Handles state image extraction stage in the pipeline.

    This class wraps the StateImageExtractor and provides additional
    pipeline-specific functionality like logging and error handling.
    """

    def __init__(self, config: ImageExtractionConfig):
        """Initialize the StateExtractor.

        Args:
            config: Configuration for image extraction
        """
        self.config = config
        self.extractor = StateImageExtractor(config=config)

    def extract_images(
        self, states: list[DetectedState], frames: list[Frame], events: list[InputEvent]
    ) -> ProcessingStep:
        """Run StateImage extraction.

        Args:
            states: Detected states to extract images from
            frames: All frames
            events: Input events

        Returns:
            ProcessingStep with extraction results
        """
        start_time = time.time()
        total_images = 0
        error = None
        success = False

        try:
            for state in states:
                extracted_images = self.extractor.extract_from_state(state, frames, events)
                state.state_images = extracted_images  # type: ignore[attr-defined]
                total_images += len(extracted_images)

            success = True
            logger.info("Image extraction complete: %d total images extracted", total_images)

            for state in states:
                logger.debug("  - %s: %d StateImages", state.name, len(state.state_images))  # type: ignore[attr-defined]

        except Exception as e:
            error = str(e)
            logger.error("Image extraction failed: %s", error, exc_info=True)

        end_time = time.time()

        return ProcessingStep(
            name="image_extraction",
            start_time=start_time,
            end_time=end_time,
            input_count=len(states),
            output_count=total_images,
            parameters=self.config.__dict__,
            success=success,
            error=error,
            metadata={
                "extract_at_clicks": self.config.extract_at_click_locations,
                "max_contours": self.config.max_contours,
            },
        )
