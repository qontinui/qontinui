"""Capture session processor for click-to-template system.

This module processes captured video + input event files to extract
template candidates from click events.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .application_profile import ApplicationProfile
from .boundary_finder import ElementBoundaryFinder
from .context_analyzer import ClickContextAnalyzer
from .models import DetectionStrategy, InferenceConfig, InferenceResult
from .template_candidate import ClickTemplateCandidate

logger = logging.getLogger(__name__)


class CaptureProcessor:
    """Processes capture sessions to extract template candidates.

    This is the main orchestrator for processing recorded video and
    input events to generate template candidates for review.

    Example:
        >>> processor = CaptureProcessor()
        >>> candidates = processor.process_capture_session(
        ...     video_path=Path("session.mp4"),
        ...     events_file=Path("events.jsonl"),
        ... )
        >>> print(f"Found {len(candidates)} template candidates")
    """

    def __init__(
        self,
        config: InferenceConfig | None = None,
        profile: ApplicationProfile | None = None,
    ) -> None:
        """Initialize the capture processor.

        Args:
            config: Inference configuration. Uses defaults if not provided.
            profile: Application profile for tuned detection parameters.
        """
        self.config = config or InferenceConfig()
        self.profile = profile
        self._apply_profile_config()

        self.boundary_finder = ElementBoundaryFinder(self.config)
        self.context_analyzer = ClickContextAnalyzer()

    def _apply_profile_config(self) -> None:
        """Apply profile-specific configuration if available."""
        if self.profile:
            self.config = self.profile.get_effective_config()

    def process_capture_session(
        self,
        video_path: Path,
        events_file: Path,
        session_id: str | None = None,
        application_hint: str | None = None,
    ) -> list[ClickTemplateCandidate]:
        """Process a complete capture session.

        Extracts frames at click timestamps from the video and runs
        boundary detection on each to generate template candidates.

        Args:
            video_path: Path to the captured video file.
            events_file: Path to the JSONL events file.
            session_id: Optional session ID. Generated if not provided.
            application_hint: Optional application name for profile lookup.

        Returns:
            List of ClickTemplateCandidate objects.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        logger.info(f"Processing capture session {session_id}")
        start_time = time.time()

        # Load click events
        click_events = self._load_click_events(events_file)
        logger.info(f"Loaded {len(click_events)} click events")

        if not click_events:
            logger.warning("No click events found in session")
            return []

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Video: {total_frames} frames at {fps:.1f} fps")

            candidates = []

            for event in click_events:
                # Calculate frame number from timestamp
                timestamp = event.get("timestamp", 0.0)
                frame_number = int(timestamp * fps)

                # Clamp to valid range
                frame_number = max(0, min(frame_number, total_frames - 1))

                # Extract frame
                frame = self._extract_frame(cap, frame_number)
                if frame is None:
                    logger.warning(f"Failed to extract frame {frame_number}")
                    continue

                # Detect boundary
                click_x = event.get("x", 0)
                click_y = event.get("y", 0)
                click_button = event.get("button", "left")

                result = self.detect_boundary(frame, (click_x, click_y))

                # Create candidate
                candidate = self._create_candidate(
                    result=result,
                    session_id=session_id,
                    click_x=click_x,
                    click_y=click_y,
                    click_button=click_button,
                    timestamp=timestamp,
                    frame_number=frame_number,
                    application_hint=application_hint,
                )

                candidates.append(candidate)

        finally:
            cap.release()

        elapsed = time.time() - start_time
        logger.info(
            f"Processed {len(candidates)} candidates in {elapsed:.2f}s "
            f"({len(candidates) / elapsed:.1f} candidates/sec)"
        )

        return candidates

    def extract_frame_at_click(
        self,
        video_path: Path,
        timestamp: float,
    ) -> np.ndarray | None:
        """Extract a single frame from video at a specific timestamp.

        Args:
            video_path: Path to the video file.
            timestamp: Time in seconds from start of video.

        Returns:
            Frame as numpy array (BGR), or None if extraction failed.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            return self._extract_frame(cap, frame_number)
        finally:
            cap.release()

    def _extract_frame(
        self,
        cap: cv2.VideoCapture,
        frame_number: int,
    ) -> np.ndarray | None:
        """Extract a specific frame from an open video capture."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            return None
        return frame

    def detect_boundary(
        self,
        frame: np.ndarray,
        click_location: tuple[int, int],
    ) -> InferenceResult:
        """Detect element boundary at a click point in a frame.

        Args:
            frame: Video frame (BGR format).
            click_location: (x, y) coordinates of the click.

        Returns:
            InferenceResult with detected boundary.
        """
        start_time = time.time()

        click_x, click_y = click_location
        height, width = frame.shape[:2]

        # Validate click location
        if not (0 <= click_x < width and 0 <= click_y < height):
            logger.warning(f"Click ({click_x}, {click_y}) outside frame bounds")
            return self._create_fallback_result(click_location, width, height)

        # Find boundaries
        candidates = self.boundary_finder.find_boundaries(
            frame, click_location, self.config.preferred_strategies
        )

        strategies_attempted: list[DetectionStrategy] = [
            s for s in self.config.preferred_strategies if s != DetectionStrategy.FIXED_SIZE
        ]

        if candidates:
            # Classify element types
            for candidate in candidates:
                if self.config.enable_element_classification:
                    element_type, type_confidence = (
                        self.context_analyzer.get_element_type_confidence(
                            frame, candidate, click_location
                        )
                    )
                    candidate.element_type = element_type
                    candidate.confidence = (candidate.confidence + type_confidence) / 2

            # Sort by confidence
            candidates.sort(key=lambda c: -c.confidence)

            primary = candidates[0]
            alternatives = candidates[1:5]

            processing_time = (time.time() - start_time) * 1000

            return InferenceResult(
                click_location=click_location,
                primary_bbox=primary,
                alternative_candidates=alternatives,
                image_width=width,
                image_height=height,
                strategies_attempted=strategies_attempted,
                processing_time_ms=processing_time,
                used_fallback=False,
            )

        # Fallback
        return self._create_fallback_result(click_location, width, height)

    def _create_fallback_result(
        self,
        click_location: tuple[int, int],
        img_width: int,
        img_height: int,
    ) -> InferenceResult:
        """Create a fallback result with fixed-size bounding box."""
        from .models import ElementType, InferredBoundingBox

        click_x, click_y = click_location
        size = self.config.fallback_box_size
        half_size = size // 2

        x = max(0, click_x - half_size)
        y = max(0, click_y - half_size)
        w = min(size, img_width - x)
        h = min(size, img_height - y)

        fallback_bbox = InferredBoundingBox(
            x=x,
            y=y,
            width=w,
            height=h,
            confidence=0.3,
            strategy_used=DetectionStrategy.FIXED_SIZE,
            element_type=ElementType.UNKNOWN,
            metadata={"fallback": True},
        )

        return InferenceResult(
            click_location=click_location,
            primary_bbox=fallback_bbox,
            image_width=img_width,
            image_height=img_height,
            strategies_attempted=[DetectionStrategy.FIXED_SIZE],
            processing_time_ms=0.0,
            used_fallback=True,
        )

    def _create_candidate(
        self,
        result: InferenceResult,
        session_id: str,
        click_x: int,
        click_y: int,
        click_button: str,
        timestamp: float,
        frame_number: int,
        application_hint: str | None,
    ) -> ClickTemplateCandidate:
        """Create a template candidate from an inference result."""
        primary = result.primary_bbox

        return ClickTemplateCandidate(
            id=str(uuid.uuid4()),
            session_id=session_id,
            click_x=click_x,
            click_y=click_y,
            click_button=click_button,
            timestamp=timestamp,
            frame_number=frame_number,
            primary_boundary=primary,
            alternative_boundaries=result.alternative_candidates,
            detection_strategies_used=result.strategies_attempted,
            pixel_data=primary.pixel_data,
            mask=primary.mask,
            application_hint=application_hint,
            created_at=datetime.now(),
            confidence_score=primary.confidence,
            element_type=primary.element_type.value,
        )

    def _load_click_events(self, events_file: Path) -> list[dict[str, Any]]:
        """Load click events from JSONL file.

        Args:
            events_file: Path to the events JSONL file.

        Returns:
            List of click event dictionaries.
        """
        if not events_file.exists():
            logger.error(f"Events file not found: {events_file}")
            return []

        click_events = []

        with open(events_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON line: {line[:50]}...")
                    continue

                # Filter for click events
                event_type = event.get("event_type", "")
                if event_type == "mouse_click" or event_type == "click":
                    click_events.append(event)

        return click_events

    def process_screenshot_with_clicks(
        self,
        screenshot: np.ndarray,
        click_locations: list[tuple[int, int]],
        session_id: str | None = None,
        application_hint: str | None = None,
    ) -> list[ClickTemplateCandidate]:
        """Process a single screenshot with multiple click locations.

        Useful for batch processing when clicks have already been
        collected separately from video.

        Args:
            screenshot: Screenshot image (BGR format).
            click_locations: List of (x, y) click coordinates.
            session_id: Optional session ID.
            application_hint: Optional application name.

        Returns:
            List of ClickTemplateCandidate objects.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        candidates = []

        for i, (click_x, click_y) in enumerate(click_locations):
            result = self.detect_boundary(screenshot, (click_x, click_y))

            candidate = self._create_candidate(
                result=result,
                session_id=session_id,
                click_x=click_x,
                click_y=click_y,
                click_button="left",
                timestamp=float(i),  # Use index as pseudo-timestamp
                frame_number=0,
                application_hint=application_hint,
            )

            candidates.append(candidate)

        return candidates
