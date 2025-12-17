"""Transition Analysis Service for qontinui-runner.

This service identifies state transitions from video, correlating visual changes
with input events. It determines both the outgoing action (which StateImage was
clicked) and incoming recognition (which StateImages appeared in the new state).

Key Concepts:
    - Outgoing (Action): The GUI action that triggers the transition (e.g., click on StateImage)
    - Incoming (Recognition): The appearance of StateImages in the destination state

The service processes captured video frames, detected states, and input events to
automatically build a complete transition graph for GUI automation.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any

from qontinui.discovery.state_analysis.models import (
    DetectedState,
    Frame,
    InputEvent,
    StateChangePoint,
    Transition,
)

logger = logging.getLogger(__name__)


@dataclass
class CaptureSession:
    """Represents a complete capture session with frames, states, and events.

    Attributes:
        frames: List of captured video frames
        states: List of detected states
        events: List of input events (clicks, keypresses, etc.)
        fps: Frames per second of the video
        metadata: Additional session metadata
    """

    frames: list[Frame]
    states: list[DetectedState]
    events: list[InputEvent]
    fps: int = 30
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TransitionAnalyzer:
    """Analyzes video captures to identify state transitions.

    This service correlates state changes with input events to build a complete
    transition graph. For each transition, it identifies:
    1. The StateImage that was clicked (outgoing action)
    2. The StateImages that appeared in the new state (incoming recognition)
    3. Timing and visual change metrics
    """

    def __init__(
        self,
        event_correlation_window_ms: int = 1000,
        click_proximity_threshold: int = 50,
        min_visual_change_score: float = 0.1,
    ):
        """Initialize the transition analyzer.

        Args:
            event_correlation_window_ms: Maximum time window (ms) to correlate events with state changes
            click_proximity_threshold: Maximum distance (pixels) to match click with StateImage
            min_visual_change_score: Minimum visual change score to consider a state change
        """
        self.event_correlation_window_ms = event_correlation_window_ms
        self.click_proximity_threshold = click_proximity_threshold
        self.min_visual_change_score = min_visual_change_score

    def analyze_transitions(
        self,
        states: list[DetectedState],
        events: list[InputEvent],
        frames: list[Frame],
    ) -> list[Transition]:
        """Analyze transitions from detected states and input events.

        This is the main entry point for transition analysis. It:
        1. Identifies state change points (when states appear/disappear)
        2. Correlates state changes with input events
        3. Identifies action targets (which StateImage was clicked)
        4. Identifies recognition images (which StateImages appeared in new state)
        5. Builds complete Transition objects

        Args:
            states: List of detected states from video analysis
            events: List of input events (clicks, keypresses)
            frames: List of video frames with timing info

        Returns:
            List of identified transitions with action and recognition data
        """
        logger.info(
            f"Analyzing transitions: {len(states)} states, {len(events)} events, {len(frames)} frames"
        )

        # Step 1: Find all state change points
        change_points = self.find_state_changes(frames, states)
        logger.info(f"Found {len(change_points)} state change points")

        # Step 2: Correlate each state change with input events
        transitions = []
        for i, change_point in enumerate(change_points):
            if not change_point.is_state_change:
                continue

            # Find the event that triggered this state change
            trigger_event = self.correlate_with_events(change_point, events)
            if not trigger_event:
                logger.debug(
                    f"No trigger event found for state change at frame {change_point.frame_number}"
                )
                continue

            # Determine which states were involved
            from_state_id = (
                list(change_point.states_disappeared)[0]
                if change_point.states_disappeared
                else None
            )
            to_state_id = (
                list(change_point.states_appeared)[0]
                if change_point.states_appeared
                else None
            )

            if not from_state_id or not to_state_id:
                logger.debug(
                    f"Incomplete state transition at frame {change_point.frame_number}"
                )
                continue

            # Get the actual state objects
            from_state = next((s for s in states if s.name == from_state_id), None)
            to_state = next((s for s in states if s.name == to_state_id), None)

            if not from_state or not to_state:
                logger.warning(
                    f"Could not find state objects for transition: {from_state_id} -> {to_state_id}"
                )
                continue

            # Step 3: Identify action target (which StateImage was clicked)
            frame = (
                frames[change_point.frame_number]
                if change_point.frame_number < len(frames)
                else None
            )
            action_target_id = self.identify_action_target(
                trigger_event, frame, from_state
            )

            # Step 4: Identify recognition images (which StateImages appeared in new state)
            recognition_ids = self.identify_recognition_images(to_state)

            # Step 5: Calculate visual change metrics
            visual_change_score = self._calculate_visual_change_score(
                change_point, frames
            )
            optical_flow = self._estimate_optical_flow_magnitude(change_point, frames)

            # Step 6: Build transition object
            transition_id = f"t_{i}_{from_state_id}_to_{to_state_id}"

            transition = Transition(
                id=transition_id,
                source_states=[from_state_id],
                target_states=[to_state_id],
                states_appeared=sorted(change_point.states_appeared),
                states_disappeared=sorted(change_point.states_disappeared),
                # Outgoing (Action)
                action_type=trigger_event.event_type,
                action_target=action_target_id,
                action_location=(
                    (trigger_event.x, trigger_event.y)
                    if trigger_event.x is not None and trigger_event.y is not None
                    else None
                ),
                action_data={
                    "button": trigger_event.button,
                    "event_index": events.index(trigger_event),
                },
                # Incoming (Recognition)
                recognition_images=recognition_ids,
                recognition_confidence=0.85,
                # Timing
                timestamp=change_point.timestamp,
                frame_before=change_point.frame_number,
                frame_after=change_point.frame_number + 1,
                duration_ms=self._estimate_transition_duration(change_point, frames),
                # Metadata
                metadata={
                    "auto_generated": True,
                    "verified": False,
                    "optical_flow_magnitude": optical_flow,
                    "visual_change_score": visual_change_score,
                },
            )

            transitions.append(transition)
            logger.info(
                f"Created transition: {transition_id} ({from_state_id} -> {to_state_id})"
            )

        logger.info(f"Generated {len(transitions)} transitions")
        return transitions

    def identify_action_target(
        self,
        event: InputEvent,
        frame: Frame | None,
        state: DetectedState,
    ) -> str | None:
        """Determine which StateImage was clicked/acted upon.

        This method analyzes the input event (typically a click) and the active
        state to determine which StateImage was the target of the action.

        Args:
            event: The input event that triggered the transition
            frame: The video frame at the time of the event (optional)
            state: The source state that was active when the event occurred

        Returns:
            StateImage ID if a target was identified, None otherwise
        """
        # Only process click events with coordinates
        if event.event_type != "mouse_click" or event.x is None or event.y is None:
            return None

        click_x, click_y = event.x, event.y
        closest_image_id = None
        closest_distance = float("inf")

        # Find the StateImage closest to the click location
        for state_image in state.state_images:
            # Get the center of the StateImage bounding box
            bbox_x, bbox_y, bbox_w, bbox_h = state_image.bbox
            center_x = bbox_x + bbox_w // 2
            center_y = bbox_y + bbox_h // 2

            # Calculate distance from click to StateImage center
            distance = math.sqrt((click_x - center_x) ** 2 + (click_y - center_y) ** 2)

            # Check if click is within the bounding box or close enough
            in_bbox = (
                bbox_x <= click_x <= bbox_x + bbox_w
                and bbox_y <= click_y <= bbox_y + bbox_h
            )

            if in_bbox or distance < self.click_proximity_threshold:
                if distance < closest_distance:
                    closest_distance = distance
                    closest_image_id = state_image.name

        if closest_image_id:
            logger.debug(
                f"Identified action target: {closest_image_id} (distance: {closest_distance:.1f}px)"
            )
        else:
            logger.debug(
                f"No action target found within {self.click_proximity_threshold}px of click at ({click_x}, {click_y})"
            )

        return closest_image_id

    def identify_recognition_images(self, state: DetectedState) -> list[str]:
        """Identify StateImages that uniquely identify this state.

        This method selects the StateImages that should be used to recognize
        when the automation has successfully transitioned to this state.

        Strategy:
        1. Prefer StateImages with higher confidence/similarity
        2. Prefer StateImages that are unique to this state
        3. Return multiple images for robust recognition

        Args:
            state: The destination state to identify

        Returns:
            List of StateImage IDs for recognition
        """
        if not state.state_images:
            logger.warning(f"State {state.name} has no StateImages for recognition")
            return []

        # Sort StateImages by similarity threshold (higher is more reliable)
        sorted_images = sorted(
            state.state_images,
            key=lambda img: img.similarity_threshold,
            reverse=True,
        )

        # Select top N images for recognition (up to 3 for redundancy)
        recognition_count = min(3, len(sorted_images))
        recognition_images = [img.name for img in sorted_images[:recognition_count]]

        logger.debug(
            f"Selected {len(recognition_images)} recognition images for state {state.name}: {recognition_images}"
        )

        return recognition_images

    def find_state_changes(
        self, frames: list[Frame], states: list[DetectedState]
    ) -> list[StateChangePoint]:
        """Find points where active states change.

        This method analyzes the frame sequence and state data to identify
        when the GUI transitions from one state to another.

        Args:
            frames: List of video frames with timing info
            states: List of detected states with frame ranges

        Returns:
            List of StateChangePoint objects marking state transitions
        """
        if not frames or not states:
            return []

        change_points = []

        # Build a mapping of frame index to active state IDs
        frame_to_states: dict[int, set[str]] = {}
        for frame in frames:
            frame_to_states[frame.frame_index] = set()

        # Populate the mapping based on state frame ranges
        for state in states:
            for frame_idx in state.frame_indices:
                if frame_idx in frame_to_states:
                    frame_to_states[frame_idx].add(state.name)

        # Find changes by comparing consecutive frames
        sorted_frame_indices = sorted(frame_to_states.keys())
        for i in range(1, len(sorted_frame_indices)):
            prev_frame_idx = sorted_frame_indices[i - 1]
            curr_frame_idx = sorted_frame_indices[i]

            prev_states = frame_to_states[prev_frame_idx]
            curr_states = frame_to_states[curr_frame_idx]

            # Detect state changes
            states_appeared = curr_states - prev_states
            states_disappeared = prev_states - curr_states

            if states_appeared or states_disappeared:
                # Find the corresponding frame for timestamp
                frame = next((f for f in frames if f.frame_index == curr_frame_idx), None)  # type: ignore[assignment]
                timestamp = frame.timestamp if frame else 0.0

                change_point = StateChangePoint(
                    frame_number=curr_frame_idx,
                    timestamp=timestamp,
                    states_before=prev_states.copy(),
                    states_after=curr_states.copy(),
                    states_appeared=states_appeared,
                    states_disappeared=states_disappeared,
                )

                change_points.append(change_point)
                logger.debug(
                    f"State change at frame {curr_frame_idx}: "
                    f"appeared={states_appeared}, disappeared={states_disappeared}"
                )

        return change_points

    def correlate_with_events(
        self, change_point: StateChangePoint, events: list[InputEvent]
    ) -> InputEvent | None:
        """Find the input event that likely caused this state change.

        This method searches for an input event that occurred shortly before
        the state change, within the correlation window.

        Args:
            change_point: The state change to correlate
            events: List of input events to search

        Returns:
            The most likely trigger event, or None if no correlation found
        """
        # Convert correlation window to seconds
        window_seconds = self.event_correlation_window_ms / 1000.0

        # Find events that occurred before the state change, within the window
        candidate_events = []
        for event in events:
            time_diff = change_point.timestamp - event.timestamp

            # Event must be before the change, but within the window
            if 0 <= time_diff <= window_seconds:
                candidate_events.append((event, time_diff))

        if not candidate_events:
            return None

        # Return the most recent event (smallest time_diff)
        candidate_events.sort(key=lambda x: x[1])
        closest_event = candidate_events[0][0]

        logger.debug(
            f"Correlated event {closest_event.event_type} at {closest_event.timestamp:.3f}s "
            f"with state change at {change_point.timestamp:.3f}s "
            f"(diff: {candidate_events[0][1]*1000:.1f}ms)"
        )

        return closest_event

    def _calculate_visual_change_score(
        self, change_point: StateChangePoint, frames: list[Frame]
    ) -> float:
        """Calculate a visual change score for the transition.

        This is a placeholder that returns a score based on the magnitude
        of the state change. In a full implementation, this would analyze
        the actual frame images to compute visual difference.

        Args:
            change_point: The state change point
            frames: List of frames for visual analysis

        Returns:
            Visual change score between 0.0 and 1.0
        """
        # Simple heuristic: normalize based on number of states changed
        magnitude = change_point.change_magnitude
        score = min(1.0, magnitude / 5.0)
        return score

    def _estimate_optical_flow_magnitude(
        self, change_point: StateChangePoint, frames: list[Frame]
    ) -> float:
        """Estimate the magnitude of optical flow during the transition.

        This is a placeholder. In a full implementation, this would use
        OpenCV to calculate actual optical flow between frames.

        Args:
            change_point: The state change point
            frames: List of frames for optical flow analysis

        Returns:
            Estimated optical flow magnitude
        """
        # Placeholder: return a value based on visual change
        return self._calculate_visual_change_score(change_point, frames) * 10.0

    def _estimate_transition_duration(
        self, change_point: StateChangePoint, frames: list[Frame]
    ) -> int:
        """Estimate the duration of the transition animation.

        This is a placeholder. In a full implementation, this would analyze
        multiple frames to detect when the visual change stabilizes.

        Args:
            change_point: The state change point
            frames: List of frames for duration analysis

        Returns:
            Estimated transition duration in milliseconds
        """
        # Placeholder: assume 200ms for typical UI transitions
        return 200


class AutoTransitionBuilder:
    """Automatically builds transitions from captured video sessions.

    This service provides a high-level interface for extracting transitions
    from a complete capture session, including multi-state transitions and
    automatic naming.
    """

    def __init__(self, analyzer: TransitionAnalyzer | None = None):
        """Initialize the auto transition builder.

        Args:
            analyzer: TransitionAnalyzer instance (creates default if None)
        """
        self.analyzer = analyzer or TransitionAnalyzer()

    def build_from_capture(self, session: CaptureSession) -> list[Transition]:
        """Automatically create transitions from captured video.

        This method processes a complete capture session to extract all
        transitions, handling:
        1. Multi-state transitions (states appearing/disappearing)
        2. Automatic action target identification
        3. Automatic recognition image selection
        4. Transition naming and metadata

        Args:
            session: Complete capture session with frames, states, and events

        Returns:
            List of automatically generated transitions
        """
        logger.info(
            f"Building transitions from capture session: "
            f"{len(session.states)} states, {len(session.events)} events"
        )

        # Use the analyzer to extract transitions
        transitions = self.analyzer.analyze_transitions(
            states=session.states,
            events=session.events,
            frames=session.frames,
        )

        # Enhance transitions with additional metadata
        for transition in transitions:
            # Add session metadata
            transition.metadata["session_fps"] = session.fps
            transition.metadata["session_metadata"] = session.metadata
            # Store suggested name in metadata
            transition.metadata["suggested_name"] = self.suggest_transition_name(
                transition
            )

        logger.info(f"Built {len(transitions)} transitions from capture session")
        return transitions

    def suggest_transition_name(self, transition: Transition) -> str:
        """Generate a human-readable transition name.

        Creates names like:
        - "Login Screen → Dashboard (click login_button)"
        - "Dashboard → Settings"
        - "Menu → Submenu (click options)"

        Args:
            transition: The transition to name

        Returns:
            Suggested human-readable name
        """
        from_name = transition.get_primary_source_state() or "Unknown"
        to_name = transition.get_primary_target_state() or "Unknown"

        # Add action target if available
        if transition.action_target:
            action_part = f" (click {transition.action_target})"
        elif transition.action_type == "key_press":
            action_part = " (keypress)"
        else:
            action_part = ""

        return f"{from_name} → {to_name}{action_part}"
