"""State Machine Builder from approved template candidates.

Builds state machine configurations from user-approved template candidates.

In model-based GUI automation:
- A State is identified by the presence of specific visual elements (StateImages)
- Multiple states can be active simultaneously (parallel states)
- StateImages are visual patterns that identify when a state is active
- Transitions are actions that change which states are active

This module converts approved templates (from click capture) into:
1. StateImages - visual elements for state identification
2. States - collections of StateImages that identify application states
3. Transitions - inferred from click sequences (user should review)

The output can be exported to qontinui automation configuration format.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .approved_template import ApprovedTemplate
from .co_occurrence_analyzer import CoOccurrenceAnalyzer, CoOccurrenceResult
from .models import InferenceConfig
from .state_grouper import GroupingResult, StateGrouper


@dataclass
class StateImageDef:
    """State image definition for automation configuration.

    A StateImage is a visual element used to identify when a state is active.
    It represents a pattern to match on screen (e.g., button, icon, text).

    Attributes:
        id: Unique identifier
        name: Human-readable name
        pixel_data: Image data (BGR format)
        mask: Optional mask for non-rectangular matching
        x: X coordinate in source screenshot
        y: Y coordinate in source screenshot
        width: Width of the element
        height: Height of the element
        similarity_threshold: Threshold for template matching (0-1)
        source_template_id: ID of the source ApprovedTemplate
        metadata: Additional metadata
    """

    id: str
    name: str
    pixel_data: np.ndarray
    mask: np.ndarray | None = None
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    similarity_threshold: float = 0.85
    source_template_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "similarity_threshold": self.similarity_threshold,
            "source_template_id": self.source_template_id,
            "metadata": self.metadata,
            # Note: pixel_data and mask should be saved separately as images
        }


@dataclass
class StateDef:
    """State definition for automation configuration.

    A State is active when its identifying StateImages are detected.
    Multiple states can be active simultaneously (parallel states).

    Attributes:
        id: Unique state identifier
        name: Human-readable state name
        description: What this state represents
        state_images: List of StateImages that identify this state
        is_initial: Whether this is an initial state
        is_terminal: Whether this is a terminal state
        confidence: Confidence score for this state definition
        metadata: Additional metadata
    """

    id: str
    name: str
    description: str = ""
    state_images: list[StateImageDef] = field(default_factory=list)
    is_initial: bool = False
    is_terminal: bool = False
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def state_image_count(self) -> int:
        """Number of identifying StateImages."""
        return len(self.state_images)

    @property
    def state_image_ids(self) -> list[str]:
        """IDs of all StateImages."""
        return [si.id for si in self.state_images]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "state_images": [si.to_dict() for si in self.state_images],
            "is_initial": self.is_initial,
            "is_terminal": self.is_terminal,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class TransitionDef:
    """Transition definition for automation configuration.

    A transition represents an action that may change active states.
    Inferred from click sequences but should be reviewed by user.

    Attributes:
        id: Unique transition identifier
        name: Human-readable name
        source_state_id: ID of the source state
        target_state_ids: IDs of potential target states
        action_type: Type of action (click, key_press, etc.)
        action_location: Click location for click actions
        recognition_image_id: StateImage to recognize before triggering
        workflow_id: Optional workflow to execute
        confidence: Confidence score for this transition
        metadata: Additional metadata
    """

    id: str
    name: str = ""
    source_state_id: str = ""
    target_state_ids: list[str] = field(default_factory=list)
    action_type: str = "click"
    action_location: tuple[int, int] | None = None
    recognition_image_id: str | None = None
    workflow_id: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "source_state_id": self.source_state_id,
            "target_state_ids": self.target_state_ids,
            "action_type": self.action_type,
            "action_location": self.action_location,
            "recognition_image_id": self.recognition_image_id,
            "workflow_id": self.workflow_id,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class StateMachineResult:
    """Result of state machine generation.

    Contains the complete state machine definition ready for export.

    Attributes:
        session_id: ID of the source capture session
        states: List of state definitions
        transitions: List of transition definitions (user should review)
        state_images: All StateImages (referenced by states)
        grouping_result: Result from state grouping phase
        processing_time_ms: Total processing time
        config_used: Configuration used for generation
        metadata: Additional metadata
    """

    session_id: str
    states: list[StateDef] = field(default_factory=list)
    transitions: list[TransitionDef] = field(default_factory=list)
    state_images: list[StateImageDef] = field(default_factory=list)
    grouping_result: GroupingResult | None = None
    processing_time_ms: float = 0.0
    config_used: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def state_count(self) -> int:
        """Number of states."""
        return len(self.states)

    @property
    def transition_count(self) -> int:
        """Number of transitions."""
        return len(self.transitions)

    @property
    def state_image_count(self) -> int:
        """Total number of StateImages."""
        return len(self.state_images)

    def get_state(self, state_id: str) -> StateDef | None:
        """Get state by ID."""
        for state in self.states:
            if state.id == state_id:
                return state
        return None

    def get_state_image(self, image_id: str) -> StateImageDef | None:
        """Get StateImage by ID."""
        for si in self.state_images:
            if si.id == image_id:
                return si
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "states": [s.to_dict() for s in self.states],
            "transitions": [t.to_dict() for t in self.transitions],
            "state_images": [si.to_dict() for si in self.state_images],
            "processing_time_ms": self.processing_time_ms,
            "state_count": self.state_count,
            "transition_count": self.transition_count,
            "state_image_count": self.state_image_count,
            "config_used": self.config_used,
            "metadata": self.metadata,
        }

    def to_automation_config(self) -> dict[str, Any]:
        """Convert to qontinui automation configuration format.

        Returns a dictionary compatible with qontinui config schema.
        """
        return {
            "version": "2.0.0",
            "metadata": {
                "generated_from": "click_to_state_machine",
                "session_id": self.session_id,
                **self.metadata,
            },
            "images": [
                {
                    "id": si.id,
                    "name": si.name,
                    "width": si.width,
                    "height": si.height,
                    "x": si.x,
                    "y": si.y,
                    "similarityThreshold": si.similarity_threshold,
                }
                for si in self.state_images
            ],
            "states": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "isInitial": s.is_initial,
                    "stateImages": [{"id": si.id, "name": si.name} for si in s.state_images],
                }
                for s in self.states
            ],
            "transitions": [
                {
                    "id": t.id,
                    "name": t.name,
                    "sourceStateId": t.source_state_id,
                    "targetStateIds": t.target_state_ids,
                    "workflowId": t.workflow_id,
                }
                for t in self.transitions
            ],
        }


class ClickToStateMachineBuilder:
    """Builds state machine configurations from approved template candidates.

    This builder converts approved templates (from click capture) into a
    state machine configuration that can be used for automation.

    Key concepts:
    - Each approved template becomes a StateImage (visual identifier)
    - States are defined by user groupings or inferred from co-occurrence
    - Transitions are suggested based on click sequences (user review needed)

    IMPORTANT: The builder does NOT automatically detect "states" from frames.
    States must be defined by the user (via state_hint field or explicit
    assignments) or inferred heuristically from template co-occurrence.

    Example:
        >>> builder = ClickToStateMachineBuilder()
        >>> result = builder.build_from_templates(
        ...     templates=approved_templates,
        ...     grouping_method="state_hints",
        ... )
        >>> # Export to automation config
        >>> config = result.to_automation_config()
    """

    def __init__(
        self,
        config: InferenceConfig | None = None,
        default_similarity_threshold: float = 0.85,
    ) -> None:
        """Initialize the state machine builder.

        Args:
            config: Optional inference configuration
            default_similarity_threshold: Default threshold for StateImages
        """
        self.config = config or InferenceConfig()
        self.default_similarity_threshold = default_similarity_threshold
        self._state_grouper = StateGrouper()

    def build_from_templates(
        self,
        templates: list[ApprovedTemplate],
        grouping_method: str = "state_hints",
        state_assignments: dict[str, list[str]] | None = None,
        session_id: str = "",
        video_path: Path | None = None,
        co_occurrence_data: CoOccurrenceResult | None = None,
        co_occurrence_sample_interval: int = 30,
    ) -> StateMachineResult:
        """Build state machine from approved templates.

        Args:
            templates: List of user-approved templates
            grouping_method: How to group templates into states:
                - "state_hints": Use template.state_hint field
                - "user_assignments": Use explicit state_assignments dict
                - "co_occurrence": Group by templates appearing together
                - "single_state": All templates in one state
                - "one_per_template": Each template is its own state
            state_assignments: For "user_assignments" method, dict mapping
                state_name to list of template IDs
            session_id: Session identifier
            video_path: Optional path to video for extracting pixel data
                and co-occurrence analysis
            co_occurrence_data: Pre-computed co-occurrence analysis result.
                If provided and grouping_method is "co_occurrence", this
                data will be used instead of running the analyzer.
            co_occurrence_sample_interval: Frames to skip between samples
                when running co-occurrence analysis (default: 30)

        Returns:
            StateMachineResult with states, transitions, and StateImages
        """
        import time

        start_time = time.time()

        if not templates:
            return StateMachineResult(
                session_id=session_id,
                metadata={"warning": "No templates provided"},
            )

        # Step 1: Extract pixel data if needed
        templates = self._ensure_pixel_data(templates, video_path)

        # Step 2: Convert templates to StateImages
        state_images = self._create_state_images(templates)

        # Step 3: Run co-occurrence analysis if needed
        if grouping_method == "co_occurrence" and co_occurrence_data is None:
            if video_path and video_path.exists():
                co_occurrence_data = self._analyze_co_occurrence(
                    video_path, templates, co_occurrence_sample_interval
                )

        # Step 4: Group templates into states
        grouping_result = self._group_templates(
            templates, grouping_method, state_assignments, co_occurrence_data
        )

        # Step 5: Create state definitions
        states = self._create_states(grouping_result, state_images)

        # Step 6: Infer transitions from click sequence
        transitions = self._infer_transitions(templates, grouping_result)

        processing_time = (time.time() - start_time) * 1000

        return StateMachineResult(
            session_id=session_id,
            states=states,
            transitions=transitions,
            state_images=state_images,
            grouping_result=grouping_result,
            processing_time_ms=processing_time,
            config_used={
                "grouping_method": grouping_method,
                "default_similarity_threshold": self.default_similarity_threshold,
                "co_occurrence_sample_interval": co_occurrence_sample_interval,
            },
            metadata={
                "co_occurrence_analyzed": co_occurrence_data is not None,
                "co_occurrence_frames": (
                    co_occurrence_data.frames_analyzed if co_occurrence_data else 0
                ),
            },
        )

    def _ensure_pixel_data(
        self,
        templates: list[ApprovedTemplate],
        video_path: Path | None,
    ) -> list[ApprovedTemplate]:
        """Ensure all templates have pixel data.

        If pixel_data is missing, try to extract from video.

        Args:
            templates: List of templates
            video_path: Optional path to source video

        Returns:
            Templates with pixel data populated where possible
        """
        if video_path is None or not video_path.exists():
            return templates

        # Find templates missing pixel data
        missing = [t for t in templates if t.pixel_data is None]
        if not missing:
            return templates

        # Extract frames from video
        frames_needed = sorted({t.frame_number for t in missing})

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return templates

        try:
            frame_data: dict[int, np.ndarray] = {}
            for frame_num in frames_needed:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if ret:
                    frame_data[frame_num] = frame

            # Populate pixel data
            for template in missing:
                if template.frame_number in frame_data:
                    frame = frame_data[template.frame_number]
                    bbox = template.boundary
                    template.pixel_data = frame[
                        bbox.y : bbox.y + bbox.height,
                        bbox.x : bbox.x + bbox.width,
                    ].copy()
        finally:
            cap.release()

        return templates

    def _create_state_images(self, templates: list[ApprovedTemplate]) -> list[StateImageDef]:
        """Convert approved templates to StateImage definitions.

        Args:
            templates: List of approved templates

        Returns:
            List of StateImageDef objects
        """
        state_images = []

        for template in templates:
            pixel_data = template.pixel_data
            if pixel_data is None:
                # Create placeholder if no pixel data
                pixel_data = np.zeros((10, 10, 3), dtype=np.uint8)

            state_image = StateImageDef(
                id=f"img_{template.id}",
                name=template.name or f"Element_{template.id[:8]}",
                pixel_data=pixel_data,
                mask=template.mask,
                x=template.boundary.x,
                y=template.boundary.y,
                width=template.boundary.width,
                height=template.boundary.height,
                similarity_threshold=self.default_similarity_threshold,
                source_template_id=template.id,
                metadata={
                    "element_type": template.element_type,
                    "click_location": (template.click_x, template.click_y),
                    "confidence": template.confidence,
                },
            )
            state_images.append(state_image)

        return state_images

    def _analyze_co_occurrence(
        self,
        video_path: Path,
        templates: list[ApprovedTemplate],
        sample_interval: int = 30,
    ) -> CoOccurrenceResult:
        """Analyze video to determine template co-occurrence.

        This runs template matching across video frames to determine which
        templates appear together (co-occur). The result is used by the
        StateGrouper's co_occurrence method.

        Args:
            video_path: Path to the video file
            templates: List of approved templates with pixel data
            sample_interval: Check every N frames (default: 30)

        Returns:
            CoOccurrenceResult with frame-to-template mapping
        """
        analyzer = CoOccurrenceAnalyzer(similarity_threshold=self.default_similarity_threshold)
        return analyzer.analyze_video(
            video_path=video_path,
            templates=templates,
            sample_interval=sample_interval,
        )

    def _group_templates(
        self,
        templates: list[ApprovedTemplate],
        method: str,
        assignments: dict[str, list[str]] | None,
        co_occurrence_data: CoOccurrenceResult | None = None,
    ) -> GroupingResult:
        """Group templates into states using specified method.

        Args:
            templates: List of approved templates
            method: Grouping method name
            assignments: Explicit assignments for "user_assignments" method
            co_occurrence_data: Pre-computed co-occurrence analysis result

        Returns:
            GroupingResult with state groupings
        """
        if method == "user_assignments" and assignments:
            return self._state_grouper.group_by_user_assignments(templates, assignments)
        elif method == "state_hints":
            return self._state_grouper.group_by_state_hints(templates)
        elif method == "co_occurrence":
            # Pass the frame-to-template mapping from co-occurrence analysis
            frames = co_occurrence_data.frame_template_map if co_occurrence_data else None
            return self._state_grouper.group_by_co_occurrence(templates, frames)
        elif method == "single_state":
            return self._state_grouper.create_single_state(templates)
        elif method == "one_per_template":
            return self._state_grouper.create_one_state_per_template(templates)
        else:
            # Default to state_hints
            return self._state_grouper.group_by_state_hints(templates)

    def _create_states(
        self,
        grouping_result: GroupingResult,
        all_state_images: list[StateImageDef],
    ) -> list[StateDef]:
        """Create state definitions from grouping result.

        Args:
            grouping_result: Result from state grouping
            all_state_images: All StateImage definitions

        Returns:
            List of StateDef objects
        """
        # Create lookup for StateImages
        image_lookup = {si.source_template_id: si for si in all_state_images}

        states = []
        for group in grouping_result.states:
            # Get StateImages for this state's templates
            state_images = []
            for template in group.state_images:
                if template.id in image_lookup:
                    state_images.append(image_lookup[template.id])

            state = StateDef(
                id=group.state_id,
                name=group.state_name,
                description=group.description,
                state_images=state_images,
                is_initial=group.is_initial,
                confidence=group.confidence,
                metadata=group.metadata,
            )
            states.append(state)

        return states

    def _infer_transitions(
        self,
        templates: list[ApprovedTemplate],
        grouping_result: GroupingResult,
    ) -> list[TransitionDef]:
        """Infer transitions from click sequence.

        This is a heuristic - assumes consecutive clicks that change
        active states represent transitions. User should review.

        Args:
            templates: List of approved templates (ordered by timestamp)
            grouping_result: Result from state grouping

        Returns:
            List of inferred TransitionDef objects
        """
        if not templates or not grouping_result.states:
            return []

        # Build template-to-state mapping
        template_to_state: dict[str, str] = {}
        for group in grouping_result.states:
            for template in group.state_images:
                template_to_state[template.id] = group.state_id

        # Sort templates by timestamp
        sorted_templates = sorted(templates, key=lambda t: t.click_timestamp)

        transitions: list[TransitionDef] = []
        seen_transitions: set[tuple[str, str]] = set()

        for i in range(len(sorted_templates) - 1):
            current = sorted_templates[i]
            next_template = sorted_templates[i + 1]

            current_state = template_to_state.get(current.id)
            next_state = template_to_state.get(next_template.id)

            if not current_state or not next_state:
                continue

            # Skip if same state (not a state-changing transition)
            if current_state == next_state:
                continue

            # Skip duplicate transitions
            trans_key = (current_state, next_state)
            if trans_key in seen_transitions:
                continue
            seen_transitions.add(trans_key)

            transition = TransitionDef(
                id=f"trans_{len(transitions)}",
                name=f"To {next_state}",
                source_state_id=current_state,
                target_state_ids=[next_state],
                action_type="click",
                action_location=(current.click_x, current.click_y),
                recognition_image_id=f"img_{current.id}",
                confidence=0.5,  # Low confidence - user should review
                metadata={
                    "inferred": True,
                    "source_template_id": current.id,
                    "target_template_id": next_template.id,
                    "note": "Inferred from click sequence - please review",
                },
            )
            transitions.append(transition)

        return transitions

    def build_from_video_and_templates(
        self,
        video_path: Path,
        events_file: Path,
        templates: list[ApprovedTemplate],
        grouping_method: str = "state_hints",
        state_assignments: dict[str, list[str]] | None = None,
    ) -> StateMachineResult:
        """Build state machine from video, events, and approved templates.

        Convenience method that extracts pixel data from video.

        Args:
            video_path: Path to captured video
            events_file: Path to JSONL events file (for reference)
            templates: List of user-approved templates
            grouping_method: How to group templates into states
            state_assignments: Explicit state assignments (optional)

        Returns:
            StateMachineResult with states, transitions, and StateImages
        """
        session_id = video_path.stem

        return self.build_from_templates(
            templates=templates,
            grouping_method=grouping_method,
            state_assignments=state_assignments,
            session_id=session_id,
            video_path=video_path,
        )
