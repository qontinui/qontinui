"""State Grouper for organizing template candidates into states.

In model-based GUI automation, a State is identified by the presence of
specific visual elements (StateImages). This module helps organize approved
template candidates into states based on:

1. User-provided groupings (explicit state assignments)
2. Co-occurrence analysis (templates that appear together likely belong
   to the same state)

IMPORTANT: Multiple states can be active simultaneously (e.g., main menu +
modal dialog). This grouper helps identify which StateImages define each
state, not which "frame cluster" is which state.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..shared import CoOccurrenceGrouper, invert_frame_template_map
from .approved_template import ApprovedTemplate


@dataclass
class StateGroup:
    """A group of StateImages that identify a particular state.

    In model-based automation, a State is active when its identifying
    StateImages are detected. This class represents a state definition
    with its associated StateImages (derived from approved templates).

    Attributes:
        state_id: Unique identifier for this state
        state_name: Human-readable name for the state
        state_images: List of approved templates that identify this state
        description: Description of what this state represents
        is_initial: Whether this is an initial state
        confidence: Confidence score for this grouping
        metadata: Additional metadata
    """

    state_id: str
    state_name: str
    state_images: list[ApprovedTemplate] = field(default_factory=list)
    description: str = ""
    is_initial: bool = False
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def state_image_count(self) -> int:
        """Number of StateImages in this state."""
        return len(self.state_images)

    @property
    def state_image_ids(self) -> list[str]:
        """IDs of all StateImages."""
        return [t.id for t in self.state_images]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "state_id": self.state_id,
            "state_name": self.state_name,
            "state_images": [t.to_dict() for t in self.state_images],
            "description": self.description,
            "is_initial": self.is_initial,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class GroupingResult:
    """Result of state grouping operation.

    Attributes:
        states: List of defined states with their StateImages
        ungrouped_templates: Templates not assigned to any state
        co_occurrence_matrix: Matrix showing which templates appear together
        grouping_method: Method used for grouping
        processing_time_ms: Time taken to process
    """

    states: list[StateGroup] = field(default_factory=list)
    ungrouped_templates: list[ApprovedTemplate] = field(default_factory=list)
    co_occurrence_matrix: np.ndarray | None = None
    grouping_method: str = ""
    processing_time_ms: float = 0.0

    @property
    def state_count(self) -> int:
        """Number of states."""
        return len(self.states)

    @property
    def total_state_images(self) -> int:
        """Total StateImages across all states."""
        return sum(s.state_image_count for s in self.states) + len(self.ungrouped_templates)

    def get_state(self, state_id: str) -> StateGroup | None:
        """Get state by ID."""
        for state in self.states:
            if state.state_id == state_id:
                return state
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "states": [s.to_dict() for s in self.states],
            "ungrouped_templates": [t.to_dict() for t in self.ungrouped_templates],
            "grouping_method": self.grouping_method,
            "processing_time_ms": self.processing_time_ms,
            "state_count": self.state_count,
            "total_state_images": self.total_state_images,
        }


class StateGrouper:
    """Organizes approved templates into state definitions.

    In model-based GUI automation, states are identified by the presence
    of specific visual elements (StateImages). This class helps organize
    approved template candidates into states.

    Two main approaches:
    1. User-defined groupings: User explicitly assigns templates to states
    2. Co-occurrence analysis: Templates appearing together are grouped

    IMPORTANT: Multiple states can be active simultaneously. A template
    belonging to State A doesn't mean it can't also appear when State B
    is active. The grouping determines which StateImages IDENTIFY each state.

    Example:
        >>> grouper = StateGrouper()
        >>> # User explicitly defines state groupings
        >>> result = grouper.group_by_user_assignments(
        ...     templates=approved_templates,
        ...     assignments={"login_state": ["img1", "img2"], "main_menu": ["img3"]}
        ... )
        >>> for state in result.states:
        ...     print(f"State {state.state_name}: {state.state_image_count} images")
    """

    def __init__(
        self,
        co_occurrence_threshold: float = 0.8,
    ) -> None:
        """Initialize the state grouper.

        Args:
            co_occurrence_threshold: Threshold for considering templates
                as belonging to the same state (0-1, higher = stricter)
        """
        self.co_occurrence_threshold = co_occurrence_threshold

    def group_by_user_assignments(
        self,
        templates: list[ApprovedTemplate],
        assignments: dict[str, list[str]],
    ) -> GroupingResult:
        """Group templates based on explicit user assignments.

        This is the primary method - users explicitly define which
        templates belong to which states.

        Args:
            templates: List of approved templates
            assignments: Dict mapping state_name to list of template IDs

        Returns:
            GroupingResult with user-defined states
        """
        import time

        start_time = time.time()

        # Create template lookup
        template_lookup = {t.id: t for t in templates}
        assigned_ids: set[str] = set()

        states = []
        for i, (state_name, template_ids) in enumerate(assignments.items()):
            state_images = []
            for tid in template_ids:
                if tid in template_lookup:
                    state_images.append(template_lookup[tid])
                    assigned_ids.add(tid)

            if state_images:
                states.append(
                    StateGroup(
                        state_id=f"state_{i}",
                        state_name=state_name,
                        state_images=state_images,
                        is_initial=(i == 0),  # First state is initial
                        confidence=1.0,  # User-defined = high confidence
                        metadata={"source": "user_assignment"},
                    )
                )

        # Collect ungrouped templates
        ungrouped = [t for t in templates if t.id not in assigned_ids]

        processing_time = (time.time() - start_time) * 1000

        return GroupingResult(
            states=states,
            ungrouped_templates=ungrouped,
            grouping_method="user_assignment",
            processing_time_ms=processing_time,
        )

    def group_by_state_hints(
        self,
        templates: list[ApprovedTemplate],
    ) -> GroupingResult:
        """Group templates by their state_hint field.

        Uses the state_hint field on ApprovedTemplate that users set
        during the review process.

        Args:
            templates: List of approved templates with state_hint set

        Returns:
            GroupingResult with hint-based states
        """
        import time

        start_time = time.time()

        # Group by state_hint
        hint_groups: dict[str, list[ApprovedTemplate]] = defaultdict(list)
        ungrouped: list[ApprovedTemplate] = []

        for template in templates:
            if template.state_hint:
                hint_groups[template.state_hint].append(template)
            else:
                ungrouped.append(template)

        # Create states from groups
        states = []
        for i, (hint, group_templates) in enumerate(hint_groups.items()):
            states.append(
                StateGroup(
                    state_id=f"state_{i}",
                    state_name=hint,
                    state_images=group_templates,
                    is_initial=(i == 0),
                    confidence=1.0,
                    metadata={"source": "state_hint"},
                )
            )

        processing_time = (time.time() - start_time) * 1000

        return GroupingResult(
            states=states,
            ungrouped_templates=ungrouped,
            grouping_method="state_hints",
            processing_time_ms=processing_time,
        )

    def group_by_co_occurrence(
        self,
        templates: list[ApprovedTemplate],
        frames: dict[int, set[str]] | None = None,
    ) -> GroupingResult:
        """Group templates by co-occurrence analysis.

        Templates that frequently appear together in the same frames
        are likely to identify the same state.

        This method uses the shared CoOccurrenceGrouper to perform grouping,
        ensuring consistency with the extraction feature's co-occurrence logic.

        IMPORTANT: This is a heuristic. User review is recommended.

        Args:
            templates: List of approved templates
            frames: Optional dict mapping frame_number to set of template IDs
                    present in that frame. This data comes from
                    CoOccurrenceAnalyzer.analyze_video() which detects
                    template presence via template matching.
                    If not provided, templates are grouped by the single
                    frame_number they were clicked in (not recommended).

        Returns:
            GroupingResult with co-occurrence-based states
        """
        import time

        start_time = time.time()

        if not templates:
            return GroupingResult(
                grouping_method="co_occurrence",
                processing_time_ms=0.0,
            )

        # Build frame-to-templates mapping if not provided
        # NOTE: This fallback is not ideal - co_occurrence should use
        # data from CoOccurrenceAnalyzer.analyze_video() which detects
        # templates across all frames, not just click frames.
        if frames is None:
            frames = defaultdict(set)
            for t in templates:
                frames[t.frame_number].add(t.id)

        # Convert to template→frames format for the shared grouper
        template_frame_map = invert_frame_template_map(frames)

        # Use shared CoOccurrenceGrouper
        grouper = CoOccurrenceGrouper(
            threshold=self.co_occurrence_threshold,
            strict_mode=False,
        )
        grouping_result = grouper.group(
            image_frame_map=template_frame_map,
            total_frames=len(frames),
        )

        # Convert CoOccurrenceGroups to StateGroups
        template_lookup = {t.id: t for t in templates}
        states: list[StateGroup] = []

        for i, group in enumerate(grouping_result.groups):
            state_images = [
                template_lookup[tid] for tid in group.image_ids if tid in template_lookup
            ]

            if not state_images:
                continue

            state_name = self._generate_state_name(i, state_images)

            states.append(
                StateGroup(
                    state_id=f"state_{i}",
                    state_name=state_name,
                    state_images=state_images,
                    is_initial=(i == 0),
                    confidence=group.confidence,
                    metadata={
                        "source": "co_occurrence",
                        "common_frames": list(group.common_frames),
                    },
                )
            )

        # Handle ungrouped templates
        ungrouped = [
            template_lookup[tid]
            for tid in grouping_result.ungrouped_images
            if tid in template_lookup
        ]

        processing_time = (time.time() - start_time) * 1000

        return GroupingResult(
            states=states,
            ungrouped_templates=ungrouped,
            co_occurrence_matrix=grouping_result.co_occurrence_matrix,
            grouping_method="co_occurrence",
            processing_time_ms=processing_time,
        )

    def create_single_state(
        self,
        templates: list[ApprovedTemplate],
        state_name: str = "Main State",
    ) -> GroupingResult:
        """Create a single state containing all templates.

        Simple approach when user wants to start with one state
        and refine later in the UI.

        Args:
            templates: List of approved templates
            state_name: Name for the single state

        Returns:
            GroupingResult with one state containing all templates
        """
        import time

        start_time = time.time()

        if not templates:
            return GroupingResult(
                grouping_method="single_state",
                processing_time_ms=0.0,
            )

        state = StateGroup(
            state_id="state_0",
            state_name=state_name,
            state_images=templates,
            is_initial=True,
            confidence=1.0,
            metadata={"source": "single_state"},
        )

        processing_time = (time.time() - start_time) * 1000

        return GroupingResult(
            states=[state],
            grouping_method="single_state",
            processing_time_ms=processing_time,
        )

    def create_one_state_per_template(
        self,
        templates: list[ApprovedTemplate],
    ) -> GroupingResult:
        """Create one state per template.

        Each template becomes the sole identifier for its own state.
        Useful when each clicked element represents a different state.

        Args:
            templates: List of approved templates

        Returns:
            GroupingResult with one state per template
        """
        import time

        start_time = time.time()

        states = []
        for i, template in enumerate(templates):
            state_name = template.name or f"State_{i}"
            states.append(
                StateGroup(
                    state_id=f"state_{i}",
                    state_name=state_name,
                    state_images=[template],
                    is_initial=(i == 0),
                    confidence=1.0,
                    metadata={"source": "one_per_template"},
                )
            )

        processing_time = (time.time() - start_time) * 1000

        return GroupingResult(
            states=states,
            grouping_method="one_per_template",
            processing_time_ms=processing_time,
        )

    def _generate_state_name(self, index: int, state_images: list[ApprovedTemplate]) -> str:
        """Generate a human-readable state name.

        Uses template names if available, otherwise generates generic name.

        Args:
            index: State index
            state_images: Templates in this state

        Returns:
            Generated state name
        """
        # Try to use template names
        named = [t.name for t in state_images if t.name]
        if named:
            # Use common prefix or first name
            if len(named) == 1:
                return named[0]
            # Find common prefix
            prefix = named[0]
            for name in named[1:]:
                while not name.startswith(prefix) and prefix:
                    prefix = prefix[:-1]
            if len(prefix) >= 3:
                return prefix.rstrip("_- ")
            return named[0]

        # Try element types
        element_types = [t.element_type for t in state_images if t.element_type != "unknown"]
        if element_types:
            most_common = max(set(element_types), key=element_types.count)
            return f"State_{index}_{most_common}"

        return f"State_{index}"
