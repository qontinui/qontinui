"""State construction implementation.

This module provides functionality for constructing state objects from
detected elements and regions.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Element:
    """Represents a UI element in a state.

    Attributes:
        element_id: Unique identifier
        element_type: Type of element (button, text_field, etc.)
        bounds: Bounding box (x, y, width, height)
        properties: Additional element properties
        is_defining: Whether this element defines the state
    """

    element_id: str
    element_type: str
    bounds: tuple[int, int, int, int]
    properties: dict = field(default_factory=dict)
    is_defining: bool = False


@dataclass
class Transition:
    """Represents a transition to another state.

    Attributes:
        target_state_id: ID of the target state
        trigger_element_id: Element that triggers the transition
        confidence: Confidence in this transition
        action_type: Type of action (click, type, etc.)
    """

    target_state_id: str
    trigger_element_id: str | None = None
    confidence: float = 0.0
    action_type: str = "click"


@dataclass
class ConstructedState:
    """A complete constructed state object.

    Attributes:
        state_id: Unique state identifier
        name: Human-readable name
        defining_elements: Elements that define this state
        optional_elements: Elements that may be present
        transitions: Possible transitions from this state
        screenshot: Reference screenshot
        metadata: Additional state metadata
    """

    state_id: str
    name: str
    defining_elements: list[Element] = field(default_factory=list)
    optional_elements: list[Element] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    screenshot: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        """String representation of constructed state."""
        return (
            f"ConstructedState(id={self.state_id}, name={self.name}, "
            f"defining={len(self.defining_elements)}, "
            f"optional={len(self.optional_elements)})"
        )


class StateBuilder:
    """Builds state objects from detection data.

    The builder takes detected elements and regions and constructs
    complete state representations with defining features, relationships,
    and metadata.
    """

    def __init__(self):
        """Initialize state builder."""
        self.defining_element_threshold = 0.8  # Confidence for defining elements
        self.optional_element_threshold = 0.5  # Confidence for optional elements

    def construct(
        self,
        state_id: str,
        name: str,
        screenshot: np.ndarray,
        detected_elements: list[Element],
        regions: list | None = None,
    ) -> ConstructedState:
        """Construct a state from detection data.

        Args:
            state_id: Unique state identifier
            name: State name
            screenshot: Reference screenshot
            detected_elements: List of detected elements
            regions: Optional list of regions

        Returns:
            Constructed state object
        """
        # Categorize elements by importance
        defining_elements = []
        optional_elements = []

        for element in detected_elements:
            if element.is_defining or self._is_defining_element(element):
                defining_elements.append(element)
            else:
                optional_elements.append(element)

        # Create state object
        state = ConstructedState(
            state_id=state_id,
            name=name,
            defining_elements=defining_elements,
            optional_elements=optional_elements,
            screenshot=screenshot.copy() if screenshot is not None else None,
            metadata={
                "total_elements": len(detected_elements),
                "regions": len(regions) if regions else 0,
            },
        )

        return state

    def _is_defining_element(self, element: Element) -> bool:
        """Determine if an element is defining for the state.

        TODO: Implement sophisticated logic to identify defining elements.

        Args:
            element: Element to evaluate

        Returns:
            True if element is defining
        """
        # Placeholder: Use simple heuristics
        # In practice, this could use ML or learned importance scores

        # Elements with high uniqueness should be defining
        if element.properties.get("uniqueness", 0) > 0.8:
            return True

        # Certain element types are more likely to be defining
        defining_types = {"logo", "title", "heading", "unique_icon"}
        if element.element_type in defining_types:
            return True

        return False

    def add_transition(self, state: ConstructedState, transition: Transition) -> None:
        """Add a transition to a state.

        Args:
            state: State to add transition to
            transition: Transition to add
        """
        # Check if transition already exists
        for existing in state.transitions:
            if existing.target_state_id == transition.target_state_id:
                # Update confidence if higher
                if transition.confidence > existing.confidence:
                    existing.confidence = transition.confidence
                    existing.trigger_element_id = transition.trigger_element_id
                return

        state.transitions.append(transition)

    def merge_similar_states(
        self, state1: ConstructedState, state2: ConstructedState, similarity: float
    ) -> ConstructedState | None:
        """Merge two similar states into one.

        Args:
            state1: First state
            state2: Second state
            similarity: Similarity score between states

        Returns:
            Merged state if similarity is high enough, None otherwise
        """
        if similarity < 0.9:  # High threshold for merging
            return None

        # Merge defining elements
        all_defining_ids = {e.element_id for e in state1.defining_elements} | {
            e.element_id for e in state2.defining_elements
        }

        merged_defining = []
        for elem in state1.defining_elements + state2.defining_elements:
            if elem.element_id in all_defining_ids:
                merged_defining.append(elem)
                all_defining_ids.remove(elem.element_id)

        # Merge optional elements
        all_optional_ids = {e.element_id for e in state1.optional_elements} | {
            e.element_id for e in state2.optional_elements
        }

        merged_optional = []
        for elem in state1.optional_elements + state2.optional_elements:
            if elem.element_id in all_optional_ids:
                merged_optional.append(elem)
                all_optional_ids.remove(elem.element_id)

        # Create merged state
        merged = ConstructedState(
            state_id=state1.state_id,
            name=state1.name,
            defining_elements=merged_defining,
            optional_elements=merged_optional,
            transitions=state1.transitions + state2.transitions,
            screenshot=state1.screenshot,
            metadata={
                "merged_from": [state1.state_id, state2.state_id],
                "similarity": similarity,
            },
        )

        return merged


class StateValidator:
    """Validates constructed states for completeness and correctness."""

    def validate(self, state: ConstructedState) -> tuple[bool, list[str]]:
        """Validate a constructed state.

        Args:
            state: State to validate

        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        errors = []

        # Check for required fields
        if not state.state_id:
            errors.append("State ID is required")

        if not state.name:
            errors.append("State name is required")

        # Check for at least one defining element
        if not state.defining_elements:
            errors.append("State must have at least one defining element")

        # Check for duplicate element IDs
        all_elements = state.defining_elements + state.optional_elements
        element_ids = [e.element_id for e in all_elements]
        if len(element_ids) != len(set(element_ids)):
            errors.append("Duplicate element IDs found")

        # Check for valid transitions
        for transition in state.transitions:
            if not transition.target_state_id:
                errors.append("Transition missing target state ID")

        return len(errors) == 0, errors


class FeatureIdentifier:
    """Identifies defining features of states.

    Analyzes elements and their properties to determine which features
    are most important for identifying a state.
    """

    def identify_defining_features(
        self, elements: list[Element], screenshot: np.ndarray | None = None
    ) -> set[str]:
        """Identify which element features are defining.

        Args:
            elements: List of elements to analyze
            screenshot: Optional screenshot for visual analysis

        Returns:
            Set of element IDs that are defining features
        """
        defining = set()

        for element in elements:
            # Check uniqueness
            if self._is_unique(element, elements):
                defining.add(element.element_id)

            # Check visual distinctiveness
            if screenshot is not None and self._is_visually_distinctive(
                element, screenshot
            ):
                defining.add(element.element_id)

        return defining

    def _is_unique(self, element: Element, all_elements: list[Element]) -> bool:
        """Check if an element is unique among all elements.

        Args:
            element: Element to check
            all_elements: All elements in the state

        Returns:
            True if element is unique
        """
        # Count similar elements
        similar_count = sum(
            1
            for e in all_elements
            if e.element_type == element.element_type
            and e.element_id != element.element_id
        )

        return similar_count == 0

    def _is_visually_distinctive(
        self, element: Element, screenshot: np.ndarray
    ) -> bool:
        """Check if an element is visually distinctive.

        Args:
            element: Element to check
            screenshot: Screenshot containing the element

        Returns:
            True if element is visually distinctive
        """
        x, y, w, h = element.bounds

        # Validate bounds
        screen_height, screen_width = screenshot.shape[:2]
        if x < 0 or y < 0 or x + w > screen_width or y + h > screen_height:
            return False

        if w <= 0 or h <= 0:
            return False

        # Extract element region
        element_region = screenshot[y : y + h, x : x + w]

        if element_region.size == 0:
            return False

        # Analyze color distinctiveness
        mean_color = np.mean(element_region, axis=(0, 1))
        std_color = np.std(element_region, axis=(0, 1))

        # High color variance suggests visual complexity
        color_variance = np.mean(std_color)
        is_color_distinctive = color_variance > 30

        # Analyze texture using Laplacian variance (edge content)
        gray = cv2.cvtColor(element_region, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = laplacian.var()

        # High texture variance suggests distinctive patterns
        is_texture_distinctive = texture_variance > 100

        # Analyze shape using contours
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Multiple distinct contours suggest complex shape
        is_shape_distinctive = len(contours) > 3

        # Check color uniqueness by comparing to surrounding area
        padding = 10
        y_start = max(0, y - padding)
        y_end = min(screen_height, y + h + padding)
        x_start = max(0, x - padding)
        x_end = min(screen_width, x + w + padding)

        surrounding = screenshot[y_start:y_end, x_start:x_end]
        if surrounding.size > element_region.size:
            surrounding_mean = np.mean(surrounding, axis=(0, 1))
            color_difference = np.linalg.norm(mean_color - surrounding_mean)
            is_color_unique = color_difference > 50
        else:
            is_color_unique = False

        # Element is distinctive if it meets multiple criteria
        distinctive_score = sum(
            [
                is_color_distinctive,
                is_texture_distinctive,
                is_shape_distinctive,
                is_color_unique,
            ]
        )

        return distinctive_score >= 2
