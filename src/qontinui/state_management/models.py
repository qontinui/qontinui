"""Data models for state management."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, cast

import numpy as np


class ElementType(Enum):
    """Types of UI elements."""

    BUTTON = "button"
    TEXT = "text"
    INPUT = "input"
    IMAGE = "image"
    ICON = "icon"
    MENU = "menu"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    DROPDOWN = "dropdown"
    LINK = "link"
    CONTAINER = "container"
    UNKNOWN = "unknown"


class TransitionType(Enum):
    """Types of state transitions."""

    CLICK = "click"
    TYPE = "type"
    HOVER = "hover"
    DRAG = "drag"
    SCROLL = "scroll"
    KEY_PRESS = "key_press"
    WAIT = "wait"
    CUSTOM = "custom"


@dataclass
class Element:
    """Represents a UI element within a state."""

    id: str
    bbox: tuple[int, int, int, int]  # (x, y, width, height)
    embedding: np.ndarray[Any, Any] | None = None
    description: str = ""
    element_type: ElementType = ElementType.UNKNOWN
    co_occurrences: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    text_content: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate element data."""
        if len(self.bbox) != 4:
            raise ValueError(f"bbox must have 4 values, got {len(self.bbox)}")

        # Ensure all bbox values are non-negative
        if any(v < 0 for v in self.bbox):
            raise ValueError(f"bbox values must be non-negative, got {self.bbox}")

    def to_dict(self) -> dict[str, Any]:
        """Convert element to dictionary representation."""
        return {
            "id": self.id,
            "bbox": self.bbox,
            "embedding": (
                self.embedding.tolist() if self.embedding is not None else None
            ),
            "description": self.description,
            "element_type": self.element_type.value,
            "co_occurrences": self.co_occurrences,
            "provenance": self.provenance,
            "confidence": self.confidence,
            "text_content": self.text_content,
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Element":
        """Create Element from dictionary."""
        embedding = None
        if data.get("embedding"):
            embedding = np.array(data["embedding"])

        element_type = ElementType(data.get("element_type", "unknown"))

        return cls(
            id=data["id"],
            bbox=tuple(data["bbox"]),
            embedding=embedding,
            description=data.get("description", ""),
            element_type=element_type,
            co_occurrences=data.get("co_occurrences", []),
            provenance=data.get("provenance", {}),
            confidence=data.get("confidence", 1.0),
            text_content=data.get("text_content"),
            attributes=data.get("attributes", {}),
        )

    def overlaps_with(self, other: "Element") -> bool:
        """Check if this element overlaps with another."""
        x1, y1, w1, h1 = self.bbox
        x2, y2, w2, h2 = other.bbox

        # Check if rectangles overlap
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)

    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is within this element's bounding box."""
        ex, ey, ew, eh = self.bbox
        return ex <= x <= ex + ew and ey <= y <= ey + eh

    def center(self) -> tuple[int, int]:
        """Get the center point of the element."""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)


@dataclass
class Transition:
    """Represents a transition between states."""

    from_state: str
    to_state: str
    action_type: TransitionType
    trigger_element: str | None = None  # Element ID that triggers transition
    action_data: dict[str, Any] = field(default_factory=dict)
    probability: float = 1.0
    conditions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert transition to dictionary."""
        return {
            "from_state": self.from_state,
            "to_state": self.to_state,
            "action_type": self.action_type.value,
            "trigger_element": self.trigger_element,
            "action_data": self.action_data,
            "probability": self.probability,
            "conditions": self.conditions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transition":
        """Create Transition from dictionary."""
        return cls(
            from_state=data["from_state"],
            to_state=data["to_state"],
            action_type=TransitionType(data["action_type"]),
            trigger_element=data.get("trigger_element"),
            action_data=data.get("action_data", {}),
            probability=data.get("probability", 1.0),
            conditions=data.get("conditions", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class State:
    """Represents an application state."""

    name: str
    elements: list[Element]
    min_elements: int = 1
    transitions: list[Transition] = field(default_factory=list)
    parent_state: str | None = None  # For hierarchical states
    sub_states: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    screenshot_path: str | None = None
    timestamp: datetime | None = None

    def __post_init__(self):
        """Initialize state properties."""
        if self.timestamp is None:
            self.timestamp = datetime.now()

        # Build element lookup
        self._element_lookup = {elem.id: elem for elem in self.elements}

    def get_element(self, element_id: str) -> Element | None:
        """Get element by ID."""
        return cast(Element | None, self._element_lookup.get(element_id))

    def find_elements_by_type(self, element_type: ElementType) -> list[Element]:
        """Find all elements of a specific type."""
        return [elem for elem in self.elements if elem.element_type == element_type]

    def find_element_at_position(self, x: int, y: int) -> Element | None:
        """Find element at a specific position."""
        for element in self.elements:
            if element.contains_point(x, y):
                return element
        return None

    def add_transition(self, transition: Transition):
        """Add a transition from this state."""
        self.transitions.append(transition)

    def get_transitions_to(self, target_state: str) -> list[Transition]:
        """Get all transitions to a specific state."""
        return [t for t in self.transitions if t.to_state == target_state]

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "name": self.name,
            "elements": [elem.to_dict() for elem in self.elements],
            "min_elements": self.min_elements,
            "transitions": [trans.to_dict() for trans in self.transitions],
            "parent_state": self.parent_state,
            "sub_states": self.sub_states,
            "metadata": self.metadata,
            "screenshot_path": self.screenshot_path,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "State":
        """Create State from dictionary."""
        elements = [Element.from_dict(e) for e in data["elements"]]
        transitions = [Transition.from_dict(t) for t in data.get("transitions", [])]

        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])

        return cls(
            name=data["name"],
            elements=elements,
            min_elements=data.get("min_elements", 1),
            transitions=transitions,
            parent_state=data.get("parent_state"),
            sub_states=data.get("sub_states", []),
            metadata=data.get("metadata", {}),
            screenshot_path=data.get("screenshot_path"),
            timestamp=timestamp,
        )

    def is_active(
        self, current_elements: list[Element], threshold: float = 0.7
    ) -> bool:
        """Check if this state is currently active based on visible elements.

        Args:
            current_elements: Currently visible elements
            threshold: Minimum match threshold

        Returns:
            True if state is active
        """
        if len(current_elements) < self.min_elements:
            return False

        # Count matching elements
        matches = 0
        for state_elem in self.elements:
            for current_elem in current_elements:
                # Simple position-based matching for now
                # In practice, would use embedding similarity
                if self._elements_match(state_elem, current_elem, threshold):
                    matches += 1
                    break

        # Check if enough elements match
        match_ratio = matches / len(self.elements) if self.elements else 0
        return match_ratio >= threshold

    def _elements_match(self, elem1: Element, elem2: Element, threshold: float) -> bool:
        """Check if two elements match.

        Args:
            elem1: First element
            elem2: Second element
            threshold: Match threshold

        Returns:
            True if elements match
        """
        # Simple bbox overlap check for now
        # In practice, would use embedding similarity
        return elem1.overlaps_with(elem2)


@dataclass
class StateGraph:
    """Graph representation of application states and transitions."""

    states: dict[str, State] = field(default_factory=dict)
    initial_state: str | None = None
    current_state: str | None = None

    def add_state(self, state: State):
        """Add a state to the graph."""
        self.states[state.name] = state

        if self.initial_state is None:
            self.initial_state = state.name

    def add_transition(self, transition: Transition):
        """Add a transition to the graph."""
        if transition.from_state in self.states:
            self.states[transition.from_state].add_transition(transition)

    def get_state(self, state_name: str) -> State | None:
        """Get state by name."""
        return self.states.get(state_name)

    def get_neighbors(self, state_name: str) -> list[str]:
        """Get neighboring states."""
        state = self.get_state(state_name)
        if not state:
            return []

        return list({t.to_state for t in state.transitions})

    def find_path(self, start: str, end: str) -> list[str] | None:
        """Find path between two states using BFS.

        Args:
            start: Starting state name
            end: Target state name

        Returns:
            List of state names forming the path, or None if no path exists
        """
        if start not in self.states or end not in self.states:
            return None

        if start == end:
            return [start]

        # BFS to find shortest path
        from collections import deque

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()

            for neighbor in self.get_neighbors(current):
                if neighbor == end:
                    return path + [end]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary."""
        return {
            "states": {name: state.to_dict() for name, state in self.states.items()},
            "initial_state": self.initial_state,
            "current_state": self.current_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateGraph":
        """Create StateGraph from dictionary."""
        graph = cls()

        # Load states
        for _name, state_data in data.get("states", {}).items():
            state = State.from_dict(state_data)
            graph.add_state(state)

        graph.initial_state = data.get("initial_state")
        graph.current_state = data.get("current_state")

        return graph

    def save(self, filepath: str):
        """Save graph to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "StateGraph":
        """Load graph from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        return cls.from_dict(data)
