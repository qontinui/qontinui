"""
Runtime extraction models.

These models represent data extracted from running applications - elements,
regions, states, and observed transitions from user interactions.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .base import BoundingBox, Screenshot, Viewport


class ElementType(Enum):
    """Types of GUI elements that can be extracted."""

    BUTTON = "button"
    TEXT_INPUT = "text_input"
    PASSWORD_INPUT = "password_input"
    TEXTAREA = "textarea"
    LINK = "link"
    DROPDOWN = "dropdown"
    CHECKBOX = "checkbox"
    RADIO = "radio"
    SLIDER = "slider"
    TOGGLE = "toggle"
    TAB = "tab"
    MENU_ITEM = "menu_item"
    ICON_BUTTON = "icon_button"
    IMAGE = "image"
    LABEL = "label"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE_CELL = "table_cell"
    UNKNOWN = "unknown"


class StateType(Enum):
    """Types of UI states/regions."""

    NAVIGATION = "navigation"
    MENU = "menu"
    DROPDOWN_MENU = "dropdown_menu"
    DIALOG = "dialog"
    MODAL = "modal"
    SIDEBAR = "sidebar"
    TOOLBAR = "toolbar"
    FORM = "form"
    CARD = "card"
    PANEL = "panel"
    TOAST = "toast"
    TOOLTIP = "tooltip"
    POPOVER = "popover"
    HEADER = "header"
    FOOTER = "footer"
    CONTENT = "content"
    UNKNOWN = "unknown"


class TransitionType(Enum):
    """Types of transitions/actions."""

    CLICK = "click"
    HOVER = "hover"
    FOCUS = "focus"
    BLUR = "blur"
    SCROLL = "scroll"
    TYPE = "type"
    SELECT = "select"
    DRAG = "drag"
    SWIPE = "swipe"
    KEY_PRESS = "key_press"
    URL_CHANGE = "url_change"
    TIME = "time"  # Auto-dismiss, timeouts


@dataclass
class ExtractedElement:
    """A GUI element extracted from a running application."""

    id: str
    bbox: BoundingBox
    element_type: ElementType
    selector: str  # CSS selector or platform-specific selector

    # Content
    text_content: str | None = None
    placeholder: str | None = None
    value: str | None = None
    alt_text: str | None = None

    # Semantic info
    semantic_role: str | None = None  # ARIA role or platform role
    aria_label: str | None = None
    name: str | None = None  # Computed accessible name

    # State
    is_interactive: bool = True
    is_enabled: bool = True
    is_visible: bool = True
    is_focused: bool = False
    element_state: str | None = None  # checked, expanded, selected, etc.

    # Hierarchy
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)

    # Additional attributes
    attributes: dict[str, Any] = field(default_factory=dict)
    tag_name: str = ""
    class_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "bbox": self.bbox.to_dict(),
            "element_type": self.element_type.value,
            "selector": self.selector,
            "text_content": self.text_content,
            "placeholder": self.placeholder,
            "value": self.value,
            "alt_text": self.alt_text,
            "semantic_role": self.semantic_role,
            "aria_label": self.aria_label,
            "name": self.name,
            "is_interactive": self.is_interactive,
            "is_enabled": self.is_enabled,
            "is_visible": self.is_visible,
            "is_focused": self.is_focused,
            "element_state": self.element_state,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "attributes": self.attributes,
            "tag_name": self.tag_name,
            "class_names": self.class_names,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractedElement":
        """Create from dictionary representation."""
        return cls(
            id=data["id"],
            bbox=BoundingBox.from_dict(data["bbox"]),
            element_type=ElementType(data["element_type"]),
            selector=data["selector"],
            text_content=data.get("text_content"),
            placeholder=data.get("placeholder"),
            value=data.get("value"),
            alt_text=data.get("alt_text"),
            semantic_role=data.get("semantic_role"),
            aria_label=data.get("aria_label"),
            name=data.get("name"),
            is_interactive=data.get("is_interactive", True),
            is_enabled=data.get("is_enabled", True),
            is_visible=data.get("is_visible", True),
            is_focused=data.get("is_focused", False),
            element_state=data.get("element_state"),
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            attributes=data.get("attributes", {}),
            tag_name=data.get("tag_name", ""),
            class_names=data.get("class_names", []),
        )


@dataclass
class DetectedRegion:
    """A detected UI region/container."""

    id: str
    bbox: BoundingBox
    region_type: StateType
    selector: str | None = None

    # Semantic info
    semantic_role: str | None = None
    aria_label: str | None = None

    # Elements
    element_count: int = 0

    # Detection confidence
    confidence: float = 1.0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractionAction:
    """An action performed during runtime extraction."""

    action_type: TransitionType
    target_selector: str | None = None
    target_element_id: str | None = None
    value: str | None = None  # For TYPE actions
    key_modifiers: list[str] = field(default_factory=list)  # ctrl, shift, alt, etc.


@dataclass
class RuntimeStateCapture:
    """A captured state of the UI at a point in time."""

    id: str
    timestamp: str
    screenshot: Screenshot
    viewport: Viewport
    elements: list[ExtractedElement]
    regions: list[DetectedRegion]

    # Navigation
    url: str | None = None
    route_params: dict[str, str] = field(default_factory=dict)

    # Trigger
    trigger_action: InteractionAction | None = None
    previous_state_id: str | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_element_by_id(self, element_id: str) -> ExtractedElement | None:
        """Get element by ID."""
        for element in self.elements:
            if element.id == element_id:
                return element
        return None

    def get_region_by_id(self, region_id: str) -> DetectedRegion | None:
        """Get region by ID."""
        for region in self.regions:
            if region.id == region_id:
                return region
        return None


@dataclass
class ObservedTransition:
    """A transition observed during runtime extraction."""

    id: str
    action: InteractionAction
    before_state_id: str
    after_state_id: str

    # Element changes
    elements_appeared: list[str] = field(default_factory=list)  # Element IDs
    elements_disappeared: list[str] = field(default_factory=list)  # Element IDs

    # Region changes
    regions_appeared: list[str] = field(default_factory=list)  # Region IDs
    regions_disappeared: list[str] = field(default_factory=list)  # Region IDs

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeExtractionResult:
    """Complete result of runtime extraction."""

    states: list[RuntimeStateCapture] = field(default_factory=list)
    transitions: list[ObservedTransition] = field(default_factory=list)
    routes_visited: list[str] = field(default_factory=list)
    elements_by_state: dict[str, list[str]] = field(default_factory=dict)  # state_id -> element_ids
    screenshots_dir: Path | None = None

    def get_state(self, state_id: str) -> RuntimeStateCapture | None:
        """Get state by ID."""
        for state in self.states:
            if state.id == state_id:
                return state
        return None

    def get_transition(self, transition_id: str) -> ObservedTransition | None:
        """Get transition by ID."""
        for transition in self.transitions:
            if transition.id == transition_id:
                return transition
        return None

    def get_initial_state(self) -> RuntimeStateCapture | None:
        """Get the initial state (first captured state)."""
        if not self.states:
            return None
        return self.states[0]
