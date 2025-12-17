"""
Data models for web extraction.

These models represent extracted GUI elements, states, and transitions
from web applications using Playwright.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


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

    PAGE = "page"  # Full page state (fallback when no semantic regions found)
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
class BoundingBox:
    """Bounding box for an element or region."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "BoundingBox":
        return cls(x=data["x"], y=data["y"], width=data["width"], height=data["height"])

    def intersects(self, other: "BoundingBox") -> bool:
        return not (
            self.x2 <= other.x
            or other.x2 <= self.x
            or self.y2 <= other.y
            or other.y2 <= self.y
        )

    def contains(self, other: "BoundingBox") -> bool:
        return (
            self.x <= other.x
            and self.y <= other.y
            and self.x2 >= other.x2
            and self.y2 >= other.y2
        )


@dataclass
class ExtractedElement:
    """A GUI element extracted from a web page."""

    id: str
    bbox: BoundingBox
    element_type: ElementType
    selector: str  # CSS selector for this element

    # Content
    text_content: str | None = None
    placeholder: str | None = None
    value: str | None = None
    alt_text: str | None = None

    # Semantic info
    semantic_role: str | None = None  # ARIA role
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
class ExtractedState:
    """A UI state/region extracted from a web page.

    A state is a collection of elements that appear/disappear together,
    such as a menu, dialog, or navigation bar.
    """

    id: str
    name: str
    bbox: BoundingBox
    state_type: StateType
    element_ids: list[str]  # Elements within this state
    screenshot_id: str  # Reference to local screenshot file

    # Detection info
    detection_method: str = "visibility_cluster"  # visibility_cluster, semantic, manual
    confidence: float = 1.0

    # Semantic info
    semantic_role: str | None = None  # ARIA landmark role
    aria_label: str | None = None

    # Metadata
    source_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "bbox": self.bbox.to_dict(),
            "state_type": self.state_type.value,
            "element_ids": self.element_ids,
            "screenshot_id": self.screenshot_id,
            "detection_method": self.detection_method,
            "confidence": self.confidence,
            "semantic_role": self.semantic_role,
            "aria_label": self.aria_label,
            "source_url": self.source_url,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractedState":
        return cls(
            id=data["id"],
            name=data["name"],
            bbox=BoundingBox.from_dict(data["bbox"]),
            state_type=StateType(data["state_type"]),
            element_ids=data["element_ids"],
            screenshot_id=data["screenshot_id"],
            detection_method=data.get("detection_method", "visibility_cluster"),
            confidence=data.get("confidence", 1.0),
            semantic_role=data.get("semantic_role"),
            aria_label=data.get("aria_label"),
            source_url=data.get("source_url", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExtractedTransition:
    """A transition between states caused by a GUI action."""

    id: str
    action_type: TransitionType
    target_element_id: str  # Element that triggers the transition
    target_selector: str  # CSS selector for the target

    # State changes
    causes_appear: list[str] = field(default_factory=list)  # State IDs that appear
    causes_disappear: list[str] = field(
        default_factory=list
    )  # State IDs that disappear

    # Action details
    action_value: str | None = None  # For type actions, the text typed
    key_modifiers: list[str] = field(default_factory=list)  # ctrl, shift, alt, etc.

    # Metadata
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "action_type": self.action_type.value,
            "target_element_id": self.target_element_id,
            "target_selector": self.target_selector,
            "causes_appear": self.causes_appear,
            "causes_disappear": self.causes_disappear,
            "action_value": self.action_value,
            "key_modifiers": self.key_modifiers,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractedTransition":
        return cls(
            id=data["id"],
            action_type=TransitionType(data["action_type"]),
            target_element_id=data["target_element_id"],
            target_selector=data["target_selector"],
            causes_appear=data.get("causes_appear", []),
            causes_disappear=data.get("causes_disappear", []),
            action_value=data.get("action_value"),
            key_modifiers=data.get("key_modifiers", []),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PageExtraction:
    """Extraction results for a single page."""

    url: str
    title: str
    viewport: tuple[int, int]
    elements: list[ExtractedElement]
    states: list[ExtractedState]
    screenshot_ids: list[str]
    extracted_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "viewport": list(self.viewport),
            "elements": [e.to_dict() for e in self.elements],
            "states": [s.to_dict() for s in self.states],
            "screenshot_ids": self.screenshot_ids,
            "extracted_at": self.extracted_at.isoformat(),
        }


@dataclass
class ExtractionResult:
    """Complete extraction results for a web application."""

    extraction_id: str
    source_urls: list[str]
    viewports: list[tuple[int, int]]

    # Extracted data
    elements: list[ExtractedElement] = field(default_factory=list)
    states: list[ExtractedState] = field(default_factory=list)
    transitions: list[ExtractedTransition] = field(default_factory=list)
    page_extractions: list[PageExtraction] = field(default_factory=list)

    # Screenshot references (stored locally on runner)
    screenshot_ids: list[str] = field(default_factory=list)

    # Metadata
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "extraction_id": self.extraction_id,
            "source_urls": self.source_urls,
            "viewports": [list(v) for v in self.viewports],
            "elements": [e.to_dict() for e in self.elements],
            "states": [s.to_dict() for s in self.states],
            "transitions": [t.to_dict() for t in self.transitions],
            "page_extractions": [p.to_dict() for p in self.page_extractions],
            "screenshot_ids": self.screenshot_ids,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "config": self.config,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractionResult":
        result = cls(
            extraction_id=data["extraction_id"],
            source_urls=data["source_urls"],
            viewports=[tuple(v) for v in data["viewports"]],
            screenshot_ids=data.get("screenshot_ids", []),
            config=data.get("config", {}),
            metadata=data.get("metadata", {}),
        )
        result.elements = [
            ExtractedElement.from_dict(e) for e in data.get("elements", [])
        ]
        result.states = [ExtractedState.from_dict(s) for s in data.get("states", [])]
        result.transitions = [
            ExtractedTransition.from_dict(t) for t in data.get("transitions", [])
        ]
        if data.get("started_at"):
            result.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            result.completed_at = datetime.fromisoformat(data["completed_at"])
        return result

    def get_element_by_id(self, element_id: str) -> ExtractedElement | None:
        for element in self.elements:
            if element.id == element_id:
                return element
        return None

    def get_state_by_id(self, state_id: str) -> ExtractedState | None:
        for state in self.states:
            if state.id == state_id:
                return state
        return None

    def get_elements_in_state(self, state_id: str) -> list[ExtractedElement]:
        state = self.get_state_by_id(state_id)
        if not state:
            return []
        return [e for e in self.elements if e.id in state.element_ids]
