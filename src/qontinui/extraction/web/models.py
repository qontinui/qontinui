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
            self.x2 <= other.x or other.x2 <= self.x or self.y2 <= other.y or other.y2 <= self.y
        )

    def contains(self, other: "BoundingBox") -> bool:
        return (
            self.x <= other.x and self.y <= other.y and self.x2 >= other.x2 and self.y2 >= other.y2
        )


@dataclass
class InteractiveElement:
    """
    An interactive element extracted from a web page.

    This is the primary model for DOM-based extraction, capturing only
    elements users can interact with: buttons, links, form inputs, etc.
    """

    id: str
    bbox: BoundingBox
    tag_name: str
    element_type: str  # button, a, input, aria_button, tabindex_div, etc.
    screenshot_id: str
    selector: str

    # Content
    text: str | None = None
    href: str | None = None
    aria_label: str | None = None
    aria_role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "bbox": self.bbox.to_dict(),
            "tag_name": self.tag_name,
            "element_type": self.element_type,
            "screenshot_id": self.screenshot_id,
            "selector": self.selector,
            "text": self.text,
            "href": self.href,
            "aria_label": self.aria_label,
            "aria_role": self.aria_role,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InteractiveElement":
        return cls(
            id=data["id"],
            bbox=BoundingBox.from_dict(data["bbox"]),
            tag_name=data["tag_name"],
            element_type=data["element_type"],
            screenshot_id=data["screenshot_id"],
            selector=data["selector"],
            text=data.get("text"),
            href=data.get("href"),
            aria_label=data.get("aria_label"),
            aria_role=data.get("aria_role"),
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

    # Debugging: why this element was extracted
    extraction_category: str = ""

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
            "extraction_category": self.extraction_category,
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
            extraction_category=data.get("extraction_category", ""),
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
    screenshot_id: str | None = None  # Reference to local screenshot file (set after capture)

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
    causes_disappear: list[str] = field(default_factory=list)  # State IDs that disappear

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
            "completed_at": (self.completed_at.isoformat() if self.completed_at else None),
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
        result.elements = [ExtractedElement.from_dict(e) for e in data.get("elements", [])]
        result.states = [ExtractedState.from_dict(s) for s in data.get("states", [])]
        result.transitions = [ExtractedTransition.from_dict(t) for t in data.get("transitions", [])]
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


# =============================================================================
# New Models for Deterministic State Machine Generation
# =============================================================================


class FunctionType(Enum):
    """Types of element functions for transition detection."""

    NAVIGATE = "navigate"  # href, router link
    SUBMIT = "submit"  # Form submission
    TOGGLE = "toggle"  # Toggle visibility (aria-expanded, etc.)
    EXPAND = "expand"  # Expand/collapse content
    OPEN_MODAL = "open_modal"  # Opens a modal/dialog
    CLOSE_MODAL = "close_modal"  # Closes a modal/dialog
    OPEN_DROPDOWN = "open_dropdown"  # Opens a dropdown menu
    API_CALL = "api_call"  # Triggers an API call
    UNKNOWN = "unknown"  # Unknown function


class SizeClass(Enum):
    """Size classification for element fingerprinting."""

    TINY = "tiny"  # < 32px in either dimension
    SMALL = "small"  # 32-100px
    MEDIUM = "medium"  # 100-300px
    LARGE = "large"  # 300-600px
    XLARGE = "xlarge"  # > 600px


class PositionRegion(Enum):
    """Viewport region classification for element position."""

    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    MIDDLE_LEFT = "middle_left"
    MIDDLE_CENTER = "middle_center"
    MIDDLE_RIGHT = "middle_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"


@dataclass
class RawElement:
    """
    Comprehensive element data captured during extraction.

    This model captures ALL visible HTML elements with rich data for:
    - Visual comparison (computed styles, colors, screenshots)
    - Transition detection (href, onclick, form actions)
    - State machine generation (containment filtering, fingerprinting)

    Unlike ExtractedElement which focuses on interactive elements,
    RawElement captures all visible elements for comprehensive analysis.
    """

    id: str
    bbox: BoundingBox
    tag_name: str
    screenshot_id: str  # Reference to page screenshot
    selector: str  # CSS selector for this element

    # Visual properties
    background_color: tuple[int, int, int, int] | None = None  # RGBA
    border_color: tuple[int, int, int, int] | None = None  # RGBA
    text_color: tuple[int, int, int, int] | None = None  # RGBA
    computed_styles: dict[str, str] = field(default_factory=dict)
    image_asset_id: str | None = None  # Reference to cropped element image

    # Content properties
    text_content: str | None = None  # Direct text content (not children)
    full_text_content: str | None = None  # All text including children
    inner_html_hash: str = ""  # Hash of innerHTML for comparison
    attributes: dict[str, str] = field(default_factory=dict)  # ALL attributes

    # Function properties (for transition detection)
    href: str | None = None
    onclick: str | None = None
    form_action: str | None = None
    form_method: str | None = None
    aria_controls: str | None = None
    aria_expanded: str | None = None
    data_attributes: dict[str, str] = field(default_factory=dict)  # All data-* attrs

    # Semantic properties (for naming only, not detection)
    semantic_role: str | None = None  # ARIA role
    aria_label: str | None = None
    tag_semantic: str | None = None  # Semantic tag: nav, header, aside, etc.

    # Structure properties
    parent_selector: str = ""
    parent_id: str | None = None
    child_count: int = 0
    dom_depth: int = 0
    z_index: int = 0

    # Classification
    is_interactive: bool = False
    is_visible: bool = True
    is_container: bool = False  # Set during containment filtering
    has_visual_content: bool = False  # Has background/border/own text

    # Debugging: why this element was extracted
    extraction_category: str = ""  # interactive_tag, interactive_aria, media, leaf_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "bbox": self.bbox.to_dict(),
            "tag_name": self.tag_name,
            "screenshot_id": self.screenshot_id,
            "selector": self.selector,
            "background_color": self.background_color,
            "border_color": self.border_color,
            "text_color": self.text_color,
            "computed_styles": self.computed_styles,
            "image_asset_id": self.image_asset_id,
            "text_content": self.text_content,
            "full_text_content": self.full_text_content,
            "inner_html_hash": self.inner_html_hash,
            "attributes": self.attributes,
            "href": self.href,
            "onclick": self.onclick,
            "form_action": self.form_action,
            "form_method": self.form_method,
            "aria_controls": self.aria_controls,
            "aria_expanded": self.aria_expanded,
            "data_attributes": self.data_attributes,
            "semantic_role": self.semantic_role,
            "aria_label": self.aria_label,
            "tag_semantic": self.tag_semantic,
            "parent_selector": self.parent_selector,
            "parent_id": self.parent_id,
            "child_count": self.child_count,
            "dom_depth": self.dom_depth,
            "z_index": self.z_index,
            "is_interactive": self.is_interactive,
            "is_visible": self.is_visible,
            "is_container": self.is_container,
            "has_visual_content": self.has_visual_content,
            "extraction_category": self.extraction_category,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RawElement":
        return cls(
            id=data["id"],
            bbox=BoundingBox.from_dict(data["bbox"]),
            tag_name=data["tag_name"],
            screenshot_id=data["screenshot_id"],
            selector=data["selector"],
            background_color=(
                tuple(data["background_color"]) if data.get("background_color") else None
            ),
            border_color=tuple(data["border_color"]) if data.get("border_color") else None,
            text_color=tuple(data["text_color"]) if data.get("text_color") else None,
            computed_styles=data.get("computed_styles", {}),
            image_asset_id=data.get("image_asset_id"),
            text_content=data.get("text_content"),
            full_text_content=data.get("full_text_content"),
            inner_html_hash=data.get("inner_html_hash", ""),
            attributes=data.get("attributes", {}),
            href=data.get("href"),
            onclick=data.get("onclick"),
            form_action=data.get("form_action"),
            form_method=data.get("form_method"),
            aria_controls=data.get("aria_controls"),
            aria_expanded=data.get("aria_expanded"),
            data_attributes=data.get("data_attributes", {}),
            semantic_role=data.get("semantic_role"),
            aria_label=data.get("aria_label"),
            tag_semantic=data.get("tag_semantic"),
            parent_selector=data.get("parent_selector", ""),
            parent_id=data.get("parent_id"),
            child_count=data.get("child_count", 0),
            dom_depth=data.get("dom_depth", 0),
            z_index=data.get("z_index", 0),
            is_interactive=data.get("is_interactive", False),
            is_visible=data.get("is_visible", True),
            is_container=data.get("is_container", False),
            has_visual_content=data.get("has_visual_content", False),
            extraction_category=data.get("extraction_category", ""),
        )

    def get_display_name(self) -> str:
        """Generate a display name from semantic properties."""
        if self.aria_label:
            return self.aria_label[:50]
        if self.text_content:
            return self.text_content[:50]
        if self.tag_semantic:
            return self.tag_semantic.title()
        return f"{self.tag_name}_{self.id[-4:]}"


@dataclass
class ElementFingerprint:
    """
    Visual fingerprint for cross-page element comparison.

    Contains computed properties that enable fast comparison of elements
    across different pages to find duplicates/similar elements.
    """

    element_id: str
    screenshot_id: str  # Which screenshot this came from

    # Size properties
    width: int
    height: int
    size_class: SizeClass

    # Position properties
    position_region: PositionRegion
    relative_x: float  # 0.0-1.0 position in viewport
    relative_y: float  # 0.0-1.0 position in viewport

    # Color properties
    color_histogram: list[int] = field(default_factory=list)  # 24-bin histogram
    dominant_color: tuple[int, int, int] | None = None  # RGB

    # Content properties
    content_hash: str = ""  # Hash of text + tag + key attrs
    text_length: int = 0

    # Visual properties
    visual_hash: str = ""  # Perceptual hash of element screenshot
    has_image: bool = False
    has_text: bool = False
    has_border: bool = False
    has_background: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "screenshot_id": self.screenshot_id,
            "width": self.width,
            "height": self.height,
            "size_class": self.size_class.value,
            "position_region": self.position_region.value,
            "relative_x": self.relative_x,
            "relative_y": self.relative_y,
            "color_histogram": self.color_histogram,
            "dominant_color": self.dominant_color,
            "content_hash": self.content_hash,
            "text_length": self.text_length,
            "visual_hash": self.visual_hash,
            "has_image": self.has_image,
            "has_text": self.has_text,
            "has_border": self.has_border,
            "has_background": self.has_background,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ElementFingerprint":
        return cls(
            element_id=data["element_id"],
            screenshot_id=data["screenshot_id"],
            width=data["width"],
            height=data["height"],
            size_class=SizeClass(data["size_class"]),
            position_region=PositionRegion(data["position_region"]),
            relative_x=data["relative_x"],
            relative_y=data["relative_y"],
            color_histogram=data.get("color_histogram", []),
            dominant_color=tuple(data["dominant_color"]) if data.get("dominant_color") else None,
            content_hash=data.get("content_hash", ""),
            text_length=data.get("text_length", 0),
            visual_hash=data.get("visual_hash", ""),
            has_image=data.get("has_image", False),
            has_text=data.get("has_text", False),
            has_border=data.get("has_border", False),
            has_background=data.get("has_background", False),
        )

    def quick_match(self, other: "ElementFingerprint") -> bool:
        """Quick check if elements could be similar (for filtering before detailed comparison)."""
        # Must be same size class
        if self.size_class != other.size_class:
            return False
        # Size must be within 20%
        width_ratio = (
            min(self.width, other.width) / max(self.width, other.width)
            if max(self.width, other.width) > 0
            else 0
        )
        height_ratio = (
            min(self.height, other.height) / max(self.height, other.height)
            if max(self.height, other.height) > 0
            else 0
        )
        if width_ratio < 0.8 or height_ratio < 0.8:
            return False
        return True


@dataclass
class ElementFunction:
    """
    Element function data for transition detection.

    Captures what an element does when interacted with, enabling
    automatic transition identification in the state machine.
    """

    element_id: str
    function_type: FunctionType

    # Navigation properties
    href: str | None = None
    target_url: str | None = None  # Resolved URL
    target_state_id: str | None = None  # Identified target state (computed later)

    # Form properties
    form_action: str | None = None
    form_method: str | None = None
    form_id: str | None = None

    # Toggle/expand properties
    aria_controls: str | None = None  # ID of controlled element
    aria_expanded: str | None = None  # Current state

    # JavaScript properties
    onclick: str | None = None
    event_listeners: list[str] = field(default_factory=list)

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "function_type": self.function_type.value,
            "href": self.href,
            "target_url": self.target_url,
            "target_state_id": self.target_state_id,
            "form_action": self.form_action,
            "form_method": self.form_method,
            "form_id": self.form_id,
            "aria_controls": self.aria_controls,
            "aria_expanded": self.aria_expanded,
            "onclick": self.onclick,
            "event_listeners": self.event_listeners,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ElementFunction":
        return cls(
            element_id=data["element_id"],
            function_type=FunctionType(data["function_type"]),
            href=data.get("href"),
            target_url=data.get("target_url"),
            target_state_id=data.get("target_state_id"),
            form_action=data.get("form_action"),
            form_method=data.get("form_method"),
            form_id=data.get("form_id"),
            aria_controls=data.get("aria_controls"),
            aria_expanded=data.get("aria_expanded"),
            onclick=data.get("onclick"),
            event_listeners=data.get("event_listeners", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SemanticRegion:
    """
    Semantic region detected on a page.

    Kept for naming purposes only (not for state detection).
    Uses existing RegionDetector results.
    """

    id: str
    name: str
    bbox: BoundingBox
    region_type: str  # nav, header, sidebar, etc.
    aria_label: str | None = None
    element_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "bbox": self.bbox.to_dict(),
            "region_type": self.region_type,
            "aria_label": self.aria_label,
            "element_count": self.element_count,
        }


@dataclass
class ExtractedPageV2:
    """
    Enhanced page extraction result for deterministic state machine generation.

    Contains all elements (after containment filtering), fingerprints,
    element functions, and semantic regions for a single page.
    """

    url: str
    title: str
    screenshot_id: str
    viewport: tuple[int, int]
    extracted_at: datetime = field(default_factory=datetime.now)

    # All elements after containment filtering (atomic elements only)
    elements: list[RawElement] = field(default_factory=list)

    # Element fingerprints for cross-page comparison
    fingerprints: list[ElementFingerprint] = field(default_factory=list)

    # Element functions for transition detection
    element_functions: list[ElementFunction] = field(default_factory=list)

    # Semantic regions (for naming, not detection)
    semantic_regions: list[SemanticRegion] = field(default_factory=list)

    # State identification (computed by StateIdentifier)
    state_signature: str = ""  # Hash of element fingerprint set

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "screenshot_id": self.screenshot_id,
            "viewport": list(self.viewport),
            "extracted_at": self.extracted_at.isoformat(),
            "elements": [e.to_dict() for e in self.elements],
            "fingerprints": [f.to_dict() for f in self.fingerprints],
            "element_functions": [ef.to_dict() for ef in self.element_functions],
            "semantic_regions": [sr.to_dict() for sr in self.semantic_regions],
            "state_signature": self.state_signature,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractedPageV2":
        return cls(
            url=data["url"],
            title=data["title"],
            screenshot_id=data["screenshot_id"],
            viewport=tuple(data["viewport"]),
            extracted_at=(
                datetime.fromisoformat(data["extracted_at"])
                if data.get("extracted_at")
                else datetime.now()
            ),
            elements=[RawElement.from_dict(e) for e in data.get("elements", [])],
            fingerprints=[ElementFingerprint.from_dict(f) for f in data.get("fingerprints", [])],
            element_functions=[
                ElementFunction.from_dict(ef) for ef in data.get("element_functions", [])
            ],
            semantic_regions=[],  # TODO: Add from_dict for SemanticRegion
            state_signature=data.get("state_signature", ""),
        )


@dataclass
class IdentifiedState:
    """
    A state identified algorithmically from element compositions.

    States are defined by which elements are present, not by semantic HTML.
    Two pages with the same element composition belong to the same state.
    """

    id: str  # Hash of element composition
    name: str  # Generated from semantic cues
    element_ids: list[str]  # Elements present in this state
    page_urls: list[str]  # URLs where this state appears
    screenshot_ids: list[str]  # Screenshots of this state

    # For comparison
    element_fingerprint_hashes: frozenset[str] = field(default_factory=frozenset)

    # Metadata
    detection_method: str = "element_composition"  # Always algorithmic
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "element_ids": self.element_ids,
            "page_urls": self.page_urls,
            "screenshot_ids": self.screenshot_ids,
            "element_fingerprint_hashes": list(self.element_fingerprint_hashes),
            "detection_method": self.detection_method,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class IdentifiedTransition:
    """
    A transition identified from element function data.

    Created by matching element functions (href, form action) to state changes.
    """

    id: str
    from_state_id: str
    to_state_id: str
    trigger_element_id: str
    trigger_type: str  # click, submit, etc.

    # Element function data
    element_function: ElementFunction | None = None

    # Navigation details
    source_url: str = ""
    target_url: str = ""

    # Metadata
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "from_state_id": self.from_state_id,
            "to_state_id": self.to_state_id,
            "trigger_element_id": self.trigger_element_id,
            "trigger_type": self.trigger_type,
            "element_function": self.element_function.to_dict() if self.element_function else None,
            "source_url": self.source_url,
            "target_url": self.target_url,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
