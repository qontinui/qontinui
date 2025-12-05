"""
Base classes for hybrid extraction.

Defines the core abstractions for States, StateImages, Transitions, and
the TechStackExtractor interface that enables pluggable tech stack support.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class StateType(Enum):
    """Type of state in the GUI state machine."""

    PAGE = "page"  # Full page/route state
    MODAL = "modal"  # Modal dialog state
    DROPDOWN = "dropdown"  # Dropdown/menu state
    SIDEBAR = "sidebar"  # Sidebar state (open/closed)
    TOAST = "toast"  # Toast notification state
    TOOLTIP = "tooltip"  # Tooltip state
    PANEL = "panel"  # Collapsible panel state
    TAB = "tab"  # Tab selection state
    LOADING = "loading"  # Loading state
    ERROR = "error"  # Error state
    EMPTY = "empty"  # Empty state
    CUSTOM = "custom"  # Custom visibility-based state


class TransitionTrigger(Enum):
    """What triggers a state transition."""

    CLICK = "click"
    HOVER = "hover"
    FOCUS = "focus"
    BLUR = "blur"
    KEY_PRESS = "key_press"
    SCROLL = "scroll"
    SUBMIT = "submit"
    NAVIGATION = "navigation"
    TIMER = "timer"
    API_RESPONSE = "api_response"
    STATE_CHANGE = "state_change"


@dataclass
class BoundingBox:
    """Precise bounding box for a UI element."""

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

    def contains(self, other: "BoundingBox") -> bool:
        return (
            self.x <= other.x and self.y <= other.y and self.x2 >= other.x2 and self.y2 >= other.y2
        )

    def intersects(self, other: "BoundingBox") -> bool:
        return not (
            self.x2 <= other.x or other.x2 <= self.x or self.y2 <= other.y or other.y2 <= self.y
        )


@dataclass
class ImagePattern:
    """A visual pattern for a StateImage (different visual states)."""

    id: str
    name: str  # e.g., "normal", "hover", "clicked", "disabled"
    pixel_data_path: Path  # Path to the cropped image file
    pixel_hash: str  # Hash for quick comparison

    # Extracted from runtime
    bbox: BoundingBox  # Where this pattern was captured

    # Metadata
    captured_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StateImage:
    """
    A visual pattern that identifies UI elements within a state.

    StateImages are the visual fingerprints of interactive elements.
    They can have multiple patterns (normal, hover, clicked, etc.)
    and are used to locate elements on screen during testing/automation.
    """

    # Required fields first
    id: str
    name: str  # e.g., "LoginButton", "SubmitForm", "MenuIcon"
    element_type: str  # "button", "input", "link", "icon", etc.

    # Source information (from static analysis)
    component_name: str | None = None
    source_file: Path | None = None
    source_line: int | None = None
    selector: str | None = None  # CSS selector from code

    # Interactivity
    is_interactive: bool = True

    # Visual patterns (from runtime extraction)
    patterns: list[ImagePattern] = field(default_factory=list)

    # Search region (where to look for this pattern on screen)
    search_region: BoundingBox | None = None

    # Text content (if applicable)
    text_content: str | None = None
    has_dynamic_text: bool = False

    # Semantic information
    aria_label: str | None = None
    semantic_role: str | None = None

    # Confidence and stability
    stability_score: float = 1.0  # How stable is this pattern across captures
    confidence: float = 1.0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_primary_pattern(self) -> ImagePattern | None:
        """Get the primary (normal) pattern."""
        for pattern in self.patterns:
            if pattern.name == "normal":
                return pattern
        return self.patterns[0] if self.patterns else None

    def get_bbox(self) -> BoundingBox | None:
        """Get the bounding box from the primary pattern."""
        pattern = self.get_primary_pattern()
        return pattern.bbox if pattern else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "component_name": self.component_name,
            "source_file": str(self.source_file) if self.source_file else None,
            "source_line": self.source_line,
            "selector": self.selector,
            "element_type": self.element_type,
            "is_interactive": self.is_interactive,
            "patterns": [
                {
                    "id": p.id,
                    "name": p.name,
                    "pixel_data_path": str(p.pixel_data_path),
                    "pixel_hash": p.pixel_hash,
                    "bbox": p.bbox.to_dict(),
                    "captured_at": p.captured_at.isoformat(),
                }
                for p in self.patterns
            ],
            "search_region": self.search_region.to_dict() if self.search_region else None,
            "text_content": self.text_content,
            "has_dynamic_text": self.has_dynamic_text,
            "aria_label": self.aria_label,
            "semantic_role": self.semantic_role,
            "stability_score": self.stability_score,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class State:
    """
    A visual screen configuration in the GUI state machine.

    A State represents a specific visual configuration of the application.
    It's identified by the presence/absence of StateImages and can have
    parent-child relationships (e.g., modal state is a child of page state).
    """

    id: str
    name: str  # e.g., "LoginPage", "Dashboard_ModalOpen", "Settings_SidebarExpanded"
    state_type: StateType

    # Source information (from static analysis)
    route_path: str | None = None  # For page states
    component_name: str | None = None
    source_file: Path | None = None
    source_line: int | None = None

    # Controlling state (for visibility-based sub-states)
    controlling_variable: str | None = None  # State variable that controls visibility
    controlling_value: Any = None  # Value that activates this state

    # Visual identification (from runtime extraction)
    state_images: list[StateImage] = field(default_factory=list)
    screenshot_path: Path | None = None
    viewport: tuple[int, int] = (1920, 1080)

    # Hierarchy
    parent_state_id: str | None = None
    child_state_ids: list[str] = field(default_factory=list)

    # URL (for page states)
    url: str | None = None

    # Confidence
    confidence: float = 1.0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_interactive_images(self) -> list[StateImage]:
        """Get all interactive StateImages in this state."""
        return [img for img in self.state_images if img.is_interactive]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "state_type": self.state_type.value,
            "route_path": self.route_path,
            "component_name": self.component_name,
            "source_file": str(self.source_file) if self.source_file else None,
            "source_line": self.source_line,
            "controlling_variable": self.controlling_variable,
            "controlling_value": self.controlling_value,
            "state_images": [img.to_dict() for img in self.state_images],
            "screenshot_path": str(self.screenshot_path) if self.screenshot_path else None,
            "viewport": list(self.viewport),
            "parent_state_id": self.parent_state_id,
            "child_state_ids": self.child_state_ids,
            "url": self.url,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class StateTransition:
    """
    A transition between states in the GUI state machine.

    Transitions connect states and are triggered by user actions.
    """

    id: str
    from_state_id: str
    to_state_id: str
    trigger: TransitionTrigger

    # Trigger element (the StateImage that triggers this transition)
    trigger_image_id: str | None = None
    trigger_selector: str | None = None

    # Source information (from static analysis)
    event_handler_name: str | None = None
    source_file: Path | None = None
    source_line: int | None = None

    # Action details
    action_value: str | None = None  # For type actions, the text typed
    key_modifiers: list[str] = field(default_factory=list)

    # Navigation
    navigation_path: str | None = None  # For navigation transitions

    # Confidence
    confidence: float = 1.0

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "from_state_id": self.from_state_id,
            "to_state_id": self.to_state_id,
            "trigger": self.trigger.value,
            "trigger_image_id": self.trigger_image_id,
            "trigger_selector": self.trigger_selector,
            "event_handler_name": self.event_handler_name,
            "source_file": str(self.source_file) if self.source_file else None,
            "source_line": self.source_line,
            "action_value": self.action_value,
            "key_modifiers": self.key_modifiers,
            "navigation_path": self.navigation_path,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class HybridExtractionResult:
    """Complete result of hybrid extraction."""

    extraction_id: str
    tech_stack: str  # e.g., "tauri-typescript", "next-js", "flutter"

    # Extracted data
    states: list[State] = field(default_factory=list)
    state_images: list[StateImage] = field(default_factory=list)
    transitions: list[StateTransition] = field(default_factory=list)

    # Screenshots directory
    screenshots_dir: Path | None = None

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    # Errors and warnings
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_state(self, state_id: str) -> State | None:
        """Get a state by ID."""
        for state in self.states:
            if state.id == state_id:
                return state
        return None

    def get_state_image(self, image_id: str) -> StateImage | None:
        """Get a StateImage by ID."""
        for image in self.state_images:
            if image.id == image_id:
                return image
        return None

    def get_transitions_from(self, state_id: str) -> list[StateTransition]:
        """Get all transitions from a state."""
        return [t for t in self.transitions if t.from_state_id == state_id]

    def get_transitions_to(self, state_id: str) -> list[StateTransition]:
        """Get all transitions to a state."""
        return [t for t in self.transitions if t.to_state_id == state_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "extraction_id": self.extraction_id,
            "tech_stack": self.tech_stack,
            "states": [s.to_dict() for s in self.states],
            "state_images": [img.to_dict() for img in self.state_images],
            "transitions": [t.to_dict() for t in self.transitions],
            "screenshots_dir": str(self.screenshots_dir) if self.screenshots_dir else None,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


@dataclass
class HybridExtractionConfig:
    """Configuration for hybrid extraction."""

    # Project paths
    project_path: Path
    frontend_path: Path | None = None  # For monorepos

    # Runtime configuration
    dev_command: str | None = None  # Command to start dev server
    dev_url: str = "http://localhost:3000"
    viewport: tuple[int, int] = (1920, 1080)
    headless: bool = True

    # Extraction settings
    capture_hover_states: bool = True
    capture_focus_states: bool = True
    max_states: int = 100
    timeout_seconds: int = 300

    # Output
    output_dir: Path | None = None

    # Framework hints
    framework_hints: dict[str, Any] = field(default_factory=dict)


class TechStackExtractor(ABC):
    """
    Abstract base class for tech stack-specific extractors.

    Implement this interface to add support for a new tech stack.
    Each tech stack extractor combines:
    1. Static analysis of the codebase
    2. Runtime extraction via the app's dev server
    3. Correlation to produce States, StateImages, and Transitions
    """

    @property
    @abstractmethod
    def tech_stack_name(self) -> str:
        """
        The name of this tech stack.

        Examples: "tauri-typescript", "next-js", "flutter", "electron-react"
        """
        pass

    @classmethod
    @abstractmethod
    def can_handle(cls, project_path: Path) -> bool:
        """
        Check if this extractor can handle the given project.

        Args:
            project_path: Root directory of the project

        Returns:
            True if this extractor can analyze the project
        """
        pass

    @classmethod
    @abstractmethod
    def detect_config(cls, project_path: Path) -> HybridExtractionConfig | None:
        """
        Auto-detect configuration from the project.

        Examines project files to determine:
        - Frontend path (for monorepos)
        - Dev command to run
        - Dev server URL
        - Framework-specific hints

        Args:
            project_path: Root directory of the project

        Returns:
            HybridExtractionConfig if detection succeeds, None otherwise
        """
        pass

    @abstractmethod
    async def extract(self, config: HybridExtractionConfig) -> HybridExtractionResult:
        """
        Perform hybrid extraction.

        This is the main entry point that:
        1. Runs static analysis on the codebase
        2. Starts the dev server
        3. Connects via Playwright/automation
        4. Extracts runtime state with bounding boxes
        5. Correlates static + runtime data
        6. Produces States, StateImages, and Transitions

        Args:
            config: Extraction configuration

        Returns:
            HybridExtractionResult with all extracted data
        """
        pass

    @abstractmethod
    async def extract_state_images_for_selector(
        self,
        selector: str,
        capture_hover: bool = True,
        capture_focus: bool = True,
    ) -> list[StateImage]:
        """
        Extract StateImage patterns for a specific CSS selector.

        Captures the element in different states (normal, hover, focus)
        and returns StateImage objects with precise bounding boxes.

        Args:
            selector: CSS selector for the element
            capture_hover: Whether to capture hover state
            capture_focus: Whether to capture focus state

        Returns:
            List of StateImage objects with captured patterns
        """
        pass


class HybridExtractor:
    """
    Main entry point for hybrid extraction.

    Automatically selects the appropriate TechStackExtractor based on
    the project structure.
    """

    def __init__(self):
        from .registry import TechStackRegistry

        self.registry = TechStackRegistry()

    async def extract(
        self,
        project_path: Path,
        config: HybridExtractionConfig | None = None,
    ) -> HybridExtractionResult:
        """
        Extract States, StateImages, and Transitions from a project.

        Args:
            project_path: Root directory of the project
            config: Optional extraction configuration (auto-detected if not provided)

        Returns:
            HybridExtractionResult with all extracted data

        Raises:
            ValueError: If no suitable extractor is found for the project
        """
        # Find the right extractor
        extractor_class = self.registry.get_extractor_for(project_path)
        if not extractor_class:
            raise ValueError(
                f"No suitable extractor found for project at {project_path}. "
                f"Available extractors: {self.registry.list_extractors()}"
            )

        # Auto-detect config if not provided
        if config is None:
            config = extractor_class.detect_config(project_path)
            if config is None:
                config = HybridExtractionConfig(project_path=project_path)

        # Create extractor and run extraction
        extractor = extractor_class()
        return await extractor.extract(config)
