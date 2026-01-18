"""
Test configuration for AWAS module tests.

Creates a stub qontinui package to avoid loading the full package
with heavy dependencies (cv2, torch, etc.).
"""

import sys
import types
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Create stub for qontinui package before any imports happen
# This prevents the actual qontinui/__init__.py from being loaded
if "qontinui" not in sys.modules:
    # Create a stub module
    stub = types.ModuleType("qontinui")
    stub.__path__ = [str(Path(__file__).parent.parent.parent / "src" / "qontinui")]
    stub.__file__ = str(
        Path(__file__).parent.parent.parent / "src" / "qontinui" / "__init__.py"
    )
    sys.modules["qontinui"] = stub


# Create stubs for extraction module to avoid heavy imports
def _create_stub_module(name: str, parent_path: str) -> types.ModuleType:
    """Create a stub module."""
    mod = types.ModuleType(name)
    mod.__path__ = [parent_path]
    mod.__file__ = f"{parent_path}/__init__.py"
    return mod


# Stub the extraction chain that would trigger cv2/torch imports
_src_path = str(Path(__file__).parent.parent.parent / "src" / "qontinui")


# Stub extraction modules at the top level to prevent import issues
# These need to exist before the actual imports try to load them
def _setup_extraction_stubs():
    """Set up stub modules for extraction to avoid heavy deps."""
    # Only set up if extraction isn't already loaded
    if "qontinui.extraction" in sys.modules:
        return

    # Create stub modules for problematic import chain
    extraction = _create_stub_module(
        "qontinui.extraction", f"{_src_path}/extraction"
    )
    sys.modules["qontinui.extraction"] = extraction

    # Create web.models stubs
    extraction_web = _create_stub_module(
        "qontinui.extraction.web", f"{_src_path}/extraction/web"
    )
    sys.modules["qontinui.extraction.web"] = extraction_web

    # Create stub classes that the extractor needs

    class StateType(str, Enum):
        PAGE = "page"
        MODAL = "modal"
        MENU = "menu"
        FORM = "form"
        LIST = "list"
        DETAIL = "detail"

    class ElementType(str, Enum):
        BUTTON = "button"
        LINK = "link"
        INPUT = "input"
        SELECT = "select"
        CHECKBOX = "checkbox"

    class TransitionType(str, Enum):
        CLICK = "click"
        NAVIGATE = "navigate"
        SUBMIT = "submit"

    @dataclass
    class BoundingBox:
        x: int = 0
        y: int = 0
        width: int = 0
        height: int = 0

        def to_dict(self) -> dict[str, Any]:
            return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @dataclass
    class ExtractedElement:
        id: str
        tag_name: str = ""
        element_type: str = "button"
        text: str = ""
        bbox: BoundingBox = field(default_factory=BoundingBox)
        is_visible: bool = True
        is_interactive: bool = True
        selector: str = ""
        computed_role: str = ""
        confidence: float = 1.0
        extraction_method: str = ""
        metadata: dict[str, Any] = field(default_factory=dict)

        def to_dict(self) -> dict[str, Any]:
            return {
                "id": self.id,
                "tag_name": self.tag_name,
                "element_type": self.element_type,
                "text": self.text,
                "bbox": self.bbox.to_dict(),
                "is_visible": self.is_visible,
                "is_interactive": self.is_interactive,
                "selector": self.selector,
                "computed_role": self.computed_role,
                "confidence": self.confidence,
                "extraction_method": self.extraction_method,
                "metadata": self.metadata,
            }

    @dataclass
    class ExtractedState:
        id: str
        name: str
        bbox: BoundingBox = field(default_factory=BoundingBox)
        state_type: StateType = StateType.PAGE
        element_ids: list[str] = field(default_factory=list)
        screenshot_id: str | None = None
        detection_method: str = ""
        confidence: float = 1.0
        source_url: str | None = None
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
                "source_url": self.source_url,
                "metadata": self.metadata,
            }

    @dataclass
    class ExtractedTransition:
        id: str
        from_state: str
        to_state: str
        transition_type: TransitionType = TransitionType.CLICK
        element_id: str | None = None
        metadata: dict[str, Any] = field(default_factory=dict)

        def to_dict(self) -> dict[str, Any]:
            return {
                "id": self.id,
                "from_state": self.from_state,
                "to_state": self.to_state,
                "transition_type": self.transition_type.value,
                "element_id": self.element_id,
                "metadata": self.metadata,
            }

    # Create models module
    extraction_web_models = types.ModuleType("qontinui.extraction.web.models")
    extraction_web_models.BoundingBox = BoundingBox
    extraction_web_models.ExtractedElement = ExtractedElement
    extraction_web_models.ExtractedState = ExtractedState
    extraction_web_models.ExtractedTransition = ExtractedTransition
    extraction_web_models.StateType = StateType
    extraction_web_models.ElementType = ElementType
    extraction_web_models.TransitionType = TransitionType
    sys.modules["qontinui.extraction.web.models"] = extraction_web_models

    # Create models.base stubs
    extraction_models = _create_stub_module(
        "qontinui.extraction.models", f"{_src_path}/extraction/models"
    )
    sys.modules["qontinui.extraction.models"] = extraction_models

    @dataclass
    class Viewport:
        width: int = 1920
        height: int = 1080

    @dataclass
    class Screenshot:
        id: str
        path: Path
        viewport: Viewport = field(default_factory=Viewport)
        metadata: dict[str, Any] = field(default_factory=dict)

    extraction_models_base = types.ModuleType("qontinui.extraction.models.base")
    extraction_models_base.BoundingBox = BoundingBox
    extraction_models_base.Screenshot = Screenshot
    extraction_models_base.Viewport = Viewport
    sys.modules["qontinui.extraction.models.base"] = extraction_models_base

    # Create runtime stubs
    extraction_runtime = _create_stub_module(
        "qontinui.extraction.runtime", f"{_src_path}/extraction/runtime"
    )
    sys.modules["qontinui.extraction.runtime"] = extraction_runtime

    # Runtime types
    class RuntimeType(Enum):
        WEB = "web"
        TAURI = "tauri"
        ELECTRON = "electron"
        NATIVE = "native"

    @dataclass
    class ExtractionTarget:
        runtime_type: RuntimeType
        url: str | None = None
        app_path: str | None = None
        app_dev_command: str | None = None
        viewport: tuple[int, int] = (1920, 1080)
        headless: bool = True
        auth_cookies: dict[str, str] = field(default_factory=dict)
        auth_headers: dict[str, str] = field(default_factory=dict)
        tauri_config_path: str | None = None
        tauri_mocks: dict[str, Any] = field(default_factory=dict)
        metadata: dict[str, Any] = field(default_factory=dict)

        def to_dict(self) -> dict[str, Any]:
            return {
                "runtime_type": self.runtime_type.value,
                "url": self.url,
                "app_path": self.app_path,
                "viewport": list(self.viewport),
                "metadata": self.metadata,
            }

    @dataclass
    class RuntimeStateCapture:
        capture_id: str
        timestamp: datetime = field(default_factory=datetime.now)
        elements: list[ExtractedElement] = field(default_factory=list)
        states: list[ExtractedState] = field(default_factory=list)
        screenshot_path: Path | None = None
        url: str | None = None
        title: str | None = None
        viewport: tuple[int, int] = (1920, 1080)
        scroll_position: tuple[int, int] = (0, 0)
        metadata: dict[str, Any] = field(default_factory=dict)

        def to_dict(self) -> dict[str, Any]:
            return {
                "capture_id": self.capture_id,
                "elements": [e.to_dict() for e in self.elements],
                "states": [s.to_dict() for s in self.states],
                "url": self.url,
                "metadata": self.metadata,
            }

    @dataclass
    class RuntimeExtractionSession:
        session_id: str
        target: ExtractionTarget
        started_at: datetime = field(default_factory=datetime.now)
        completed_at: datetime | None = None
        captures: list[RuntimeStateCapture] = field(default_factory=list)
        transitions: list[ExtractedTransition] = field(default_factory=list)
        storage_dir: Path | None = None
        metadata: dict[str, Any] = field(default_factory=dict)

    extraction_runtime_types = types.ModuleType("qontinui.extraction.runtime.types")
    extraction_runtime_types.RuntimeType = RuntimeType
    extraction_runtime_types.ExtractionTarget = ExtractionTarget
    extraction_runtime_types.RuntimeStateCapture = RuntimeStateCapture
    extraction_runtime_types.RuntimeExtractionSession = RuntimeExtractionSession
    extraction_runtime_types.BoundingBox = BoundingBox
    extraction_runtime_types.ExtractedElement = ExtractedElement
    extraction_runtime_types.ExtractedState = ExtractedState
    extraction_runtime_types.ExtractedTransition = ExtractedTransition
    extraction_runtime_types.StateType = StateType
    extraction_runtime_types.ElementType = ElementType
    extraction_runtime_types.TransitionType = TransitionType
    sys.modules["qontinui.extraction.runtime.types"] = extraction_runtime_types

    # Runtime base stubs
    from abc import ABC, abstractmethod

    @dataclass
    class DetectedRegion:
        id: str
        name: str
        bbox: BoundingBox
        region_type: str
        confidence: float = 1.0
        metadata: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class InteractionAction:
        action_type: str
        target_selector: str | None = None
        target_element_id: str | None = None
        action_value: str | None = None
        metadata: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class StateChange:
        appeared_elements: list[str] = field(default_factory=list)
        disappeared_elements: list[str] = field(default_factory=list)
        modified_elements: list[str] = field(default_factory=list)
        appeared_states: list[str] = field(default_factory=list)
        disappeared_states: list[str] = field(default_factory=list)
        url_changed: bool = False
        new_url: str | None = None
        screenshot_before: str | None = None
        screenshot_after: str | None = None
        metadata: dict[str, Any] = field(default_factory=dict)

    @dataclass
    class RuntimeExtractionResult:
        session_id: str
        target: ExtractionTarget
        captures: list[RuntimeStateCapture] = field(default_factory=list)
        transitions: list[ExtractedTransition] = field(default_factory=list)
        metadata: dict[str, Any] = field(default_factory=dict)

    class RuntimeExtractor(ABC):
        @abstractmethod
        async def connect(self, target: ExtractionTarget) -> None:
            pass

        @abstractmethod
        async def extract_current_state(self) -> RuntimeStateCapture:
            pass

        @abstractmethod
        async def extract_elements(self) -> list[ExtractedElement]:
            pass

        @abstractmethod
        async def detect_regions(self) -> list[DetectedRegion]:
            pass

        @abstractmethod
        async def capture_screenshot(
            self, region: BoundingBox | None = None
        ) -> Screenshot:
            pass

        @abstractmethod
        async def navigate_to_route(self, route: str) -> None:
            pass

        @abstractmethod
        async def simulate_interaction(self, action: InteractionAction) -> StateChange:
            pass

        @abstractmethod
        async def disconnect(self) -> None:
            pass

        @classmethod
        @abstractmethod
        def supports_target(cls, target: ExtractionTarget) -> bool:
            pass

    extraction_runtime_base = types.ModuleType("qontinui.extraction.runtime.base")
    extraction_runtime_base.DetectedRegion = DetectedRegion
    extraction_runtime_base.InteractionAction = InteractionAction
    extraction_runtime_base.StateChange = StateChange
    extraction_runtime_base.RuntimeExtractionResult = RuntimeExtractionResult
    extraction_runtime_base.RuntimeExtractor = RuntimeExtractor
    sys.modules["qontinui.extraction.runtime.base"] = extraction_runtime_base

    # Set up AWAS runtime module
    extraction_runtime_awas = _create_stub_module(
        "qontinui.extraction.runtime.awas",
        f"{_src_path}/extraction/runtime/awas",
    )
    sys.modules["qontinui.extraction.runtime.awas"] = extraction_runtime_awas


# Set up stubs before imports
_setup_extraction_stubs()
