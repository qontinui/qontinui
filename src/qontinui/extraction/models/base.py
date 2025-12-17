"""
Shared base models for extraction architecture.

These models are used across static analysis, runtime extraction, and
correlation phases.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when extraction configuration is invalid or incomplete."""

    pass


class FrameworkType(Enum):
    """Supported application frameworks."""

    # Web
    REACT = "react"
    NEXT = "next"
    REMIX = "remix"
    VUE = "vue"
    NUXT = "nuxt"
    SVELTE = "svelte"
    SVELTE_KIT = "svelte_kit"
    ANGULAR = "angular"
    SOLID = "solid"
    ASTRO = "astro"

    # Desktop
    ELECTRON = "electron"
    TAURI = "tauri"

    # Mobile
    FLUTTER = "flutter"
    REACT_NATIVE = "react_native"

    # Generic
    WEB = "web"
    DESKTOP = "desktop"
    MOBILE = "mobile"
    UNKNOWN = "unknown"


class ExtractionMode(Enum):
    """Extraction mode determines what analysis is performed."""

    # Static only - analyze source code without running
    STATIC_ONLY = "static_only"

    # Runtime only - black box extraction without source code
    BLACK_BOX = "black_box"

    # Both static and runtime - correlate and verify
    WHITE_BOX = "white_box"

    # Hybrid extraction - combined static analysis + runtime with precise bounding boxes
    # Uses tech stack-specific extractors (e.g., TauriTypeScriptExtractor)
    HYBRID = "hybrid"


class OutputFormat(Enum):
    """Format for extraction output."""

    STATE_STRUCTURE = "state_structure"
    STATE_GRAPH = "state_graph"
    TRAINING_DATA = "training_data"


@dataclass
class BoundingBox:
    """Bounding box for an element or region."""

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        """Right edge x coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom edge y coordinate."""
        return self.y + self.height

    @property
    def center(self) -> tuple[int, int]:
        """Center point of the bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """Area of the bounding box in pixels."""
        return self.width * self.height

    def intersects(self, other: "BoundingBox") -> bool:
        """Check if this box intersects with another."""
        return not (
            self.x2 <= other.x
            or other.x2 <= self.x
            or self.y2 <= other.y
            or other.y2 <= self.y
        )

    def contains(self, other: "BoundingBox") -> bool:
        """Check if this box completely contains another."""
        return (
            self.x <= other.x
            and self.y <= other.y
            and self.x2 >= other.x2
            and self.y2 >= other.y2
        )

    def intersection_area(self, other: "BoundingBox") -> int:
        """Calculate the intersection area with another box."""
        if not self.intersects(other):
            return 0

        x_overlap = min(self.x2, other.x2) - max(self.x, other.x)
        y_overlap = min(self.y2, other.y2) - max(self.y, other.y)
        return x_overlap * y_overlap

    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union (IoU) with another box."""
        intersection = self.intersection_area(other)
        if intersection == 0:
            return 0.0

        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary representation."""
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "BoundingBox":
        """Create from dictionary representation."""
        return cls(x=data["x"], y=data["y"], width=data["width"], height=data["height"])


@dataclass
class Viewport:
    """Viewport/screen dimensions."""

    width: int
    height: int
    scale_factor: float = 1.0

    def to_tuple(self) -> tuple[int, int]:
        """Convert to (width, height) tuple."""
        return (self.width, self.height)


@dataclass
class Screenshot:
    """Reference to a captured screenshot."""

    id: str
    path: Path
    viewport: Viewport
    thumbnail_path: Path | None = None
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "path": str(self.path),
            "viewport": {
                "width": self.viewport.width,
                "height": self.viewport.height,
                "scale_factor": self.viewport.scale_factor,
            },
            "thumbnail_path": str(self.thumbnail_path) if self.thumbnail_path else None,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Screenshot":
        """Create from dictionary representation."""
        viewport_data = data["viewport"]
        viewport = Viewport(
            width=viewport_data["width"],
            height=viewport_data["height"],
            scale_factor=viewport_data.get("scale_factor", 1.0),
        )

        return cls(
            id=data["id"],
            path=Path(data["path"]),
            viewport=viewport,
            thumbnail_path=(
                Path(data["thumbnail_path"]) if data.get("thumbnail_path") else None
            ),
            timestamp=data.get("timestamp", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExtractionTarget:
    """Target application to extract from."""

    # Project location
    project_path: Path | None = None

    # Runtime access
    url: str | None = None
    executable_path: Path | None = None
    app_id: str | None = None  # For mobile apps

    # Framework hint (optional - will be auto-detected)
    framework: FrameworkType | None = None

    # Authentication
    auth_cookies: dict[str, str] = field(default_factory=dict)
    auth_headers: dict[str, str] = field(default_factory=dict)
    login_url: str | None = None

    def validate(self, mode: ExtractionMode) -> None:
        """Validate target has required fields for the extraction mode."""
        if mode == ExtractionMode.STATIC_ONLY:
            if not self.project_path:
                raise ConfigError("STATIC_ONLY mode requires project_path")
            if not self.project_path.exists():
                raise ConfigError(f"Project path does not exist: {self.project_path}")

        elif mode == ExtractionMode.BLACK_BOX:
            if not (self.url or self.executable_path or self.app_id):
                raise ConfigError(
                    "BLACK_BOX mode requires url, executable_path, or app_id"
                )

        elif mode == ExtractionMode.WHITE_BOX:
            if not self.project_path:
                raise ConfigError("WHITE_BOX mode requires project_path")
            if not self.project_path.exists():
                raise ConfigError(f"Project path does not exist: {self.project_path}")
            if not (self.url or self.executable_path or self.app_id):
                raise ConfigError(
                    "WHITE_BOX mode requires url, executable_path, or app_id for runtime extraction"
                )


@dataclass
class ExtractionConfig:
    """Configuration for extraction process."""

    # Target and mode
    target: ExtractionTarget
    mode: ExtractionMode = ExtractionMode.WHITE_BOX

    # Output
    output_dir: Path | None = None

    # Runtime extraction settings
    viewports: list[tuple[int, int]] = field(
        default_factory=lambda: [(1920, 1080), (768, 1024), (375, 667)]
    )
    capture_hover_states: bool = True
    capture_focus_states: bool = True
    capture_scroll_states: bool = True
    max_interaction_depth: int = 3  # How many levels of interactions to explore

    # Static analysis settings
    include_tests: bool = False
    include_node_modules: bool = False

    # White-box settings
    correlation_threshold: float = (
        0.8  # Similarity threshold for matching static/runtime
    )
    require_correlation: bool = True  # Fail if correlation is low

    # Performance
    parallel_workers: int = 1
    timeout_seconds: int = 300

    def validate(self) -> None:
        """Validate configuration is complete and consistent."""
        # Validate target for mode
        self.target.validate(self.mode)

        # Validate thresholds
        if not 0.0 <= self.correlation_threshold <= 1.0:
            raise ConfigError("correlation_threshold must be between 0.0 and 1.0")

        if self.max_interaction_depth < 0:
            raise ConfigError("max_interaction_depth must be non-negative")

        if self.timeout_seconds <= 0:
            raise ConfigError("timeout_seconds must be positive")


@dataclass
class StaticAnalysisResult:
    """Results from static code analysis."""

    framework: FrameworkType
    components: list[dict[str, Any]] = field(default_factory=list)
    routes: list[dict[str, Any]] = field(default_factory=list)
    state_definitions: list[dict[str, Any]] = field(default_factory=list)
    event_handlers: list[dict[str, Any]] = field(default_factory=list)
    navigation_flows: list[dict[str, Any]] = field(default_factory=list)

    # Metadata
    analyzed_files: int = 0
    analysis_duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class RuntimeExtractionResult:
    """Results from runtime extraction."""

    # Import from web extraction models
    elements: list[Any] = field(default_factory=list)
    states: list[Any] = field(default_factory=list)
    transitions: list[Any] = field(default_factory=list)
    screenshots: list[str] = field(default_factory=list)

    # Metadata
    pages_visited: int = 0
    extraction_duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class CorrelatedState:
    """A state that has been correlated between static and runtime analysis."""

    id: str
    name: str
    confidence: float

    # Static analysis info
    component_name: str | None = None
    route_path: str | None = None
    state_variables: list[str] = field(default_factory=list)
    source_file: str | None = None
    line_number: int | None = None

    # Runtime info
    runtime_state_id: str | None = None
    screenshot_id: str | None = None
    url: str | None = None
    visible_elements: list[str] = field(default_factory=list)

    # Correlation
    correlation_method: str = "automatic"  # automatic, manual, inferred
    correlation_score: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferredTransition:
    """A transition inferred from static or runtime analysis."""

    id: str
    from_state_id: str
    to_state_id: str
    trigger_type: str  # click, navigation, state_change, etc.

    # Static info
    event_handler: str | None = None
    source_location: str | None = None

    # Runtime info
    runtime_transition_id: str | None = None
    target_element: str | None = None

    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Complete extraction result with all phases."""

    extraction_id: str
    framework: FrameworkType
    mode: ExtractionMode

    # Phase results
    static_analysis: StaticAnalysisResult | None = None
    runtime_extraction: RuntimeExtractionResult | None = None

    # Correlated results (WHITE_BOX only)
    states: list[CorrelatedState] = field(default_factory=list)
    transitions: list[InferredTransition] = field(default_factory=list)

    # Metadata
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "extraction_id": self.extraction_id,
            "framework": self.framework.value,
            "mode": self.mode.value,
            "static_analysis": (
                {
                    "framework": self.static_analysis.framework.value,
                    "components": self.static_analysis.components,
                    "routes": self.static_analysis.routes,
                    "state_definitions": self.static_analysis.state_definitions,
                    "event_handlers": self.static_analysis.event_handlers,
                    "navigation_flows": self.static_analysis.navigation_flows,
                    "analyzed_files": self.static_analysis.analyzed_files,
                    "analysis_duration_ms": self.static_analysis.analysis_duration_ms,
                    "errors": self.static_analysis.errors,
                    "warnings": self.static_analysis.warnings,
                }
                if self.static_analysis
                else None
            ),
            "runtime_extraction": (
                {
                    "pages_visited": self.runtime_extraction.pages_visited,
                    "extraction_duration_ms": self.runtime_extraction.extraction_duration_ms,
                    "errors": self.runtime_extraction.errors,
                    "elements_count": len(self.runtime_extraction.elements),
                    "states_count": len(self.runtime_extraction.states),
                    "transitions_count": len(self.runtime_extraction.transitions),
                }
                if self.runtime_extraction
                else None
            ),
            "states": [
                {
                    "id": s.id,
                    "name": s.name,
                    "confidence": s.confidence,
                    "component_name": s.component_name,
                    "route_path": s.route_path,
                    "correlation_score": s.correlation_score,
                }
                for s in self.states
            ],
            "transitions": [
                {
                    "id": t.id,
                    "from_state_id": t.from_state_id,
                    "to_state_id": t.to_state_id,
                    "trigger_type": t.trigger_type,
                    "confidence": t.confidence,
                }
                for t in self.transitions
            ],
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "errors": self.errors,
            "warnings": self.warnings,
        }


# Abstract base classes for components


class StaticAnalyzer(ABC):
    """Base class for static code analyzers."""

    @abstractmethod
    async def analyze(self, project_path: Path) -> StaticAnalysisResult:
        """
        Analyze project source code.

        Args:
            project_path: Root directory of the project

        Returns:
            Static analysis results
        """
        pass

    @abstractmethod
    def supports_framework(self, framework: FrameworkType) -> bool:
        """Check if this analyzer supports the given framework."""
        pass


class RuntimeExtractor(ABC):
    """Base class for runtime extractors."""

    @abstractmethod
    async def extract(
        self, target: ExtractionTarget, config: ExtractionConfig
    ) -> RuntimeExtractionResult:
        """
        Extract states and transitions at runtime.

        Args:
            target: Target application
            config: Extraction configuration

        Returns:
            Runtime extraction results
        """
        pass

    @abstractmethod
    def supports_target(self, target: ExtractionTarget) -> bool:
        """Check if this extractor can handle the target."""
        pass


class StateMatcher(ABC):
    """Base class for matching static and runtime states."""

    @abstractmethod
    async def match(
        self,
        static: StaticAnalysisResult,
        runtime: RuntimeExtractionResult,
        threshold: float = 0.8,
    ) -> list[CorrelatedState]:
        """
        Match static components with runtime states.

        Args:
            static: Static analysis results
            runtime: Runtime extraction results
            threshold: Minimum correlation score

        Returns:
            List of correlated states
        """
        pass

    @abstractmethod
    def supports_framework(self, framework: FrameworkType) -> bool:
        """Check if this matcher supports the given framework."""
        pass
