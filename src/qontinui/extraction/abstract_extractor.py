"""
Abstract base class for unified GUI extraction.

All extraction backends (DOM, Vision, Accessibility) implement this interface
to provide consistent behavior and output format regardless of the underlying
extraction technology.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ExtractionContext:
    """
    Context passed to extractors during extraction.

    Contains information about the target and environment needed
    for extraction. Different backends use different fields.
    """

    # Screenshot-based extraction (Vision)
    screenshot_path: Path | None = None

    # URL-based extraction (DOM)
    url: str | None = None

    # Application-based extraction (Accessibility)
    app_name: str | None = None
    window_title: str | None = None
    process_id: int | None = None

    # Viewport/screen dimensions
    viewport: tuple[int, int] = (1920, 1080)

    # Platform information
    platform: str = "unknown"  # windows, darwin, linux, web

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Auto-detect platform if not specified."""
        if self.platform == "unknown":
            import sys

            if self.url:
                self.platform = "web"
            else:
                self.platform = sys.platform


@dataclass
class ExtractedElement:
    """
    A UI element extracted from the GUI.

    This is the unified element format produced by all extraction backends.
    """

    id: str
    element_type: str  # button, input, link, label, image, container, etc.
    bbox: tuple[int, int, int, int]  # x, y, width, height
    confidence: float  # 0.0 - 1.0

    # Identification
    text: str | None = None
    selector: str | None = None  # CSS selector (DOM), accessibility path, etc.

    # Properties
    is_interactive: bool = False
    is_visible: bool = True
    is_enabled: bool = True
    is_focused: bool = False

    # Accessibility information
    aria_role: str | None = None
    aria_label: str | None = None
    name: str | None = None  # Accessible name

    # Additional attributes
    attributes: dict[str, str] = field(default_factory=dict)

    # Extraction metadata
    extraction_method: str = "unknown"  # dom, vision, accessibility
    source_backend: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> tuple[int, int]:
        """Get center point of element."""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)

    @property
    def area(self) -> int:
        """Get area of element in pixels."""
        _, _, w, h = self.bbox
        return w * h

    def iou(self, other: "ExtractedElement") -> float:
        """Calculate Intersection over Union with another element."""
        x1, y1, w1, h1 = self.bbox
        x2, y2, w2, h2 = other.bbox

        # Calculate intersection
        xi = max(x1, x2)
        yi = max(y1, y2)
        wi = min(x1 + w1, x2 + w2) - xi
        hi = min(y1 + h1, y2 + h2) - yi

        if wi <= 0 or hi <= 0:
            return 0.0

        intersection = wi * hi
        union = (w1 * h1) + (w2 * h2) - intersection

        return intersection / union if union > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "element_type": self.element_type,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "text": self.text,
            "selector": self.selector,
            "is_interactive": self.is_interactive,
            "is_visible": self.is_visible,
            "is_enabled": self.is_enabled,
            "aria_role": self.aria_role,
            "aria_label": self.aria_label,
            "extraction_method": self.extraction_method,
            "attributes": self.attributes,
            "metadata": self.metadata,
        }


@dataclass
class ExtractedState:
    """
    A UI state (region/view) extracted from the GUI.

    States represent logical groupings of UI that can appear/disappear
    together, such as modals, menus, pages, etc.
    """

    id: str
    name: str
    state_type: str  # page, modal, panel, menu, dropdown, sidebar, toolbar, etc.

    # Bounding box (optional - some states are abstract)
    bbox: tuple[int, int, int, int] | None = None

    # Contained elements
    element_ids: list[str] = field(default_factory=list)

    # Visual reference
    screenshot_id: str | None = None
    screenshot_path: Path | None = None

    # Confidence and method
    confidence: float = 1.0
    extraction_method: str = "unknown"
    detection_method: str = "unknown"  # semantic, clustering, containment

    # Source information
    source_url: str | None = None
    source_file: str | None = None
    source_line: int | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "state_type": self.state_type,
            "bbox": list(self.bbox) if self.bbox else None,
            "element_ids": self.element_ids,
            "screenshot_id": self.screenshot_id,
            "confidence": self.confidence,
            "extraction_method": self.extraction_method,
            "source_url": self.source_url,
            "metadata": self.metadata,
        }


@dataclass
class ExtractedTransition:
    """
    A transition between states.

    Transitions represent actions that cause state changes.
    """

    id: str
    from_state_id: str
    to_state_id: str

    # Trigger information
    trigger_type: str  # click, hover, type, scroll, navigate, keyboard, etc.
    trigger_element_id: str | None = None
    trigger_selector: str | None = None
    trigger_value: str | None = None  # For type/scroll - the value

    # Confidence and verification
    confidence: float = 1.0
    verified: bool = False  # True if actually executed and observed

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "from_state_id": self.from_state_id,
            "to_state_id": self.to_state_id,
            "trigger_type": self.trigger_type,
            "trigger_element_id": self.trigger_element_id,
            "trigger_selector": self.trigger_selector,
            "confidence": self.confidence,
            "verified": self.verified,
            "metadata": self.metadata,
        }


@dataclass
class ExtractionResult:
    """
    Unified result from any extraction backend.

    All extractors produce this format regardless of the underlying
    extraction method.
    """

    extraction_id: str
    extraction_method: str  # dom, vision, accessibility, hybrid

    # Extracted data
    elements: list[ExtractedElement] = field(default_factory=list)
    states: list[ExtractedState] = field(default_factory=list)
    transitions: list[ExtractedTransition] = field(default_factory=list)

    # Visual references
    screenshots: list[str] = field(default_factory=list)  # screenshot IDs/paths
    screenshots_dir: Path | None = None

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    duration_ms: float = 0.0

    # Status
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Context
    context: ExtractionContext | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def complete(self) -> None:
        """Mark extraction as complete and calculate duration."""
        self.completed_at = datetime.now()
        self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000

    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)

    @property
    def is_successful(self) -> bool:
        """Check if extraction completed without errors."""
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "extraction_id": self.extraction_id,
            "extraction_method": self.extraction_method,
            "elements": [e.to_dict() for e in self.elements],
            "states": [s.to_dict() for s in self.states],
            "transitions": [t.to_dict() for t in self.transitions],
            "screenshots": self.screenshots,
            "screenshots_dir": str(self.screenshots_dir) if self.screenshots_dir else None,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


class AbstractExtractor(ABC):
    """
    Abstract base class for all GUI extraction backends.

    All extractors must implement these methods to provide a unified
    extraction interface regardless of the underlying technology.

    Implementations:
    - DOMExtractor: Uses Playwright for web DOM extraction
    - VisionExtractor: Uses CV/ML for screenshot-based extraction
    - AccessibilityExtractor: Uses OS accessibility APIs

    Example usage:
        >>> extractor = VisionExtractor()
        >>> context = ExtractionContext(screenshot_path=Path("screenshot.png"))
        >>> config = ExtractorConfig(backend=ExtractionBackend.VISION)
        >>> result = await extractor.extract(context, config)
        >>> print(f"Found {len(result.elements)} elements")
    """

    @abstractmethod
    async def extract(
        self,
        context: ExtractionContext,
        config: "ExtractorConfig",
    ) -> ExtractionResult:
        """
        Perform extraction and return unified results.

        This is the main entry point for extraction. It should:
        1. Validate the context and config
        2. Extract elements from the GUI
        3. Detect states/regions
        4. Infer transitions (if enabled)
        5. Capture screenshots (if enabled)
        6. Return unified ExtractionResult

        Args:
            context: Extraction context with target information
            config: Extractor-specific configuration

        Returns:
            ExtractionResult with elements, states, and transitions

        Raises:
            ExtractionError: If extraction fails
            ValueError: If context is invalid for this extractor
        """
        pass

    @abstractmethod
    async def extract_elements(
        self,
        context: ExtractionContext,
        config: "ExtractorConfig",
    ) -> list[ExtractedElement]:
        """
        Extract interactive elements from the current GUI state.

        This method extracts individual UI elements (buttons, inputs, etc.)
        with their bounding boxes, types, and properties.

        Args:
            context: Extraction context
            config: Extractor configuration

        Returns:
            List of extracted elements with bounding boxes

        Raises:
            ExtractionError: If element extraction fails
        """
        pass

    @abstractmethod
    async def extract_states(
        self,
        context: ExtractionContext,
        config: "ExtractorConfig",
        elements: list[ExtractedElement] | None = None,
    ) -> list[ExtractedState]:
        """
        Extract UI states/regions from the current GUI state.

        States represent logical groupings of UI elements that can
        appear/disappear together (modals, menus, pages, etc.).

        Args:
            context: Extraction context
            config: Extractor configuration
            elements: Pre-extracted elements (optional, for efficiency)

        Returns:
            List of extracted states

        Raises:
            ExtractionError: If state detection fails
        """
        pass

    @abstractmethod
    async def capture_screenshot(
        self,
        context: ExtractionContext,
        region: tuple[int, int, int, int] | None = None,
    ) -> Path:
        """
        Capture a screenshot of the current state.

        Args:
            context: Extraction context
            region: Optional region to capture (x, y, width, height)

        Returns:
            Path to saved screenshot

        Raises:
            ScreenshotError: If screenshot capture fails
        """
        pass

    @classmethod
    @abstractmethod
    def supports_target(cls, context: ExtractionContext) -> bool:
        """
        Check if this extractor can handle the given target.

        Used by the orchestrator to auto-select the appropriate
        extractor based on the target type.

        Args:
            context: Extraction context to check

        Returns:
            True if this extractor can extract from the target
        """
        pass

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """
        Return the extractor name for logging and configuration.

        Returns:
            String name of the extractor (e.g., "dom", "vision", "accessibility")
        """
        pass

    @classmethod
    def get_priority(cls) -> int:
        """
        Return the extractor priority for auto-selection.

        Higher priority extractors are preferred when multiple
        extractors support the same target.

        Returns:
            Integer priority (higher = more preferred)
        """
        return 0

    async def validate_context(self, context: ExtractionContext) -> None:
        """
        Validate that the context is suitable for this extractor.

        Subclasses should override this to add specific validation.

        Args:
            context: Extraction context to validate

        Raises:
            ValueError: If context is invalid
        """
        if not self.supports_target(context):
            raise ValueError(f"Extractor {self.get_name()} does not support target: {context}")

    def filter_elements(
        self,
        elements: list[ExtractedElement],
        config: "ExtractorConfig",
    ) -> list[ExtractedElement]:
        """
        Filter elements based on configuration.

        Applies element type filtering, size filtering, and confidence
        thresholds from the configuration.

        Args:
            elements: List of elements to filter
            config: Configuration with filter settings

        Returns:
            Filtered list of elements
        """
        from .extractor_config import ElementFilter

        filtered = []

        for element in elements:
            # Confidence filter
            if element.confidence < config.confidence.element_min:
                continue

            # Size filter
            _, _, w, h = element.bbox
            if not config.size_threshold.matches(w, h):
                continue

            # Type filter
            if config.element_filter == ElementFilter.INTERACTIVE_ONLY:
                if not element.is_interactive:
                    continue
            elif config.element_filter == ElementFilter.BY_TYPE:
                if element.element_type not in config.element_types:
                    continue
            elif config.element_filter == ElementFilter.CUSTOM:
                if config.custom_filter and not config.custom_filter(element):
                    continue

            filtered.append(element)

        return filtered


# Type alias for ExtractorConfig (imported from extractor_config.py)
# This allows the abstract class to reference it without circular imports
ExtractorConfig = Any  # Will be properly typed when imported


class ExtractionError(Exception):
    """Base exception for extraction errors."""

    pass


class ScreenshotError(ExtractionError):
    """Exception raised when screenshot capture fails."""

    pass


class ConnectionError(ExtractionError):
    """Exception raised when connection to target fails."""

    pass
