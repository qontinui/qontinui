"""
Unified configuration for the multi-backend extraction architecture.

This module provides user-configurable extraction rules that work across
all extraction backends (DOM, Vision, Accessibility).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ExtractionBackend(Enum):
    """Available extraction backends."""

    DOM = "dom"  # Playwright/DOM-based
    VISION = "vision"  # Computer Vision
    ACCESSIBILITY = "a11y"  # OS Accessibility APIs
    HYBRID = "hybrid"  # Combine multiple backends
    AUTO = "auto"  # Auto-select best backend


class ElementFilter(Enum):
    """Element filtering strategies."""

    INTERACTIVE_ONLY = "interactive_only"  # Buttons, inputs, links
    ALL_VISIBLE = "all_visible"  # All visible elements
    BY_TYPE = "by_type"  # Filter by element type
    BY_SIZE = "by_size"  # Filter by size threshold
    CUSTOM = "custom"  # Custom filter function


class HybridStrategy(Enum):
    """Strategies for hybrid extraction."""

    PRIMARY_FALLBACK = "primary_fallback"  # Use primary, fall back on failure
    PARALLEL_MERGE = "parallel_merge"  # Run all, merge results
    VALIDATION = "validation"  # Use secondary to validate primary
    SEQUENTIAL = "sequential"  # Run in sequence, stop on first success


@dataclass
class SizeThreshold:
    """Size thresholds for element filtering."""

    min_width: int = 10
    min_height: int = 10
    max_width: int = 10000
    max_height: int = 10000

    def matches(self, width: int, height: int) -> bool:
        """Check if size is within thresholds."""
        return (
            self.min_width <= width <= self.max_width
            and self.min_height <= height <= self.max_height
        )


@dataclass
class ConfidenceThreshold:
    """Confidence thresholds for extraction."""

    element_min: float = 0.5  # Minimum confidence for elements
    state_min: float = 0.5  # Minimum confidence for states
    transition_min: float = 0.5  # Minimum confidence for transitions

    def __post_init__(self) -> None:
        """Validate thresholds are in valid range."""
        for name, value in [
            ("element_min", self.element_min),
            ("state_min", self.state_min),
            ("transition_min", self.transition_min),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")


@dataclass
class VisionConfig:
    """Configuration specific to Vision extraction."""

    # Detection methods
    use_classical_cv: bool = True  # OpenCV-based detection
    use_ml_detection: bool = False  # ML model detection (requires model)
    use_ocr: bool = True  # Text detection
    use_segmentation: bool = False  # SAM/segmentation models

    # Model settings
    detection_model: str | None = None  # Path to custom detection model
    ocr_engine: str = "easyocr"  # easyocr, tesseract, paddleocr

    # Classical CV thresholds
    edge_detection_low: int = 50
    edge_detection_high: int = 150
    contour_min_area: int = 100
    contour_approximation_epsilon: float = 0.02  # For polygon approximation

    # Text detection
    text_confidence_threshold: float = 0.6
    text_merge_threshold: int = 10  # Merge text boxes within this distance

    # Button detection
    button_aspect_ratio_min: float = 1.5
    button_aspect_ratio_max: float = 8.0
    button_width_min: int = 20
    button_width_max: int = 300
    button_height_min: int = 15
    button_height_max: int = 100

    # Performance
    max_elements: int = 1000
    downscale_factor: float = 1.0  # Downscale for performance (0.5 = half size)

    # Deduplication
    iou_threshold: float = 0.5  # IoU threshold for merging overlapping detections


@dataclass
class DOMConfig:
    """Configuration specific to DOM extraction."""

    # Browser settings
    browser: str = "chromium"  # chromium, firefox, webkit
    headless: bool = True
    viewport: tuple[int, int] = (1920, 1080)

    # Navigation
    wait_for_network_idle: bool = True
    navigation_timeout_ms: int = 30000
    stability_timeout_ms: int = 500

    # Element extraction
    include_shadow_dom: bool = True
    include_iframes: bool = True
    include_hidden: bool = False

    # Selector generation
    prefer_id: bool = True
    prefer_data_testid: bool = True
    prefer_aria_label: bool = True

    # Authentication
    cookies: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)

    # Crawling (for multi-page extraction)
    max_pages: int = 100
    max_depth: int = 5
    same_origin_only: bool = True
    follow_links: bool = True


@dataclass
class AccessibilityConfig:
    """Configuration specific to Accessibility extraction."""

    # Platform-specific (auto-detected if not specified)
    use_uia: bool = True  # Windows UI Automation
    use_atspi: bool = True  # Linux AT-SPI
    use_ax: bool = True  # macOS Accessibility

    # Window targeting
    window_title: str | None = None  # Filter by window title
    process_name: str | None = None  # Filter by process name
    process_id: int | None = None  # Filter by PID

    # Filtering
    include_invisible: bool = False
    include_disabled: bool = True
    include_offscreen: bool = False

    # Tree traversal
    max_depth: int = 50
    include_children: bool = True

    # Role filtering
    include_roles: list[str] | None = None  # If set, only include these roles
    exclude_roles: list[str] | None = None  # Roles to exclude

    # Performance
    timeout_ms: int = 10000


@dataclass
class HybridConfig:
    """Configuration for hybrid extraction."""

    # Strategy
    strategy: HybridStrategy = HybridStrategy.PRIMARY_FALLBACK

    # Backend order (primary first)
    backend_order: list[ExtractionBackend] = field(
        default_factory=lambda: [
            ExtractionBackend.DOM,
            ExtractionBackend.ACCESSIBILITY,
            ExtractionBackend.VISION,
        ]
    )

    # Merge settings (for parallel strategy)
    merge_iou_threshold: float = 0.5  # IoU threshold for deduplication
    prefer_higher_confidence: bool = True

    # Validation settings (for validation strategy)
    validation_iou_threshold: float = 0.7  # Elements must overlap this much


@dataclass
class ExtractorConfig:
    """
    Main configuration for unified extraction.

    Users can configure:
    - Which backends to use
    - Element filtering rules
    - Confidence thresholds
    - Backend-specific settings

    This configuration is shared across all extraction backends.
    """

    # Backend selection
    backend: ExtractionBackend = ExtractionBackend.AUTO
    fallback_backends: list[ExtractionBackend] = field(
        default_factory=lambda: [ExtractionBackend.VISION]
    )

    # Element filtering
    element_filter: ElementFilter = ElementFilter.INTERACTIVE_ONLY
    element_types: list[str] = field(
        default_factory=lambda: [
            "button",
            "input",
            "link",
            "select",
            "checkbox",
            "radio",
            "textarea",
        ]
    )
    size_threshold: SizeThreshold = field(default_factory=SizeThreshold)
    custom_filter: Callable[[Any], bool] | None = None  # For ElementFilter.CUSTOM

    # Confidence thresholds
    confidence: ConfidenceThreshold = field(default_factory=ConfidenceThreshold)

    # State detection
    detect_states: bool = True
    detect_transitions: bool = True
    state_detection_method: str = "semantic"  # semantic, clustering, containment

    # Screenshots
    capture_screenshots: bool = True
    screenshot_format: str = "png"
    screenshot_quality: int = 90  # For JPEG

    # Backend-specific configs
    vision: VisionConfig = field(default_factory=VisionConfig)
    dom: DOMConfig = field(default_factory=DOMConfig)
    accessibility: AccessibilityConfig = field(default_factory=AccessibilityConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)

    # Output
    output_dir: Path | None = None

    # Metadata
    extraction_name: str = "extraction"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "backend": self.backend.value,
            "fallback_backends": [b.value for b in self.fallback_backends],
            "element_filter": self.element_filter.value,
            "element_types": self.element_types,
            "size_threshold": {
                "min_width": self.size_threshold.min_width,
                "min_height": self.size_threshold.min_height,
                "max_width": self.size_threshold.max_width,
                "max_height": self.size_threshold.max_height,
            },
            "confidence": {
                "element_min": self.confidence.element_min,
                "state_min": self.confidence.state_min,
                "transition_min": self.confidence.transition_min,
            },
            "detect_states": self.detect_states,
            "detect_transitions": self.detect_transitions,
            "capture_screenshots": self.capture_screenshots,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "extraction_name": self.extraction_name,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractorConfig":
        """Create from dictionary."""
        config = cls()

        if "backend" in data:
            config.backend = ExtractionBackend(data["backend"])

        if "fallback_backends" in data:
            config.fallback_backends = [ExtractionBackend(b) for b in data["fallback_backends"]]

        if "element_filter" in data:
            config.element_filter = ElementFilter(data["element_filter"])

        if "element_types" in data:
            config.element_types = data["element_types"]

        if "size_threshold" in data:
            st = data["size_threshold"]
            config.size_threshold = SizeThreshold(
                min_width=st.get("min_width", 10),
                min_height=st.get("min_height", 10),
                max_width=st.get("max_width", 10000),
                max_height=st.get("max_height", 10000),
            )

        if "confidence" in data:
            ct = data["confidence"]
            config.confidence = ConfidenceThreshold(
                element_min=ct.get("element_min", 0.5),
                state_min=ct.get("state_min", 0.5),
                transition_min=ct.get("transition_min", 0.5),
            )

        if "detect_states" in data:
            config.detect_states = data["detect_states"]

        if "detect_transitions" in data:
            config.detect_transitions = data["detect_transitions"]

        if "capture_screenshots" in data:
            config.capture_screenshots = data["capture_screenshots"]

        if "output_dir" in data and data["output_dir"]:
            config.output_dir = Path(data["output_dir"])

        if "extraction_name" in data:
            config.extraction_name = data["extraction_name"]

        if "tags" in data:
            config.tags = data["tags"]

        if "metadata" in data:
            config.metadata = data["metadata"]

        return config

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "ExtractorConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml

        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
