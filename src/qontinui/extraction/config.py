"""
Configuration models for the abstract extraction architecture.

These models define the configuration options for extracting GUI structure
and behavior from various application types (web, desktop, mobile).
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ExtractionMode(Enum):
    """Extraction mode determining which analysis to perform."""

    BLACK_BOX = "black_box"  # Runtime-only extraction
    WHITE_BOX = "white_box"  # Static + runtime correlation
    STATIC_ONLY = "static_only"  # Static analysis only


class FrameworkType(Enum):
    """Supported application frameworks."""

    # Web frameworks
    REACT = "react"
    NEXT_JS = "next_js"
    REMIX = "remix"
    VUE = "vue"
    NUXT = "nuxt"
    ANGULAR = "angular"
    SVELTE = "svelte"
    SVELTEKIT = "sveltekit"
    SOLID = "solid"
    QWIK = "qwik"
    ASTRO = "astro"

    # Desktop frameworks
    TAURI = "tauri"
    ELECTRON = "electron"

    # Mobile frameworks
    REACT_NATIVE = "react_native"
    FLUTTER = "flutter"

    # Other
    VANILLA = "vanilla"
    UNKNOWN = "unknown"


class OutputFormat(Enum):
    """Format for extraction output."""

    STATE_STRUCTURE = "state_structure"  # Hierarchical state structure
    STATE_GRAPH = "state_graph"  # State machine graph
    TRAINING_DATA = "training_data"  # Format for model training


class InjectionPoint(Enum):
    """When to inject runtime scripts."""

    BEFORE_LOAD = "before_load"
    AFTER_LOAD = "after_load"
    ON_INTERACTION = "on_interaction"


@dataclass
class InjectionScript:
    """Script to inject into runtime for enhanced extraction."""

    content: str
    inject_at: InjectionPoint = InjectionPoint.AFTER_LOAD
    description: str = ""


@dataclass
class ExtractionTarget:
    """Target application configuration."""

    # Runtime target (one of these required for runtime extraction)
    url: str | None = None  # For web apps
    app_binary: Path | None = None  # For desktop/mobile apps
    app_dev_command: str | None = None  # Command to start dev server

    # Static analysis target (required for white-box or static-only)
    source_root: Path | None = None  # Root directory of source code
    entry_points: list[Path] = field(default_factory=list)  # Main entry files
    project_root: Path | None = None  # Project root (for package.json, etc.)


@dataclass
class RuntimeConfig:
    """Configuration for runtime extraction."""

    # Viewport configurations
    viewports: list[tuple[int, int]] = field(
        default_factory=lambda: [(1920, 1080), (1366, 768), (375, 667)]
    )

    # State capture options
    capture_hover_states: bool = True
    capture_scroll_states: bool = True
    max_interaction_depth: int = 3  # How deep to explore interactions

    # Route extraction
    extract_all_routes: bool = True  # Try to discover all routes
    known_routes: list[str] = field(default_factory=list)  # Routes to visit

    # Script injection for enhanced extraction
    injection_scripts: list[InjectionScript] = field(default_factory=list)

    # Performance
    screenshot_quality: int = 90  # JPEG quality 1-100
    timeout_ms: int = 30000  # Default timeout for operations
    navigation_timeout_ms: int = 60000  # Timeout for navigation

    # Browser-specific (for web)
    headless: bool = True
    browser_type: str = "chromium"  # chromium, firefox, webkit


@dataclass
class StaticConfig:
    """Configuration for static analysis."""

    # Source paths
    source_root: Path  # Required: Root directory of source code

    # File filtering
    include_patterns: list[str] = field(
        default_factory=lambda: [
            "**/*.tsx",
            "**/*.ts",
            "**/*.jsx",
            "**/*.js",
            "**/*.vue",
            "**/*.svelte",
        ]
    )
    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "**/node_modules/**",
            "**/dist/**",
            "**/build/**",
            "**/.next/**",
            "**/__tests__/**",
            "**/*.test.*",
            "**/*.spec.*",
        ]
    )

    # Analysis depth
    follow_imports: bool = True
    max_import_depth: int = 5

    # Framework-specific analysis
    analyze_server_components: bool = True  # React Server Components
    analyze_api_routes: bool = True  # Next.js API routes, etc.
    analyze_tauri_commands: bool = True  # Tauri command handlers

    # Type analysis
    use_typescript_types: bool = True
    infer_prop_types: bool = True

    # AST parsing
    parse_jsx: bool = True
    parse_typescript: bool = True


@dataclass
class OutputConfig:
    """Configuration for extraction output."""

    format: OutputFormat = OutputFormat.STATE_STRUCTURE
    include_screenshots: bool = True
    include_source_references: bool = True
    output_dir: Path = field(default_factory=lambda: Path("./qontinui_output"))

    # Filtering
    min_state_confidence: float = 0.5
    min_transition_confidence: float = 0.5

    # Compression
    compress_screenshots: bool = True
    compress_output: bool = False


@dataclass
class ExtractionConfig:
    """Main configuration for GUI extraction."""

    target: ExtractionTarget
    mode: ExtractionMode = ExtractionMode.BLACK_BOX
    framework: FrameworkType = FrameworkType.UNKNOWN

    # Sub-configurations
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    static: StaticConfig = field(default_factory=StaticConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Metadata
    extraction_name: str = "extraction"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        # Validate mode requirements
        if self.mode in (ExtractionMode.WHITE_BOX, ExtractionMode.STATIC_ONLY):
            if self.target.source_root is None:
                raise ValueError(f"{self.mode.value} mode requires source_root")

        if self.mode in (ExtractionMode.BLACK_BOX, ExtractionMode.WHITE_BOX):
            if (
                self.target.url is None
                and self.target.app_binary is None
                and self.target.app_dev_command is None
            ):
                raise ValueError(
                    f"{self.mode.value} mode requires url, app_binary, or app_dev_command"
                )
