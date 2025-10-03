"""Qontinui properties - configuration framework using Pydantic.

Centralized configuration using Pydantic for type safety and validation.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CoreConfig(BaseModel):
    """Core framework settings."""

    model_config = ConfigDict(validate_assignment=True)

    image_path: str = Field(default="classpath:images/", description="Path to image resources")
    mock: bool = Field(default=False, description="Enable mock mode for testing without GUI")
    headless: bool = Field(default=False, description="Run in headless mode without display")
    sikuli_jar_path: str | None = Field(default=None, description="Path to SikuliX jar file")
    tesseract_path: str | None = Field(default=None, description="Path to Tesseract executable")
    image_cache_size: int = Field(
        default=100, ge=0, description="Maximum number of images to cache"
    )
    auto_wait_timeout: float = Field(
        default=3.0, ge=0, description="Default wait timeout in seconds"
    )


class MouseConfig(BaseModel):
    """Mouse action configuration."""

    model_config = ConfigDict(validate_assignment=True)

    move_delay: float = Field(default=0.5, ge=0, description="Delay for mouse movement in seconds")
    pause_before_down: float = Field(
        default=0.0, ge=0, description="Pause before mouse down in seconds"
    )
    pause_after_down: float = Field(
        default=0.0, ge=0, description="Pause after mouse down in seconds"
    )
    pause_before_up: float = Field(
        default=0.0, ge=0, description="Pause before mouse up in seconds"
    )
    pause_after_up: float = Field(default=0.0, ge=0, description="Pause after mouse up in seconds")
    click_delay: float = Field(
        default=0.0, ge=0, description="Delay between clicks in double-click"
    )
    drag_delay: float = Field(default=0.5, ge=0, description="Delay during drag operations")


class MockConfig(BaseModel):
    """Mock mode timing configuration."""

    model_config = ConfigDict(validate_assignment=True)

    click_duration: float = Field(
        default=0.5, ge=0, description="Simulated click duration in seconds"
    )
    type_duration: float = Field(
        default=2.0, ge=0, description="Simulated typing duration in seconds"
    )
    find_duration: float = Field(
        default=0.3, ge=0, description="Simulated find duration in seconds"
    )
    drag_duration: float = Field(
        default=1.0, ge=0, description="Simulated drag duration in seconds"
    )
    scroll_duration: float = Field(
        default=0.5, ge=0, description="Simulated scroll duration in seconds"
    )
    wait_duration: float = Field(
        default=0.1, ge=0, description="Simulated wait duration in seconds"
    )
    vanish_duration: float = Field(
        default=1.0, ge=0, description="Simulated vanish check duration in seconds"
    )
    exists_duration: float = Field(
        default=0.3, ge=0, description="Simulated exists check duration in seconds"
    )


class ScreenshotConfig(BaseModel):
    """Screenshot and history settings."""

    model_config = ConfigDict(validate_assignment=True)

    save_snapshots: bool = Field(default=True, description="Save screenshots during execution")
    path: str = Field(default="screenshots/", description="Path to save screenshots")
    max_history: int = Field(default=50, ge=0, description="Maximum screenshot history to maintain")
    format: str = Field(
        default="png", pattern="^(png|jpg|jpeg|bmp)$", description="Screenshot image format"
    )
    quality: int = Field(default=90, ge=1, le=100, description="JPEG quality (1-100)")
    include_timestamp: bool = Field(default=True, description="Include timestamp in filename")
    capture_on_error: bool = Field(
        default=True, description="Automatically capture screenshot on error"
    )


class IllustrationConfig(BaseModel):
    """Action illustration settings."""

    model_config = ConfigDict(validate_assignment=True)

    enabled: bool = Field(default=True, description="Enable action illustrations")
    show_click: bool = Field(default=True, description="Illustrate click actions")
    show_drag: bool = Field(default=True, description="Illustrate drag actions")
    show_type: bool = Field(default=True, description="Illustrate type actions")
    show_find: bool = Field(default=True, description="Illustrate find operations")
    highlight_color: str = Field(default="red", description="Color for highlighting elements")
    highlight_thickness: int = Field(
        default=3, ge=1, le=10, description="Highlight border thickness"
    )
    annotation_font_size: int = Field(
        default=12, ge=8, le=72, description="Font size for annotations"
    )


class AnalysisConfig(BaseModel):
    """Color analysis settings."""

    model_config = ConfigDict(validate_assignment=True)

    kmeans_clusters: int = Field(
        default=3, ge=1, le=20, description="Number of k-means clusters for color analysis"
    )
    color_tolerance: int = Field(default=30, ge=0, le=255, description="Color matching tolerance")
    hsv_bins: list[int] = Field(
        default=[50, 60, 60], description="HSV histogram bins [hue, saturation, value]"
    )
    min_contour_area: int = Field(
        default=100, ge=1, description="Minimum contour area for color regions"
    )
    max_contour_area: int = Field(
        default=100000, ge=1, description="Maximum contour area for color regions"
    )


class RecordingConfig(BaseModel):
    """Screen recording settings."""

    model_config = ConfigDict(validate_assignment=True)

    enabled: bool = Field(default=False, description="Enable screen recording")
    path: str = Field(default="recordings/", description="Path to save recordings")
    fps: int = Field(default=30, ge=1, le=60, description="Frames per second for recording")
    codec: str = Field(default="mp4v", description="Video codec to use")
    quality: str = Field(
        default="medium", pattern="^(low|medium|high)$", description="Recording quality preset"
    )
    include_audio: bool = Field(default=False, description="Include audio in recording")
    max_duration_minutes: int = Field(
        default=60, ge=1, description="Maximum recording duration in minutes"
    )


class DatasetConfig(BaseModel):
    """AI dataset generation settings."""

    model_config = ConfigDict(validate_assignment=True)

    collect: bool = Field(default=False, description="Enable dataset collection")
    path: str = Field(default="datasets/", description="Path to save datasets")
    include_screenshots: bool = Field(default=True, description="Include screenshots in dataset")
    include_actions: bool = Field(default=True, description="Include action data in dataset")
    include_timing: bool = Field(default=True, description="Include timing information")
    include_results: bool = Field(default=True, description="Include action results")
    format: str = Field(
        default="json", pattern="^(json|csv|parquet)$", description="Dataset file format"
    )
    compression: str | None = Field(
        default=None, pattern="^(gzip|bzip2|xz)?$", description="Dataset compression"
    )


class TestingConfig(BaseModel):
    """Test execution settings."""

    model_config = ConfigDict(validate_assignment=True)

    timeout_multiplier: float = Field(
        default=2.0, ge=1.0, description="Multiply timeouts during testing"
    )
    retry_failed: bool = Field(default=True, description="Automatically retry failed tests")
    max_retries: int = Field(default=3, ge=0, description="Maximum test retry attempts")
    screenshot_on_failure: bool = Field(
        default=True, description="Capture screenshot on test failure"
    )
    verbose_logging: bool = Field(default=True, description="Enable verbose logging during tests")
    parallel_execution: bool = Field(default=False, description="Enable parallel test execution")
    random_seed: int | None = Field(default=None, description="Random seed for reproducible tests")
    iteration: int = Field(default=1, ge=1, description="Current test iteration")
    send_logs: bool = Field(default=True, description="Send logs to external systems")


class MonitorConfig(BaseModel):
    """Monitor configuration settings."""

    model_config = ConfigDict(validate_assignment=True)

    default_screen_index: int = Field(
        default=-1,
        ge=-1,
        description="Monitor index to use for automation (0=primary, 1=secondary, -1=primary)",
    )
    multi_monitor_enabled: bool = Field(default=False, description="Enable multi-monitor support")
    search_all_monitors: bool = Field(
        default=False, description="Search across all monitors when finding elements"
    )
    log_monitor_info: bool = Field(
        default=True, description="Log monitor information for each operation"
    )
    operation_monitor_map: dict[str, int] = Field(
        default_factory=dict, description="Monitor assignment for specific operations"
    )


class DpiConfig(BaseModel):
    """DPI and scaling configuration."""

    model_config = ConfigDict(validate_assignment=True)

    disable: bool = Field(
        default=True, description="Disable DPI awareness to force physical resolution capture"
    )
    resize_factor: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Resize factor for pattern matching"
    )
    pattern_source: str = Field(
        default="WINDOWS_TOOL",
        pattern="^(SIKULI_IDE|WINDOWS_TOOL|FFMPEG_TOOL)$",
        description="Pattern source hint for scaling",
    )


class CaptureConfig(BaseModel):
    """Screen capture provider configuration."""

    model_config = ConfigDict(validate_assignment=True)

    provider: str = Field(
        default="AUTO",
        pattern="^(AUTO|ROBOT|FFMPEG|JAVACV_FFMPEG|SIKULIX|MSS)$",
        description="Capture provider to use",
    )
    prefer_physical: bool = Field(default=True, description="Prefer physical resolution captures")
    fallback_enabled: bool = Field(
        default=True, description="Enable fallback to other providers if preferred fails"
    )
    fallback_chain: list[str] = Field(
        default=["MSS", "ROBOT"], description="Fallback chain priority for capture providers"
    )
    enable_logging: bool = Field(default=False, description="Enable capture operation logging")
    auto_retry: bool = Field(default=True, description="Auto-retry failed captures")
    retry_count: int = Field(
        default=3, ge=0, le=10, description="Number of retry attempts for failed captures"
    )


class SikuliConfig(BaseModel):
    """SikuliX integration settings."""

    model_config = ConfigDict(validate_assignment=True)

    highlight: bool = Field(default=False, description="Enable SikuliX highlighting")
    highlight_duration: int = Field(default=2, ge=0, description="Highlight duration in seconds")
    auto_wait_timeout: float = Field(default=0.0, ge=0, description="Auto wait timeout in seconds")
    delay_before_mouse_down: float = Field(
        default=0.0, ge=0, description="Delay before mouse down in seconds"
    )
    delay_after_drag: float = Field(default=0.0, ge=0, description="Delay after drag in seconds")
    move_mouse_delay: float = Field(default=0.5, ge=0, description="Move mouse delay in seconds")


class StartupConfig(BaseModel):
    """Startup configuration settings."""

    model_config = ConfigDict(validate_assignment=True)

    verify_initial_states: bool = Field(
        default=False, description="Automatically verify initial states on startup"
    )
    initial_states: list[str] = Field(
        default_factory=list, description="List of state names to verify at startup"
    )
    fallback_search: bool = Field(
        default=False, description="Search all states if specified states not found"
    )
    activate_first_only: bool = Field(
        default=False, description="Activate only the first found state"
    )
    startup_delay: int = Field(
        default=0, ge=0, description="Delay in seconds before initial state verification"
    )


class AutomationConfig(BaseModel):
    """Automation failure handling configuration."""

    model_config = ConfigDict(validate_assignment=True)

    exit_on_failure: bool = Field(
        default=False, description="Exit application when automation fails"
    )
    failure_exit_code: int = Field(
        default=1, ge=0, description="Exit code when exitOnFailure is true"
    )
    throw_on_failure: bool = Field(
        default=False, description="Throw exceptions when automation fails"
    )
    log_stack_traces: bool = Field(
        default=True, description="Log stack traces for automation failures"
    )
    max_retries: int = Field(
        default=0, ge=0, description="Maximum number of automation retry attempts"
    )
    retry_delay_ms: int = Field(
        default=1000, ge=0, description="Delay in milliseconds between retry attempts"
    )
    continue_on_failure: bool = Field(
        default=False, description="Continue with remaining automation steps after failure"
    )
    timeout_seconds: int = Field(
        default=0,
        ge=0,
        description="Timeout in seconds for entire automation sequence (0=no timeout)",
    )
    fail_fast: bool = Field(
        default=False, description="Stop immediately on first failure without retries"
    )


class AutoScalingConfig(BaseModel):
    """Automatic pattern scaling configuration."""

    model_config = ConfigDict(validate_assignment=True)

    enabled: bool = Field(default=True, description="Enable automatic pattern scaling detection")
    cache_enabled: bool = Field(default=True, description="Enable scaling cache")
    global_learning: bool = Field(
        default=True, description="Enable global learning across patterns"
    )
    min_confidence: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Minimum confidence for scaling detection"
    )


class LoggingConfig(BaseModel):
    """Comprehensive logging configuration."""

    model_config = ConfigDict(validate_assignment=True)

    # Global settings
    global_level: str = Field(
        default="INFO",
        pattern="^(OFF|ERROR|WARN|INFO|DEBUG|TRACE)$",
        description="Global log level",
    )

    # Category-specific levels
    actions_level: str = Field(default="INFO", description="Log level for actions")
    transitions_level: str = Field(default="INFO", description="Log level for state transitions")
    matching_level: str = Field(default="WARN", description="Log level for pattern matching")
    performance_level: str = Field(default="INFO", description="Log level for performance metrics")
    state_level: str = Field(default="DEBUG", description="Log level for state management")

    # Output configuration
    output_format: str = Field(
        default="STRUCTURED", pattern="^(SIMPLE|STRUCTURED|JSON)$", description="Output format"
    )
    include_timestamp: bool = Field(default=True, description="Include timestamp in logs")
    include_thread: bool = Field(default=False, description="Include thread name in logs")
    include_correlation_id: bool = Field(default=True, description="Include correlation ID")

    # Performance
    async_logging: bool = Field(default=True, description="Use asynchronous logging")
    buffer_size: int = Field(default=8192, ge=1024, description="Buffer size for async logging")

    # Enrichment
    include_screenshots: bool = Field(default=False, description="Attach screenshots to logs")
    include_similarity_scores: bool = Field(default=True, description="Include similarity scores")
    include_timing_breakdown: bool = Field(default=True, description="Include timing breakdown")


class HighlightConfig(BaseModel):
    """Visual highlighting configuration."""

    model_config = ConfigDict(validate_assignment=True)

    # Global settings
    enabled: bool = Field(default=True, description="Global highlighting enable/disable")
    auto_highlight_finds: bool = Field(
        default=True, description="Automatically highlight successful finds"
    )

    # Find highlighting
    find_color: str = Field(default="#00FF00", description="Color for highlighting found images")
    find_duration: float = Field(
        default=2.0, ge=0, description="Duration to show highlight (seconds)"
    )
    find_border_width: int = Field(default=3, ge=1, le=10, description="Border width in pixels")

    # Click highlighting
    click_enabled: bool = Field(default=True, description="Enable click highlighting")
    click_color: str = Field(default="#FFFF00", description="Color for click highlights")
    click_duration: float = Field(default=0.5, ge=0, description="Duration (seconds)")
    click_radius: int = Field(
        default=20, ge=5, le=100, description="Radius of click indicator circle"
    )


class ConsoleActionConfig(BaseModel):
    """Console action reporting configuration."""

    model_config = ConfigDict(validate_assignment=True)

    enabled: bool = Field(default=True, description="Enable console action reporting")
    level: str = Field(
        default="NORMAL", pattern="^(QUIET|NORMAL|VERBOSE)$", description="Verbosity level"
    )
    show_timing: bool = Field(default=True, description="Show timing information for actions")
    use_colors: bool = Field(default=True, description="Use colored output (ANSI)")
    use_icons: bool = Field(default=True, description="Use unicode icons in output")

    # Performance thresholds
    performance_warn_threshold: int = Field(
        default=1000, ge=0, description="Warning threshold (milliseconds)"
    )
    performance_error_threshold: int = Field(
        default=5000, ge=0, description="Error threshold (milliseconds)"
    )

    # Action reporting settings
    console_actions: bool = Field(default=True, description="Enable console action output")
    report_individual_actions: bool = Field(
        default=True, description="Report each individual action as it executes"
    )


class ImageDebugConfig(BaseModel):
    """Image debugging configuration."""

    model_config = ConfigDict(validate_assignment=True)

    enabled: bool = Field(default=False, description="Master switch for image debugging")
    level: str = Field(
        default="VISUAL", pattern="^(OFF|BASIC|DETAILED|VISUAL|FULL)$", description="Debug level"
    )
    save_screenshots: bool = Field(default=True, description="Save screenshots of entire screen")
    save_patterns: bool = Field(default=True, description="Save pattern images")
    save_comparisons: bool = Field(default=True, description="Save comparison images")
    output_dir: str = Field(default="debug/image-finding", description="Output directory")

    # Visual properties
    show_search_regions: bool = Field(default=True, description="Show search regions")
    show_match_scores: bool = Field(default=True, description="Show match scores")
    create_heatmap: bool = Field(default=False, description="Create heatmap visualization")


class GuiAccessConfig(BaseModel):
    """GUI access verification configuration."""

    model_config = ConfigDict(validate_assignment=True)

    report_problems: bool = Field(default=True, description="Report GUI access problems")
    verbose_errors: bool = Field(default=True, description="Show verbose error details")
    suggest_solutions: bool = Field(
        default=True, description="Suggest solutions for detected problems"
    )
    check_on_startup: bool = Field(default=True, description="Check GUI access on startup")
    continue_on_error: bool = Field(
        default=True, description="Continue execution despite GUI problems"
    )
    platform_specific_advice: bool = Field(
        default=True, description="Include platform-specific advice"
    )


class QontinuiProperties(BaseModel):
    """Centralized configuration properties for the Qontinui framework.

    Modern configuration framework using Pydantic.

    This class provides a modern, type-safe approach to framework configuration,
    with validation and environment variable support.

    Properties are organized into logical groups:
    - Core: Essential framework settings like image paths and mock mode
    - Mouse: Mouse action timing and behavior configuration
    - Mock: Simulated execution timings for testing
    - Screenshot: Screen capture and history settings
    - Illustration: Visual feedback and annotation settings
    - Analysis: Color profiling and k-means clustering settings
    - Recording: Screen recording configuration
    - Dataset: AI training data generation settings
    - Testing: Test execution configuration
    - Monitor: Monitor configuration settings
    - DPI: DPI and scaling configuration
    - Capture: Screen capture provider configuration
    - Sikuli: SikuliX integration settings
    - Startup: Startup configuration
    - Automation: Automation failure handling
    - AutoScaling: Automatic pattern scaling
    - Logging: Comprehensive logging configuration
    - Highlight: Visual highlighting configuration
    - Console: Console action reporting
    - ImageDebug: Image debugging configuration
    - GuiAccess: GUI access verification

    Example usage:
        # Load from environment variables
        config = QontinuiProperties()

        # Load from dict
        config = QontinuiProperties(**config_dict)

        # Load from YAML
        import yaml
        with open('config.yaml') as f:
            config = QontinuiProperties(**yaml.safe_load(f))

        # Access nested properties
        print(config.mouse.move_delay)

        # Update properties (with validation)
        config.core.mock = True
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    core: CoreConfig = Field(default_factory=CoreConfig, description="Core framework settings")
    mouse: MouseConfig = Field(
        default_factory=MouseConfig, description="Mouse action configuration"
    )
    mock: MockConfig = Field(
        default_factory=MockConfig, description="Mock mode timing configuration"
    )
    screenshot: ScreenshotConfig = Field(
        default_factory=ScreenshotConfig, description="Screenshot and history settings"
    )
    illustration: IllustrationConfig = Field(
        default_factory=IllustrationConfig, description="Action illustration settings"
    )
    analysis: AnalysisConfig = Field(
        default_factory=AnalysisConfig, description="Color analysis settings"
    )
    recording: RecordingConfig = Field(
        default_factory=RecordingConfig, description="Screen recording settings"
    )
    dataset: DatasetConfig = Field(
        default_factory=DatasetConfig, description="AI dataset generation settings"
    )
    testing: TestingConfig = Field(
        default_factory=TestingConfig, description="Test execution settings"
    )
    monitor: MonitorConfig = Field(
        default_factory=MonitorConfig, description="Monitor configuration settings"
    )
    dpi: DpiConfig = Field(default_factory=DpiConfig, description="DPI and scaling configuration")
    capture: CaptureConfig = Field(
        default_factory=CaptureConfig, description="Screen capture provider configuration"
    )
    sikuli: SikuliConfig = Field(
        default_factory=SikuliConfig, description="SikuliX integration settings"
    )
    startup: StartupConfig = Field(
        default_factory=StartupConfig, description="Startup configuration"
    )
    automation: AutomationConfig = Field(
        default_factory=AutomationConfig, description="Automation failure handling"
    )
    autoscaling: AutoScalingConfig = Field(
        default_factory=AutoScalingConfig, description="Automatic pattern scaling"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Comprehensive logging configuration"
    )
    highlight: HighlightConfig = Field(
        default_factory=HighlightConfig, description="Visual highlighting configuration"
    )
    console: ConsoleActionConfig = Field(
        default_factory=ConsoleActionConfig, description="Console action reporting"
    )
    image_debug: ImageDebugConfig = Field(
        default_factory=ImageDebugConfig, description="Image debugging configuration"
    )
    gui_access: GuiAccessConfig = Field(
        default_factory=GuiAccessConfig, description="GUI access verification"
    )

    def to_yaml(self, path: Path | None = None) -> str:
        """Export configuration to YAML format.

        Args:
            path: Optional path to save YAML file

        Returns:
            YAML string representation
        """
        import yaml

        yaml_str = yaml.dump(self.model_dump(), default_flow_style=False)

        if path:
            path.write_text(yaml_str)

        return yaml_str

    def to_env_file(self, path: Path | None = None) -> str:
        """Export configuration to .env format.

        Args:
            path: Optional path to save .env file

        Returns:
            Environment variable format string
        """
        lines = []

        def flatten_dict(d, prefix="QONTINUI"):
            for key, value in d.items():
                env_key = f"{prefix}__{key.upper()}"
                if isinstance(value, dict):
                    flatten_dict(value, env_key)
                elif isinstance(value, list):
                    lines.append(f"{env_key}={','.join(map(str, value))}")
                else:
                    lines.append(f"{env_key}={value}")

        flatten_dict(self.model_dump())
        env_str = "\n".join(lines)

        if path:
            path.write_text(env_str)

        return env_str

    @classmethod
    def from_yaml(cls, path: Path) -> "QontinuiProperties":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            QontinuiProperties instance
        """
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_env_file(cls, path: Path) -> "QontinuiProperties":
        """Load configuration from .env file.

        Args:
            path: Path to .env file

        Returns:
            QontinuiProperties instance
        """
        from dotenv import dotenv_values

        env_vars = dotenv_values(path)

        # Parse environment variables into nested dict
        config: dict[str, Any] = {}
        for key, value in env_vars.items():
            if key.startswith("QONTINUI__") and value is not None:
                parts = key[10:].lower().split("__")
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Convert value types
                parsed_value: Any = value
                if value is not None:
                    if value.lower() in ("true", "false"):
                        parsed_value = value.lower() == "true"
                    elif value.isdigit():
                        parsed_value = int(value)
                    elif "." in value and value.replace(".", "").isdigit():
                        parsed_value = float(value)
                    elif "," in value:
                        parsed_value = value.split(",")

                current[parts[-1]] = parsed_value

        return cls(**config)
