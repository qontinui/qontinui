"""Display and visual configuration properties.

Contains settings for visual feedback, highlighting, screen capture,
monitor configuration, and DPI scaling.
"""

from pydantic import BaseModel, ConfigDict, Field


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
        default=True,
        description="Disable DPI awareness to force physical resolution capture",
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
        default=True,
        description="Enable fallback to other providers if preferred fails",
    )
    fallback_chain: list[str] = Field(
        default=["MSS", "ROBOT"],
        description="Fallback chain priority for capture providers",
    )
    enable_logging: bool = Field(default=False, description="Enable capture operation logging")
    auto_retry: bool = Field(default=True, description="Auto-retry failed captures")
    retry_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retry attempts for failed captures",
    )


class DisplayProperties(BaseModel):
    """Display and visual configuration properties.

    Includes:
    - IllustrationConfig: Action illustration settings
    - HighlightConfig: Visual highlighting configuration
    - MonitorConfig: Monitor configuration
    - DpiConfig: DPI and scaling settings
    - CaptureConfig: Screen capture provider settings
    """

    model_config = ConfigDict(validate_assignment=True)

    illustration: IllustrationConfig = Field(
        default_factory=IllustrationConfig, description="Action illustration settings"
    )
    highlight: HighlightConfig = Field(
        default_factory=HighlightConfig, description="Visual highlighting configuration"
    )
    monitor: MonitorConfig = Field(
        default_factory=MonitorConfig, description="Monitor configuration settings"
    )
    dpi: DpiConfig = Field(default_factory=DpiConfig, description="DPI and scaling configuration")
    capture: CaptureConfig = Field(
        default_factory=CaptureConfig,
        description="Screen capture provider configuration",
    )
