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
