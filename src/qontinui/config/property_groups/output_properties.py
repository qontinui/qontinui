"""Output and capture configuration properties.

Contains settings for screenshots, screen recordings, and dataset generation
for AI training.
"""

from pydantic import BaseModel, ConfigDict, Field


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


class OutputProperties(BaseModel):
    """Output and capture configuration properties.

    Includes:
    - ScreenshotConfig: Screenshot capture and history
    - RecordingConfig: Screen recording settings
    - DatasetConfig: AI dataset generation
    """

    model_config = ConfigDict(validate_assignment=True)

    screenshot: ScreenshotConfig = Field(
        default_factory=ScreenshotConfig, description="Screenshot and history settings"
    )
    recording: RecordingConfig = Field(
        default_factory=RecordingConfig, description="Screen recording settings"
    )
    dataset: DatasetConfig = Field(
        default_factory=DatasetConfig, description="AI dataset generation settings"
    )
