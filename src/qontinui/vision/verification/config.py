"""Vision verification configuration.

Provides configuration classes for controlling vision verification behavior
including detection thresholds, timeouts, and comparison settings.
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DetectionConfig(BaseModel):
    """Configuration for element detection."""

    # Template matching
    template_threshold: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for template matches",
    )
    template_method: str = Field(
        "TM_CCOEFF_NORMED",
        description="OpenCV template matching method",
    )
    multi_scale: bool = Field(
        False,
        description="Enable multi-scale template matching",
    )
    scale_range: tuple[float, float] = Field(
        (0.8, 1.2),
        description="Scale range for multi-scale matching",
    )
    scale_steps: int = Field(
        5,
        description="Number of scale steps",
    )

    # OCR settings
    ocr_engine: str = Field(
        "easyocr",
        description="OCR engine: 'easyocr', 'tesseract', 'auto'",
    )
    ocr_language: str = Field(
        "en",
        description="OCR language code",
    )
    ocr_confidence_threshold: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Minimum OCR confidence",
    )
    ocr_gpu: bool = Field(
        False,
        description="Use GPU for OCR (EasyOCR)",
    )


class WaitConfig(BaseModel):
    """Configuration for wait operations."""

    default_timeout: int = Field(
        5000,
        gt=0,
        description="Default timeout in milliseconds",
    )
    polling_interval: int = Field(
        100,
        gt=0,
        description="Polling interval in milliseconds",
    )
    stability_duration: int = Field(
        500,
        gt=0,
        description="Duration to check for stability in ms",
    )


class ScreenshotConfig(BaseModel):
    """Configuration for screenshot handling."""

    capture_on_failure: bool = Field(
        True,
        description="Capture screenshot on assertion failure",
    )
    annotate_failures: bool = Field(
        True,
        description="Annotate screenshots with failure info",
    )
    save_directory: str = Field(
        ".dev-logs/screenshots",
        description="Directory for saving screenshots",
    )
    baseline_dir: str = Field(
        ".dev-logs/baselines",
        description="Directory for baseline screenshots",
    )
    diff_dir: str = Field(
        ".dev-logs/diffs",
        description="Directory for diff screenshots",
    )
    annotation_color: tuple[int, int, int] = Field(
        (0, 0, 255),
        description="BGR color for annotations (default: red)",
    )
    annotation_thickness: int = Field(
        2,
        description="Line thickness for annotations",
    )
    max_saved: int = Field(
        100,
        description="Maximum screenshots to keep",
    )


class ComparisonConfig(BaseModel):
    """Configuration for visual comparison."""

    # Default comparison settings
    default_threshold: float = Field(
        0.95,
        ge=0.0,
        le=1.0,
        description="Default similarity threshold for comparisons",
    )
    default_method: str = Field(
        "ssim",
        description="Default comparison method: 'pixel', 'ssim', 'phash', 'feature'",
    )

    # SSIM settings
    ssim_threshold: float = Field(
        0.95,
        ge=0.0,
        le=1.0,
        description="SSIM similarity threshold",
    )
    ssim_multichannel: bool = Field(
        True,
        description="Use multichannel SSIM",
    )

    # Pixel diff settings
    pixel_threshold: int = Field(
        10,
        ge=0,
        le=255,
        description="Per-pixel difference threshold",
    )
    diff_percentage_threshold: float = Field(
        0.01,
        ge=0.0,
        le=1.0,
        description="Maximum allowed different pixels percentage",
    )

    # Color settings
    color_tolerance: int = Field(
        10,
        ge=0,
        le=255,
        description="Color comparison tolerance (0-255)",
    )

    # Position settings
    position_tolerance: int = Field(
        5,
        ge=0,
        description="Position tolerance in pixels",
    )

    # Perceptual hash
    hash_size: int = Field(
        16,
        ge=4,
        description="Size for perceptual hash",
    )
    hash_threshold: int = Field(
        10,
        ge=0,
        description="Maximum hamming distance for hash comparison",
    )


class EnvironmentConfig(BaseModel):
    """Configuration for GUI environment integration."""

    use_environment: bool = Field(
        True,
        description="Use discovered GUI environment",
    )
    environment_path: str | None = Field(
        None,
        description="Path to environment JSON file",
    )
    auto_mask_dynamic: bool = Field(
        True,
        description="Auto-mask discovered dynamic regions",
    )
    use_semantic_colors: bool = Field(
        True,
        description="Use discovered semantic colors",
    )
    use_learned_states: bool = Field(
        True,
        description="Use learned visual states for detection",
    )
    use_discovered_regions: bool = Field(
        True,
        description="Use discovered layout regions",
    )


class VisionConfig(BaseModel):
    """Complete vision verification configuration."""

    detection: DetectionConfig = Field(
        default_factory=DetectionConfig,
        description="Detection settings",
    )
    wait: WaitConfig = Field(
        default_factory=WaitConfig,
        description="Wait settings",
    )
    screenshot: ScreenshotConfig = Field(
        default_factory=ScreenshotConfig,
        description="Screenshot settings",
    )
    comparison: ComparisonConfig = Field(
        default_factory=ComparisonConfig,
        description="Comparison settings",
    )
    environment: EnvironmentConfig = Field(
        default_factory=EnvironmentConfig,
        description="Environment integration settings",
    )

    # Debug settings
    debug_mode: bool = Field(
        False,
        description="Enable debug mode with verbose logging",
    )
    save_intermediate: bool = Field(
        False,
        description="Save intermediate results for debugging",
    )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisionConfig":
        """Create config from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            VisionConfig instance.
        """
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Configuration dictionary.
        """
        return self.model_dump()

    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file.

        Args:
            path: Output file path.
        """
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "VisionConfig":
        """Load configuration from JSON file.

        Args:
            path: Input file path.

        Returns:
            VisionConfig instance.
        """
        import json

        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return cls.model_validate(data)


# Default configuration instance
DEFAULT_CONFIG = VisionConfig()


def get_default_config() -> VisionConfig:
    """Get the default configuration.

    Returns:
        Default VisionConfig instance.
    """
    return DEFAULT_CONFIG.model_copy(deep=True)


__all__ = [
    "DetectionConfig",
    "WaitConfig",
    "ScreenshotConfig",
    "ComparisonConfig",
    "EnvironmentConfig",
    "VisionConfig",
    "DEFAULT_CONFIG",
    "get_default_config",
]
