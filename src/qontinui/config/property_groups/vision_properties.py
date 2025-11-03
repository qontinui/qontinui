"""Vision and image finding configuration properties.

Contains settings for pattern scaling, color analysis, and image debugging
for visual element recognition.
"""

from pydantic import BaseModel, ConfigDict, Field


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

    # Debug data emission
    emit_match_details: bool = Field(
        default=False,
        description="When True, emit detailed match info including top N matches and debug data",
    )


class VisionProperties(BaseModel):
    """Vision and image finding configuration properties.

    Includes:
    - AutoScalingConfig: Automatic pattern scaling detection
    - AnalysisConfig: Color analysis and profiling
    - ImageDebugConfig: Image debugging and visualization
    """

    model_config = ConfigDict(validate_assignment=True)

    autoscaling: AutoScalingConfig = Field(
        default_factory=AutoScalingConfig, description="Automatic pattern scaling"
    )
    analysis: AnalysisConfig = Field(
        default_factory=AnalysisConfig, description="Color analysis settings"
    )
    image_debug: ImageDebugConfig = Field(
        default_factory=ImageDebugConfig, description="Image debugging configuration"
    )
