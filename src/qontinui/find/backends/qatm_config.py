"""Configuration for QATM (Quality-Aware Template Matching) backend.

Settings are driven by environment variables with QONTINUI_QATM_ prefix,
following the same pattern as OmniParserSettings.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QATMSettings(BaseSettings):
    """Settings for QATM deep template matching backend.

    Configure via environment variables with QONTINUI_QATM_ prefix.

    Examples:
        # Enable with auto GPU detection
        QONTINUI_QATM_ENABLED=true

        # Force CPU mode
        QONTINUI_QATM_ENABLED=true
        QONTINUI_QATM_DEVICE=cpu

        # Custom confidence threshold
        QONTINUI_QATM_CONFIDENCE_THRESHOLD=0.6
    """

    model_config = SettingsConfigDict(
        env_prefix="QONTINUI_QATM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = Field(
        default=False,
        description="Enable QATM detection backend. Off by default (requires PyTorch).",
    )

    device: str = Field(
        default="auto",
        description="Device for inference: 'auto', 'cuda', 'cuda:0', 'cpu'.",
    )

    confidence_threshold: float = Field(
        default=0.7,
        description="Minimum quality-aware confidence to accept a match.",
    )

    feature_layer: str = Field(
        default="relu4_1",
        description=(
            "VGG-19 layer for feature extraction. "
            "Lower layers (relu3_1) capture texture; "
            "higher layers (relu5_1) capture semantics."
        ),
    )

    lazy_load: bool = Field(
        default=True,
        description="Load VGG-19 model only on first detection call.",
    )

    unload_after_seconds: float = Field(
        default=300.0,
        description="Unload model after N seconds of inactivity. 0 = never unload.",
    )

    alpha: float = Field(
        default=25.0,
        description="QATM softmax temperature. Higher = sharper quality discrimination.",
    )

    def resolve_device(self) -> str:
        """Resolve 'auto' device to actual device string."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
