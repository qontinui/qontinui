"""Configuration for OmniParser integration.

Settings are driven by environment variables with QONTINUI_OMNIPARSER_ prefix,
following the same pattern as UITARSSettings.
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OmniParserSettings(BaseSettings):
    """Settings for OmniParser zero-shot UI element detection.

    Configure via environment variables with QONTINUI_OMNIPARSER_ prefix.

    Examples:
        # Enable with auto GPU detection
        QONTINUI_OMNIPARSER_ENABLED=true

        # Force CPU mode
        QONTINUI_OMNIPARSER_ENABLED=true
        QONTINUI_OMNIPARSER_DEVICE=cpu

        # Use remote service instead of local models
        QONTINUI_OMNIPARSER_PROVIDER=service
        QONTINUI_OMNIPARSER_SERVICE_URL=http://localhost:8080
    """

    model_config = SettingsConfigDict(
        env_prefix="QONTINUI_OMNIPARSER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = Field(
        default=False,
        description="Enable OmniParser detection backend. Off by default (requires GPU for best performance).",
    )

    provider: Literal["local", "service"] = Field(
        default="local",
        description="Inference provider: 'local' runs models in-process, 'service' connects to remote HTTP endpoint.",
    )

    # Model settings (local provider)
    model_path: str | None = Field(
        default=None,
        description="Path to custom OmniParser model weights. If None, downloads from HuggingFace.",
    )
    device: str = Field(
        default="auto",
        description="Device for inference: 'auto', 'cuda', 'cuda:0', 'cpu'.",
    )
    yolo_model: str = Field(
        default="microsoft/OmniParser-v2.0",
        description="HuggingFace model ID or local path for the YOLO detection model.",
    )
    caption_model: str = Field(
        default="microsoft/Florence-2-base",
        description="HuggingFace model ID or local path for the Florence-2 captioning model.",
    )

    # Detection parameters
    iou_threshold: float = Field(
        default=0.3,
        description="IoU threshold for YOLO Non-Maximum Suppression.",
    )
    confidence_threshold: float = Field(
        default=0.3,
        description="Minimum YOLO detection confidence.",
    )
    caption_batch_size: int = Field(
        default=64,
        description="Batch size for Florence-2 icon captioning.",
    )

    # Resource management
    lazy_load: bool = Field(
        default=True,
        description="Load models only on first detection call.",
    )
    unload_after_seconds: float = Field(
        default=0.0,
        description="Unload models after N seconds of inactivity. 0 = never unload.",
    )

    # Service provider settings
    service_url: str = Field(
        default="http://localhost:8080",
        description="URL of remote OmniParser HTTP endpoint.",
    )
    service_timeout: float = Field(
        default=30.0,
        description="Timeout in seconds for service requests.",
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
