"""Configuration for UI-TARS integration.

This module provides settings for UI-TARS exploration and execution,
supporting cloud (HuggingFace) and local (transformers, vLLM) inference.
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class UITARSSettings(BaseSettings):
    """Settings for UI-TARS exploration and execution.

    Configure via environment variables with QONTINUI_UITARS_ prefix.

    Examples:
        # Local 2B model on GTX 1080
        QONTINUI_UITARS_PROVIDER=local_transformers
        QONTINUI_UITARS_MODEL_SIZE=2B
        QONTINUI_UITARS_QUANTIZATION=int4

        # Cloud HuggingFace endpoint
        QONTINUI_UITARS_PROVIDER=cloud
        QONTINUI_UITARS_HUGGINGFACE_ENDPOINT=https://...
        QONTINUI_UITARS_HUGGINGFACE_API_TOKEN=hf_xxxxx
    """

    model_config = SettingsConfigDict(
        env_prefix="QONTINUI_UITARS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider selection
    provider: Literal["cloud", "local_vllm", "local_transformers"] = Field(
        default="local_transformers",
        description="Inference provider: cloud (HuggingFace), local_transformers, or local_vllm",
    )
    model_size: str = Field(
        default="2B",
        description="Model size: 2B, 7B, or 72B. Use 2B for local inference on consumer GPUs.",
    )

    # Cloud (HuggingFace Inference Endpoints)
    huggingface_endpoint: str | None = Field(
        default=None,
        description="HuggingFace Inference Endpoint URL",
    )
    huggingface_api_token: str | None = Field(
        default=None,
        description="HuggingFace API token for authentication",
    )
    huggingface_model_id: str = Field(
        default="ByteDance-Seed/UI-TARS-2B-SFT",
        description="HuggingFace model ID (used when endpoint is not specified)",
    )

    # Local inference settings
    quantization: Literal["none", "int8", "int4"] = Field(
        default="int4",
        description="Quantization for local inference. int4 recommended for GTX 1080 (8GB).",
    )
    device: str = Field(
        default="auto",
        description="Device for local inference: auto, cuda, cuda:0, cpu",
    )
    torch_dtype: str = Field(
        default="auto",
        description="Torch dtype: auto, float16, bfloat16, float32",
    )

    # vLLM-specific settings
    vllm_server_url: str = Field(
        default="http://localhost:8000",
        description="URL of running vLLM server",
    )
    vllm_model_name: str | None = Field(
        default=None,
        description="Model name as registered in vLLM (defaults to model_size mapping)",
    )

    # Exploration settings
    max_exploration_steps: int = Field(
        default=50,
        description="Maximum steps during exploration",
    )
    exploration_timeout_seconds: int = Field(
        default=600,
        description="Total timeout for exploration session",
    )
    step_timeout_seconds: float = Field(
        default=30.0,
        description="Timeout for a single exploration step",
    )
    save_screenshots: bool = Field(
        default=True,
        description="Save screenshots during exploration",
    )
    screenshot_format: Literal["png", "jpg"] = Field(
        default="png",
        description="Format for saved screenshots",
    )

    # Execution settings
    execution_mode: Literal["local", "uitars", "hybrid"] = Field(
        default="local",
        description=(
            "Execution mode: 'local' uses RAG/template (DEFAULT), "
            "'uitars' always uses UI-TARS, "
            "'hybrid' tries local first then falls back to UI-TARS"
        ),
    )
    uitars_fallback_enabled: bool = Field(
        default=True,
        description="Allow UI-TARS fallback in hybrid mode when local confidence is low",
    )
    uitars_execution_timeout: float = Field(
        default=10.0,
        description="Timeout per UI-TARS grounding request in seconds",
    )
    confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence for local grounding before UI-TARS fallback",
    )

    # Inference parameters
    max_new_tokens: int = Field(
        default=512,
        description="Maximum new tokens to generate",
    )
    temperature: float = Field(
        default=0.0,
        description="Sampling temperature (0.0 for greedy decoding)",
    )
    top_p: float = Field(
        default=1.0,
        description="Top-p (nucleus) sampling parameter",
    )

    # System prompt (UI-TARS specific)
    system_prompt: str = Field(
        default=(
            "You are a GUI automation assistant. Analyze the screenshot and "
            "perform the requested action. Output your reasoning as Thought: "
            "and the action as Action:. Use absolute pixel coordinates."
        ),
        description="System prompt for UI-TARS inference",
    )

    def get_model_id(self) -> str:
        """Get the full model ID based on model_size setting."""
        model_map = {
            "2B": "ByteDance-Seed/UI-TARS-2B-SFT",
            "7B": "ByteDance-Seed/UI-TARS-7B-SFT",
            "72B": "ByteDance-Seed/UI-TARS-72B-SFT",
        }
        return model_map.get(self.model_size, self.model_size)

    def is_local(self) -> bool:
        """Check if using local inference."""
        return self.provider in ("local_transformers", "local_vllm")

    def is_cloud(self) -> bool:
        """Check if using cloud inference."""
        return self.provider == "cloud"

    def needs_quantization(self) -> bool:
        """Check if quantization is configured."""
        return self.quantization != "none"
