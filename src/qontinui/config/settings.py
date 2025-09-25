"""Configuration management for Qontinui using pydantic-settings.

This replaces Brobot's Spring-based configuration with a Python-native solution
that supports environment variables, .env files, and type validation.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class QontinuiSettings(BaseSettings):
    """Main configuration settings for Qontinui framework."""

    # Core settings
    timeout: float = Field(30.0, description="Default timeout in seconds for operations")
    retry_count: int = Field(3, description="Number of retries for failed operations")
    retry_delay: float = Field(1.0, description="Delay between retries in seconds")
    debug_mode: bool = Field(False, description="Enable debug logging and features")

    # Perception settings
    similarity_threshold: float = Field(
        0.85, ge=0.0, le=1.0, description="Default similarity threshold for matching"
    )
    segmentation_model: str = Field("SAM2", description="Model to use for segmentation")
    embedding_model: str = Field("CLIP", description="Model to use for embeddings")
    ocr_engine: str = Field("easyocr", description="OCR engine to use")

    # Vector DB settings
    vector_db_type: Literal["faiss", "qdrant"] = Field(
        "faiss", description="Vector database backend"
    )
    vector_index_path: Path = Field(Path("./vectors"), description="Path to store vector indices")
    vector_dimension: int = Field(512, description="Dimension of vector embeddings")

    # Screen settings
    screenshot_delay: float = Field(0.1, description="Delay after taking screenshot")
    multi_monitor: bool = Field(False, description="Enable multi-monitor support")
    dpi_scaling: float = Field(1.0, description="DPI scaling factor")
    default_monitor: int | None = Field(None, description="Default monitor index")

    # RAG settings
    rag_enabled: bool = Field(True, description="Enable RAG-based state recognition")
    semantic_weight: float = Field(0.6, ge=0.0, le=1.0, description="Weight for semantic matching")
    deterministic_weight: float = Field(
        0.4, ge=0.0, le=1.0, description="Weight for deterministic matching"
    )
    min_semantic_confidence: float = Field(
        0.6, ge=0.0, le=1.0, description="Minimum confidence for semantic matches"
    )

    # Action settings
    action_delay: float = Field(0.0, description="Default delay between actions")
    mouse_move_duration: float = Field(0.5, description="Duration for mouse movements")
    typing_delay: float = Field(0.05, description="Delay between key presses")
    safe_mode: bool = Field(True, description="Enable safety checks before destructive actions")

    # State management settings
    state_cache_size: int = Field(100, description="Maximum number of states to cache")
    state_transition_timeout: float = Field(60.0, description="Timeout for state transitions")
    auto_discover_states: bool = Field(False, description="Enable automatic state discovery")

    # Performance settings
    parallel_actions: bool = Field(True, description="Enable parallel action execution")
    max_workers: int = Field(4, description="Maximum number of worker threads")
    batch_size: int = Field(32, description="Batch size for vector operations")
    use_gpu: bool = Field(True, description="Use GPU if available")

    # Storage settings
    data_path: Path = Field(Path("./data"), description="Path for data storage")
    cache_path: Path = Field(Path("./cache"), description="Path for cache storage")
    log_path: Path = Field(Path("./logs"), description="Path for log files")

    # Monitoring settings
    metrics_enabled: bool = Field(True, description="Enable metrics collection")
    metrics_port: int = Field(8000, description="Port for metrics endpoint")
    health_check_interval: float = Field(30.0, description="Interval for health checks in seconds")

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_prefix = "QONTINUI_"
        case_sensitive = False
        extra = "forbid"

    def validate_weights(self) -> None:
        """Validate that semantic and deterministic weights sum to 1.0."""
        total = self.semantic_weight + self.deterministic_weight
        if abs(total - 1.0) > 0.001:  # Allow small floating point errors
            raise ValueError(f"Semantic and deterministic weights must sum to 1.0, got {total}")

    def model_post_init(self, __context) -> None:
        """Post-initialization validation."""
        self.validate_weights()

        # Create directories if they don't exist
        for path_field in ["vector_index_path", "data_path", "cache_path", "log_path"]:
            path = getattr(self, path_field)
            if path and not path.exists():
                path.mkdir(parents=True, exist_ok=True)


class DevelopmentSettings(QontinuiSettings):
    """Development-specific settings."""

    debug_mode: bool = True
    log_level: str = "DEBUG"
    save_screenshots: bool = True
    verbose_logging: bool = True

    class Config:
        env_file = ".env.development"


class ProductionSettings(QontinuiSettings):
    """Production-specific settings."""

    debug_mode: bool = False
    log_level: str = "INFO"
    safe_mode: bool = True
    retry_count: int = 5

    class Config:
        env_file = ".env.production"


class TestSettings(QontinuiSettings):
    """Test-specific settings."""

    timeout: float = 5.0
    retry_count: int = 1
    mock_hardware: bool = True
    data_path: Path = Path("./test_data")
    cache_path: Path = Path("./test_cache")
    log_path: Path = Path("./test_logs")

    class Config:
        env_file = ".env.test"


# Singleton instance
_settings: QontinuiSettings | None = None


def get_settings(env: str | None = None) -> QontinuiSettings:
    """Get the singleton settings instance.

    Args:
        env: Environment name ('development', 'production', 'test')

    Returns:
        QontinuiSettings instance
    """
    global _settings

    if _settings is None:
        if env == "development":
            _settings = DevelopmentSettings()
        elif env == "production":
            _settings = ProductionSettings()
        elif env == "test":
            _settings = TestSettings()
        else:
            # Default to base settings, auto-detect from env
            import os

            env_name = os.getenv("QONTINUI_ENV", "development")
            if env_name == "development":
                _settings = DevelopmentSettings()
            elif env_name == "production":
                _settings = ProductionSettings()
            elif env_name == "test":
                _settings = TestSettings()
            else:
                _settings = QontinuiSettings()

    return _settings


def reset_settings() -> None:
    """Reset the settings singleton (mainly for testing)."""
    global _settings
    _settings = None
