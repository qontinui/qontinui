"""HAL configuration management."""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CaptureBackend(Enum):
    """Available screen capture backends."""

    MSS = "mss"
    PILLOW = "pillow"
    NATIVE = "native"
    PYAUTOGUI = "pyautogui"  # Legacy fallback


class InputBackend(Enum):
    """Available input control backends."""

    PYNPUT = "pynput"
    NATIVE = "native"
    SELENIUM = "selenium"
    PYAUTOGUI = "pyautogui"  # Legacy fallback


class MatcherBackend(Enum):
    """Available pattern matching backends."""

    OPENCV = "opencv"
    TENSORFLOW = "tensorflow"
    NATIVE = "native"
    PYAUTOGUI = "pyautogui"  # Legacy fallback


class OCRBackend(Enum):
    """Available OCR backends."""

    EASYOCR = "easyocr"
    TESSERACT = "tesseract"
    CLOUD = "cloud"
    NONE = "none"


class PlatformOverride(Enum):
    """Platform override options."""

    AUTO = "auto"
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"


@dataclass
class HALConfig:
    """HAL configuration settings.

    Configuration can be set via:
    1. Environment variables (QONTINUI_* prefix)
    2. Direct instantiation
    3. Configuration file
    """

    # Backend selections
    capture_backend: str = field(
        default_factory=lambda: os.getenv("QONTINUI_CAPTURE_BACKEND", CaptureBackend.MSS.value)
    )
    input_backend: str = field(
        default_factory=lambda: os.getenv("QONTINUI_INPUT_BACKEND", InputBackend.PYNPUT.value)
    )
    matcher_backend: str = field(
        default_factory=lambda: os.getenv("QONTINUI_MATCHER_BACKEND", MatcherBackend.OPENCV.value)
    )
    ocr_backend: str = field(
        default_factory=lambda: os.getenv("QONTINUI_OCR_BACKEND", OCRBackend.EASYOCR.value)
    )
    platform_override: str = field(
        default_factory=lambda: os.getenv("QONTINUI_PLATFORM_OVERRIDE", PlatformOverride.AUTO.value)
    )

    # Performance settings
    capture_cache_enabled: bool = field(
        default_factory=lambda: os.getenv("QONTINUI_CAPTURE_CACHE_ENABLED", "true").lower()
        == "true"
    )
    capture_cache_ttl: float = field(
        default_factory=lambda: float(os.getenv("QONTINUI_CAPTURE_CACHE_TTL", "1.0"))
    )
    matcher_threads: int = field(
        default_factory=lambda: int(os.getenv("QONTINUI_MATCHER_THREADS", "4"))
    )
    ocr_gpu_enabled: bool = field(
        default_factory=lambda: os.getenv("QONTINUI_OCR_GPU_ENABLED", "false").lower() == "true"
    )

    # Fallback settings
    use_fallback: bool = field(
        default_factory=lambda: os.getenv("QONTINUI_USE_FALLBACK", "true").lower() == "true"
    )
    fallback_to_pyautogui: bool = field(
        default_factory=lambda: os.getenv("QONTINUI_FALLBACK_TO_PYAUTOGUI", "false").lower()
        == "true"
    )

    # Debug settings
    debug_mode: bool = field(
        default_factory=lambda: os.getenv("QONTINUI_HAL_DEBUG", "false").lower() == "true"
    )
    log_performance: bool = field(
        default_factory=lambda: os.getenv("QONTINUI_LOG_PERFORMANCE", "false").lower() == "true"
    )

    # Feature flags
    enable_multi_monitor: bool = field(
        default_factory=lambda: os.getenv("QONTINUI_MULTI_MONITOR", "true").lower() == "true"
    )
    enable_dpi_scaling: bool = field(
        default_factory=lambda: os.getenv("QONTINUI_DPI_SCALING", "true").lower() == "true"
    )

    # Paths
    screenshot_dir: str = field(
        default_factory=lambda: os.getenv("QONTINUI_SCREENSHOT_DIR", "./screenshots")
    )
    temp_dir: str = field(default_factory=lambda: os.getenv("QONTINUI_TEMP_DIR", "/tmp/qontinui"))

    def validate(self) -> bool:
        """Validate configuration settings.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate backend selections
        if self.capture_backend not in [b.value for b in CaptureBackend]:
            raise ValueError(f"Invalid capture backend: {self.capture_backend}")

        if self.input_backend not in [b.value for b in InputBackend]:
            raise ValueError(f"Invalid input backend: {self.input_backend}")

        if self.matcher_backend not in [b.value for b in MatcherBackend]:
            raise ValueError(f"Invalid matcher backend: {self.matcher_backend}")

        if self.ocr_backend not in [b.value for b in OCRBackend]:
            raise ValueError(f"Invalid OCR backend: {self.ocr_backend}")

        if self.platform_override not in [p.value for p in PlatformOverride]:
            raise ValueError(f"Invalid platform override: {self.platform_override}")

        # Validate numeric settings
        if self.capture_cache_ttl < 0:
            raise ValueError("Cache TTL must be non-negative")

        if self.matcher_threads < 1:
            raise ValueError("Matcher threads must be at least 1")

        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "capture_backend": self.capture_backend,
            "input_backend": self.input_backend,
            "matcher_backend": self.matcher_backend,
            "ocr_backend": self.ocr_backend,
            "platform_override": self.platform_override,
            "capture_cache_enabled": self.capture_cache_enabled,
            "capture_cache_ttl": self.capture_cache_ttl,
            "matcher_threads": self.matcher_threads,
            "ocr_gpu_enabled": self.ocr_gpu_enabled,
            "use_fallback": self.use_fallback,
            "fallback_to_pyautogui": self.fallback_to_pyautogui,
            "debug_mode": self.debug_mode,
            "log_performance": self.log_performance,
            "enable_multi_monitor": self.enable_multi_monitor,
            "enable_dpi_scaling": self.enable_dpi_scaling,
            "screenshot_dir": self.screenshot_dir,
            "temp_dir": self.temp_dir,
        }

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "HALConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            HALConfig instance
        """
        return cls(**config_dict)

    def __str__(self) -> str:
        """String representation."""
        return (
            f"HALConfig(capture={self.capture_backend}, "
            f"input={self.input_backend}, "
            f"matcher={self.matcher_backend}, "
            f"ocr={self.ocr_backend})"
        )


# Global configuration instance
_config: HALConfig | None = None


def get_config() -> HALConfig:
    """Get global HAL configuration.

    Returns:
        HALConfig instance
    """
    global _config
    if _config is None:
        _config = HALConfig()
        _config.validate()
    return _config


def set_config(config: HALConfig) -> None:
    """Set global HAL configuration.

    Args:
        config: New configuration
    """
    global _config
    config.validate()
    _config = config


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _config
    _config = None
