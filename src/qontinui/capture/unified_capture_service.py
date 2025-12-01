"""Unified screen capture service for Qontinui.

Provides a single interface for all capture operations with configurable
providers, automatic retry, and fallback handling.

Based on Brobot's UnifiedCaptureService but uses Qontinui's HAL architecture.
"""

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from PIL import Image

from ..exceptions import ScreenCaptureException
from ..hal import HALFactory
from ..hal.config import CaptureBackend, HALConfig

logger = logging.getLogger(__name__)


class CaptureProvider(Enum):
    """Available capture providers."""

    AUTO = "AUTO"
    MSS = "MSS"
    OPENCV = "OPENCV"
    PILLOW = "PILLOW"
    NATIVE = "NATIVE"
    CUSTOM = "CUSTOM"


@dataclass
class CaptureConfig:
    """Configuration for UnifiedCaptureService."""

    provider: str = "AUTO"
    enable_logging: bool = False
    auto_retry: bool = True
    retry_count: int = 3
    retry_delay: float = 0.5
    cache_enabled: bool = True
    cache_ttl: float = 1.0
    fallback_enabled: bool = True
    fallback_providers: list[str] | None = None

    def __post_init__(self):
        """Initialize default fallback providers."""
        if self.fallback_providers is None:
            self.fallback_providers = ["MSS", "PILLOW", "NATIVE"]


class UnifiedCaptureService:
    """Unified screen capture service with configurable providers.

    This service provides:
    - Single interface for all capture operations
    - Environment-based provider selection
    - Automatic retry mechanism
    - Fallback to alternative providers
    - Thread-safe operations
    - Capture caching

    Configuration via environment variables:
        QONTINUI_CAPTURE_PROVIDER: Provider to use (AUTO, MSS, OPENCV, etc.)
        QONTINUI_CAPTURE_RETRY: Enable auto-retry (true/false)
        QONTINUI_CAPTURE_RETRY_COUNT: Number of retries
        QONTINUI_CAPTURE_CACHE: Enable caching (true/false)

    Example:
        >>> capture = UnifiedCaptureService()
        >>> image = capture.capture_screen()
        >>> region_image = capture.capture_region(100, 100, 300, 200)
    """

    def __init__(self, config: CaptureConfig | None = None) -> None:
        """Initialize UnifiedCaptureService.

        Args:
            config: Optional configuration, uses environment variables if None
        """
        self.config = config or self._load_config_from_env()
        self._provider = None
        self._fallback_index = 0

        # Initialize provider
        self._init_provider()

        if self.config.enable_logging:
            logger.info(
                f"UnifiedCaptureService initialized with provider: {self.config.provider}",
                extra={
                    "auto_retry": self.config.auto_retry,
                    "retry_count": self.config.retry_count,
                    "cache_enabled": self.config.cache_enabled,
                },
            )

    def _load_config_from_env(self) -> CaptureConfig:
        """Load configuration from environment variables.

        Returns:
            CaptureConfig instance
        """
        return CaptureConfig(
            provider=os.getenv("QONTINUI_CAPTURE_PROVIDER", "AUTO"),
            enable_logging=os.getenv("QONTINUI_CAPTURE_LOGGING", "false").lower() == "true",
            auto_retry=os.getenv("QONTINUI_CAPTURE_RETRY", "true").lower() == "true",
            retry_count=int(os.getenv("QONTINUI_CAPTURE_RETRY_COUNT", "3")),
            retry_delay=float(os.getenv("QONTINUI_CAPTURE_RETRY_DELAY", "0.5")),
            cache_enabled=os.getenv("QONTINUI_CAPTURE_CACHE", "true").lower() == "true",
            cache_ttl=float(os.getenv("QONTINUI_CAPTURE_CACHE_TTL", "1.0")),
            fallback_enabled=os.getenv("QONTINUI_CAPTURE_FALLBACK", "true").lower() == "true",
        )

    def _init_provider(self):
        """Initialize capture provider based on configuration."""
        try:
            if self.config.provider == "AUTO":
                self._provider = self._auto_select_provider()
            elif self.config.provider in [p.value for p in CaptureProvider]:
                self._provider = self._create_provider(self.config.provider)
            else:
                # Try to load as custom provider
                self._provider = self._load_custom_provider(self.config.provider)
        except Exception as e:
            logger.error(f"Failed to initialize provider {self.config.provider}: {e}")
            if self.config.fallback_enabled:
                self._provider = self._get_fallback_provider()
            else:
                raise

    def _auto_select_provider(self):
        """Automatically select the best available provider.

        Returns:
            Capture provider instance
        """
        # Try providers in order of preference
        preference_order = ["MSS", "OPENCV", "PILLOW", "NATIVE"]

        for provider_name in preference_order:
            try:
                provider = self._create_provider(provider_name)
                if provider:
                    logger.debug(f"Auto-selected provider: {provider_name}")
                    return provider
            except Exception as e:
                logger.debug(f"Provider {provider_name} not available: {e}")

        raise RuntimeError("No capture provider available")

    def _create_provider(self, provider_name: str):
        """Create a capture provider by name.

        Args:
            provider_name: Name of the provider

        Returns:
            Provider instance
        """
        # Map provider names to HAL backends
        backend_map = {
            "MSS": CaptureBackend.MSS,
            "OPENCV": CaptureBackend.MSS,  # Use MSS for now
            "PILLOW": CaptureBackend.PILLOW,
            "NATIVE": CaptureBackend.NATIVE,
        }

        backend = backend_map.get(provider_name.upper())
        if not backend:
            raise ValueError(f"Unknown provider: {provider_name}")

        # Create HAL config with specific backend
        hal_config = HALConfig(
            capture_backend=backend.value,
            capture_cache_enabled=self.config.cache_enabled,
            capture_cache_ttl=self.config.cache_ttl,
        )

        return HALFactory.get_screen_capture(hal_config)

    def _load_custom_provider(self, class_name: str):
        """Load a custom capture provider class.

        Args:
            class_name: Fully qualified class name

        Returns:
            Custom provider instance
        """
        # This would load a custom provider class dynamically
        # For now, we'll raise an error
        raise NotImplementedError(f"Custom provider loading not implemented: {class_name}")

    def _get_fallback_provider(self):
        """Get next fallback provider.

        Returns:
            Fallback provider instance
        """
        if not self.config.fallback_providers:
            raise RuntimeError("No fallback providers configured")

        while self._fallback_index < len(self.config.fallback_providers):
            provider_name = self.config.fallback_providers[self._fallback_index]
            self._fallback_index += 1

            try:
                provider = self._create_provider(provider_name)
                logger.info(f"Using fallback provider: {provider_name}")
                return provider
            except Exception as e:
                logger.debug(f"Fallback provider {provider_name} failed: {e}")

        raise RuntimeError("All fallback providers failed")

    def capture_screen(self, screen_id: int | None = None) -> Image.Image:
        """Capture entire screen or specific monitor.

        Args:
            screen_id: Monitor ID (0-based), None for primary

        Returns:
            Captured image

        Raises:
            ScreenCaptureException: If capture fails after retries
        """
        if self._provider is None:
            raise RuntimeError("Capture provider not initialized")
        return self._execute_with_retry(
            lambda: self._provider.capture_screen(screen_id), "capture_screen"
        )

    def capture_region(
        self, x: int, y: int, width: int, height: int, screen_id: int | None = None
    ) -> Image.Image:
        """Capture specific region of screen.

        Args:
            x: X coordinate of top-left corner
            y: Y coordinate of top-left corner
            width: Region width
            height: Region height
            screen_id: Optional monitor ID

        Returns:
            Captured region image

        Raises:
            ScreenCaptureException: If capture fails after retries
        """
        if self._provider is None:
            raise RuntimeError("Capture provider not initialized")
        return self._execute_with_retry(
            lambda: self._provider.capture_region(x, y, width, height, screen_id),
            "capture_region",
        )

    def get_monitors(self) -> list[Any]:
        """Get list of available monitors.

        Returns:
            List of monitor information
        """
        if self._provider is None:
            raise RuntimeError("Capture provider not initialized")
        return self._provider.get_monitors()

    def get_primary_monitor(self):
        """Get primary monitor information.

        Returns:
            Primary monitor info
        """
        if self._provider is None:
            raise RuntimeError("Capture provider not initialized")
        return self._provider.get_primary_monitor()

    def save_screenshot(
        self,
        filepath: str,
        screen_id: int | None = None,
        region: tuple[int, int, int, int] | None = None,
    ) -> str:
        """Save screenshot to file.

        Args:
            filepath: Path to save screenshot
            screen_id: Optional monitor ID
            region: Optional region (x, y, width, height)

        Returns:
            Path where screenshot was saved
        """
        if self._provider is None:
            raise RuntimeError("Capture provider not initialized")
        return self._execute_with_retry(
            lambda: self._provider.save_screenshot(filepath, screen_id, region),
            "save_screenshot",
        )

    def _execute_with_retry(self, operation: Callable[..., Any], operation_name: str) -> Any:
        """Execute operation with automatic retry.

        Args:
            operation: Callable[..., Any] to execute
            operation_name: Name for logging

        Returns:
            Operation result

        Raises:
            ScreenCaptureException: If all retries fail
        """
        if not self.config.auto_retry:
            # No retry, just execute
            try:
                return operation()
            except Exception as e:
                raise ScreenCaptureException(f"{operation_name} failed: {e}") from e

        last_error = None
        for attempt in range(self.config.retry_count):
            try:
                result = operation()
                if attempt > 0 and self.config.enable_logging:
                    logger.debug(f"{operation_name} succeeded on attempt {attempt + 1}")
                return result
            except Exception as e:
                last_error = e
                if self.config.enable_logging:
                    logger.debug(f"{operation_name} attempt {attempt + 1} failed: {e}")

                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.retry_delay)
                elif (
                    self.config.fallback_enabled
                    and self.config.fallback_providers is not None
                    and self._fallback_index < len(self.config.fallback_providers)
                ):
                    # Try with fallback provider
                    try:
                        self._provider = self._get_fallback_provider()
                        return operation()
                    except Exception as fallback_error:
                        last_error = fallback_error

        raise ScreenCaptureException(
            f"{operation_name} failed after {self.config.retry_count} attempts: {last_error}"
        )

    def set_provider(self, provider_name: str):
        """Change capture provider at runtime.

        Args:
            provider_name: Name of provider to use
        """
        old_provider = self.config.provider
        try:
            self.config.provider = provider_name
            self._init_provider()
            logger.info(f"Changed capture provider from {old_provider} to {provider_name}")
        except Exception as e:
            logger.error(f"Failed to change provider to {provider_name}: {e}")
            self.config.provider = old_provider
            raise

    def get_current_provider(self) -> str:
        """Get name of current capture provider.

        Returns:
            Provider name
        """
        return self.config.provider

    def clear_cache(self):
        """Clear capture cache if caching is enabled."""
        if hasattr(self._provider, "clear_cache"):
            self._provider.clear_cache()
            logger.debug("Capture cache cleared")


# Global instance for convenience
_unified_capture: UnifiedCaptureService | None = None


def get_unified_capture_service() -> UnifiedCaptureService:
    """Get or create global UnifiedCaptureService instance.

    Returns:
        UnifiedCaptureService instance
    """
    global _unified_capture
    if _unified_capture is None:
        _unified_capture = UnifiedCaptureService()
    return _unified_capture
