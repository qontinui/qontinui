"""MSS-based screen capture implementation."""

import time
from pathlib import Path

import mss
import mss.tools
from PIL import Image

from ...exceptions import ScreenCaptureException
from ...logging import get_logger
from ..config import HALConfig
from ..interfaces.screen_capture import IScreenCapture, Monitor

logger = get_logger(__name__)


class MSSScreenCapture(IScreenCapture):
    """Fast screen capture implementation using MSS.

    MSS provides direct system-level screen capture without dependencies
    on GUI automation libraries, resulting in 10-50x faster captures.
    """

    def __init__(self, config: HALConfig | None = None):
        """Initialize MSS screen capture.

        Args:
            config: HAL configuration
        """
        self.config = config or HALConfig()
        self._sct = None  # Will be lazily initialized per thread
        self._monitors = self._detect_monitors()

        # Caching
        self._cache: dict[str, Image.Image] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._cache_enabled = self.config.capture_cache_enabled
        self._cache_ttl = self.config.capture_cache_ttl

        logger.info(
            "mss_capture_initialized",
            monitor_count=len(self._monitors),
            cache_enabled=self._cache_enabled,
        )

    @property
    def sct(self) -> mss.mss:
        """Get or create thread-local mss instance.

        This ensures each thread has its own mss instance to avoid
        thread-local storage issues with Windows GDI.
        """
        import threading

        if not hasattr(self, '_thread_local'):
            self._thread_local = threading.local()

        if not hasattr(self._thread_local, 'sct'):
            self._thread_local.sct = mss.mss()
            logger.debug("mss_instance_created", thread_id=threading.current_thread().ident)

        return self._thread_local.sct

    def _detect_monitors(self) -> list[Monitor]:
        """Detect all available monitors.

        Returns:
            List of Monitor objects
        """
        monitors = []

        # Skip index 0 as it's the combined virtual monitor
        for i, mon in enumerate(self.sct.monitors[1:], 1):
            monitor = Monitor(
                index=i - 1,  # 0-based index
                x=mon["left"],
                y=mon["top"],
                width=mon["width"],
                height=mon["height"],
                scale=1.0 if not self.config.enable_dpi_scaling else self._get_dpi_scale(),
                is_primary=(i == 1),  # First monitor is usually primary
                name=f"Monitor {i}",
            )
            monitors.append(monitor)

            logger.debug(
                "monitor_detected",
                index=monitor.index,
                bounds=monitor.bounds,
                is_primary=monitor.is_primary,
            )

        return monitors

    def _get_dpi_scale(self) -> float:
        """Get DPI scaling factor for current system.

        Returns:
            DPI scale factor
        """
        try:
            # Platform-specific DPI detection
            import sys

            if sys.platform == "win32":
                import ctypes

                user32 = ctypes.windll.user32
                user32.SetProcessDPIAware()
                dpi = user32.GetDpiForSystem()
                return dpi / 96.0
            elif sys.platform == "darwin":
                # macOS typically uses 2x for Retina displays
                # This would need proper detection via PyObjC
                return 1.0
            else:
                # Linux X11
                return 1.0
        except Exception as e:
            logger.debug(f"Could not detect DPI scale: {e}")
            return 1.0

    def capture_screen(self, monitor: int | None = None) -> Image.Image:
        """Capture entire screen or specific monitor.

        Args:
            monitor: Monitor index (0-based), None for all monitors

        Returns:
            PIL Image of screenshot

        Raises:
            ScreenCaptureException: If capture fails
        """
        try:
            # Check cache
            cache_key = f"screen_{monitor}"
            if self._cache_enabled and self._is_cached(cache_key):
                return self._cache[cache_key]

            # Determine capture region
            if monitor is None:
                if self.config.enable_multi_monitor:
                    # Capture all monitors (virtual desktop)
                    sct_img = self.sct.grab(self.sct.monitors[0])
                else:
                    # Capture primary monitor
                    sct_img = self.sct.grab(self.sct.monitors[1])
            else:
                # Capture specific monitor
                if 0 <= monitor < len(self._monitors):
                    mon_dict = self.sct.monitors[monitor + 1]  # +1 for mss indexing
                    sct_img = self.sct.grab(mon_dict)
                else:
                    raise ValueError(f"Invalid monitor index: {monitor}")

            # Convert to PIL Image
            image = Image.frombytes(
                "RGB", (sct_img.width, sct_img.height), sct_img.bgra, "raw", "BGRX"
            )

            # Update cache
            if self._cache_enabled:
                self._update_cache(cache_key, image)

            logger.debug("screen_captured", monitor=monitor, size=(image.width, image.height))

            return image

        except Exception as e:
            raise ScreenCaptureException(
                f"Failed to capture screen (monitor={monitor}): {e}"
            ) from e

    def capture_region(
        self, x: int, y: int, width: int, height: int, monitor: int | None = None
    ) -> Image.Image:
        """Capture specific region.

        Args:
            x: X coordinate of top-left corner
            y: Y coordinate of top-left corner
            width: Region width in pixels
            height: Region height in pixels
            monitor: Optional monitor index for relative coordinates

        Returns:
            PIL Image of region

        Raises:
            ScreenCaptureException: If capture fails
        """
        try:
            # Build region dict for mss
            region = {"left": x, "top": y, "width": width, "height": height}

            # Adjust for monitor offset if specified
            if monitor is not None:
                if 0 <= monitor < len(self._monitors):
                    mon = self.sct.monitors[monitor + 1]
                    region["left"] += mon["left"]
                    region["top"] += mon["top"]
                else:
                    raise ValueError(f"Invalid monitor index: {monitor}")

            # Check cache
            cache_key = f"region_{x}_{y}_{width}_{height}_{monitor}"
            if self._cache_enabled and self._is_cached(cache_key):
                return self._cache[cache_key]

            # Capture region
            sct_img = self.sct.grab(region)

            # Convert to PIL Image
            image = Image.frombytes(
                "RGB", (sct_img.width, sct_img.height), sct_img.bgra, "raw", "BGRX"
            )

            # Update cache
            if self._cache_enabled:
                self._update_cache(cache_key, image)

            logger.debug("region_captured", region=(x, y, width, height), monitor=monitor)

            return image

        except Exception as e:
            raise ScreenCaptureException(
                f"Failed to capture region (monitor={monitor}): {e}"
            ) from e

    def get_monitors(self) -> list[Monitor]:
        """Get list of available monitors.

        Returns:
            List of Monitor objects
        """
        return self._monitors.copy()

    def get_primary_monitor(self) -> Monitor:
        """Get primary monitor.

        Returns:
            Primary Monitor object

        Raises:
            RuntimeError: If no monitors are available
        """
        for monitor in self._monitors:
            if monitor.is_primary:
                return monitor
        # Fallback to first monitor
        if self._monitors:
            return self._monitors[0]
        raise RuntimeError("No monitors available")

    def get_screen_size(self) -> tuple[int, int]:
        """Get screen size.

        Returns:
            Tuple of (width, height) in pixels
        """
        primary = self.get_primary_monitor()
        return (primary.width, primary.height)

    def get_pixel_color(self, x: int, y: int, monitor: int | None = None) -> tuple[int, int, int]:
        """Get color of pixel at coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            monitor: Optional monitor index

        Returns:
            RGB color tuple
        """
        try:
            # Adjust coordinates for monitor
            if monitor is not None:
                if 0 <= monitor < len(self._monitors):
                    mon = self._monitors[monitor]
                    x += mon.x
                    y += mon.y
                else:
                    raise ValueError(f"Invalid monitor index: {monitor}")

            # Capture single pixel
            region = {"left": x, "top": y, "width": 1, "height": 1}

            sct_img = self.sct.grab(region)

            # Extract RGB values from BGRA
            # MSS returns BGRA format
            pixel = sct_img.pixel(0, 0)  # Get pixel at (0,0) of 1x1 capture
            return pixel[:3]  # Return RGB, ignore alpha

        except Exception as e:
            logger.error("get_pixel_color_failed", x=x, y=y, monitor=monitor, error=str(e))
            return (0, 0, 0)

    def save_screenshot(
        self,
        filepath: str,
        monitor: int | None = None,
        region: tuple[int, int, int, int] | None = None,
    ) -> str:
        """Save screenshot to file.

        Args:
            filepath: Path to save screenshot
            monitor: Optional monitor to capture
            region: Optional region (x, y, width, height)

        Returns:
            Path where screenshot was saved

        Raises:
            ScreenCaptureException: If save fails
        """
        try:
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Capture image
            if region:
                image = self.capture_region(*region, monitor=monitor)
            else:
                image = self.capture_screen(monitor=monitor)

            # Save image
            image.save(path)

            logger.info("screenshot_saved", path=str(path), size=(image.width, image.height))

            return str(path)

        except Exception as e:
            raise ScreenCaptureException(
                f"Failed to save screenshot (monitor={monitor}): {e}"
            ) from e

    def _is_cached(self, key: str) -> bool:
        """Check if cache entry is valid.

        Args:
            key: Cache key

        Returns:
            True if cached and valid
        """
        if key not in self._cache:
            return False

        timestamp = self._cache_timestamps.get(key, 0)
        age = time.time() - timestamp

        if age > self._cache_ttl:
            # Expired, remove from cache
            del self._cache[key]
            del self._cache_timestamps[key]
            return False

        return True

    def _update_cache(self, key: str, image: Image.Image) -> None:
        """Update cache entry.

        Args:
            key: Cache key
            image: Image to cache
        """
        self._cache[key] = image
        self._cache_timestamps[key] = time.time()

        # Limit cache size
        max_cache_size = 50
        if len(self._cache) > max_cache_size:
            # Remove oldest entries
            sorted_keys = sorted(
                self._cache_timestamps.keys(), key=lambda k: self._cache_timestamps[k]
            )
            for old_key in sorted_keys[: len(self._cache) - max_cache_size]:
                del self._cache[old_key]
                del self._cache_timestamps[old_key]

    def clear_cache(self) -> None:
        """Clear screenshot cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.debug("screenshot_cache_cleared")

    def close(self) -> None:
        """Close screen capture resources."""
        if hasattr(self, '_thread_local') and hasattr(self._thread_local, 'sct'):
            try:
                self._thread_local.sct.close()
            except Exception as e:
                logger.debug(f"Error closing mss instance: {e}")
        self.clear_cache()
        logger.debug("mss_capture_closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        try:
            self.close()
        except Exception:
            pass
