"""CaptureWrapper - Routes screen capture to mock or real implementations (Brobot pattern).

This wrapper provides the routing layer for screen capture operations,
delegating to either MockCapture (pre-captured screenshots) or HAL implementations
(real-time screen capture via MSS) based on ExecutionMode.

Architecture:
    FindImage/StateDetector (high-level)
      ↓
    CaptureWrapper (this layer) ← Routes based on ExecutionMode
      ↓
    ├─ if mock → MockCapture → Cached screenshots → Returns PIL.Image
    └─ if real → HAL Layer → MSSScreenCapture → Returns PIL.Image
"""

import logging
from typing import TYPE_CHECKING, Optional

from PIL import Image

from .base import BaseWrapper

if TYPE_CHECKING:
    from ..hal.interfaces.screen_capture import Monitor
    from ..model.element.region import Region

logger = logging.getLogger(__name__)


class CaptureWrapper(BaseWrapper):
    """Wrapper for screen capture operations.

    Routes screen capture operations to either mock or real implementations
    based on ExecutionMode. This follows the Brobot pattern where high-level
    code is agnostic to whether it's running in mock or real mode.

    Example:
        # Initialize wrapper
        wrapper = CaptureWrapper()

        # Capture screen (automatically routed to mock or real)
        screenshot = wrapper.capture()

        # High-level code doesn't know or care whether this used:
        # - MockCapture.get_cached_screenshot() → PIL.Image from file
        # - HAL MSSScreenCapture.capture_screen() → PIL.Image from live screen

    Attributes:
        mock_capture: MockCapture instance for cached screenshots
        hal_capture: Screen capture for real mode
    """

    def __init__(self) -> None:
        """Initialize CaptureWrapper.

        Sets up both mock and real implementations. The actual implementation
        used is determined at runtime based on ExecutionMode.
        """
        super().__init__()

        # Lazy initialization to avoid circular imports
        self._mock_capture = None
        self._hal_capture = None

        logger.debug("CaptureWrapper initialized")

    @property
    def mock_capture(self):
        """Get MockCapture instance (lazy initialization).

        Returns:
            MockCapture instance
        """
        if self._mock_capture is None:
            from ..mock.mock_capture import MockCapture

            self._mock_capture = MockCapture()
            logger.debug("MockCapture initialized")
        return self._mock_capture

    @property
    def hal_capture(self):
        """Get HAL screen capture (lazy initialization).

        Returns:
            IScreenCapture implementation (MSSScreenCapture)
        """
        if self._hal_capture is None:
            from ..hal.factory import HALFactory

            self._hal_capture = HALFactory.get_screen_capture()
            logger.debug("HAL screen capture initialized")
        return self._hal_capture

    def capture(self, monitor: int | None = None) -> Image.Image:
        """Capture entire screen or specific monitor.

        Routes to MockCapture or HAL based on ExecutionMode.

        Args:
            monitor: Monitor index (0-based), None for all monitors

        Returns:
            PIL Image of screenshot

        Example:
            wrapper = CaptureWrapper()
            screenshot = wrapper.capture()
            screenshot.save("screen.png")
        """
        if self.is_mock_mode():
            logger.debug(f"CaptureWrapper.capture (MOCK): monitor={monitor}")
            return self.mock_capture.capture(monitor)
        else:
            logger.debug(f"CaptureWrapper.capture (REAL): monitor={monitor}")
            screenshot = self.hal_capture.capture_screen(monitor)

            # Record screenshot if recording is enabled
            self._record_screenshot(screenshot)

            return screenshot

    def capture_region(
        self,
        region: "Region",
        monitor: int | None = None,
    ) -> Image.Image:
        """Capture specific region.

        Routes to MockCapture or HAL based on ExecutionMode.

        Args:
            region: Region to capture
            monitor: Optional monitor index

        Returns:
            PIL Image of region

        Example:
            from qontinui.model.element.region import Region
            wrapper = CaptureWrapper()
            region = Region(100, 200, 300, 400)
            screenshot = wrapper.capture_region(region)
        """
        if self.is_mock_mode():
            logger.debug(f"CaptureWrapper.capture_region (MOCK): {region}")
            return self.mock_capture.capture_region(region, monitor)
        else:
            logger.debug(f"CaptureWrapper.capture_region (REAL): {region}")
            screenshot = self.hal_capture.capture_region(
                x=region.x,
                y=region.y,
                width=region.w,
                height=region.h,
                monitor=monitor,
            )

            # Record screenshot if recording is enabled
            self._record_screenshot(screenshot)

            return screenshot

    def get_monitors(self) -> list["Monitor"]:
        """Get list of available monitors.

        Routes to MockCapture or HAL based on ExecutionMode.

        Returns:
            List of Monitor objects

        Example:
            wrapper = CaptureWrapper()
            monitors = wrapper.get_monitors()
            for i, mon in enumerate(monitors):
                print(f"Monitor {i}: {mon.width}x{mon.height}")
        """
        if self.is_mock_mode():
            logger.debug("CaptureWrapper.get_monitors (MOCK)")
            return self.mock_capture.get_monitors()
        else:
            logger.debug("CaptureWrapper.get_monitors (REAL)")
            return self.hal_capture.get_monitors()

    def get_primary_monitor(self) -> "Monitor":
        """Get primary monitor.

        Routes to MockCapture or HAL based on ExecutionMode.

        Returns:
            Primary Monitor object

        Example:
            wrapper = CaptureWrapper()
            primary = wrapper.get_primary_monitor()
            print(f"Primary: {primary.width}x{primary.height}")
        """
        if self.is_mock_mode():
            logger.debug("CaptureWrapper.get_primary_monitor (MOCK)")
            return self.mock_capture.get_primary_monitor()
        else:
            logger.debug("CaptureWrapper.get_primary_monitor (REAL)")
            return self.hal_capture.get_primary_monitor()

    def get_screen_size(self) -> tuple[int, int]:
        """Get screen size.

        Routes to MockCapture or HAL based on ExecutionMode.

        Returns:
            Tuple of (width, height) in pixels

        Example:
            wrapper = CaptureWrapper()
            width, height = wrapper.get_screen_size()
            print(f"Screen: {width}x{height}")
        """
        if self.is_mock_mode():
            logger.debug("CaptureWrapper.get_screen_size (MOCK)")
            return self.mock_capture.get_screen_size()
        else:
            logger.debug("CaptureWrapper.get_screen_size (REAL)")
            return self.hal_capture.get_screen_size()

    def get_pixel_color(
        self,
        x: int,
        y: int,
        monitor: int | None = None,
    ) -> tuple[int, int, int]:
        """Get color of pixel at coordinates.

        Routes to MockCapture or HAL based on ExecutionMode.

        Args:
            x: X coordinate
            y: Y coordinate
            monitor: Optional monitor index

        Returns:
            RGB color tuple

        Example:
            wrapper = CaptureWrapper()
            color = wrapper.get_pixel_color(100, 200)
            print(f"Color at (100, 200): RGB{color}")
        """
        if self.is_mock_mode():
            logger.debug(f"CaptureWrapper.get_pixel_color (MOCK): ({x}, {y})")
            return self.mock_capture.get_pixel_color(x, y, monitor)
        else:
            logger.debug(f"CaptureWrapper.get_pixel_color (REAL): ({x}, {y})")
            return self.hal_capture.get_pixel_color(x, y, monitor)

    def save_screenshot(
        self,
        filepath: str,
        monitor: int | None = None,
        region: Optional["Region"] = None,
    ) -> str:
        """Save screenshot to file.

        Routes to MockCapture or HAL based on ExecutionMode.

        Args:
            filepath: Path to save screenshot
            monitor: Optional monitor to capture
            region: Optional region to capture

        Returns:
            Path where screenshot was saved

        Example:
            wrapper = CaptureWrapper()
            path = wrapper.save_screenshot("test.png")
            print(f"Saved to: {path}")
        """
        if self.is_mock_mode():
            logger.debug(f"CaptureWrapper.save_screenshot (MOCK): {filepath}")
            return self.mock_capture.save_screenshot(filepath, monitor, region)
        else:
            logger.debug(f"CaptureWrapper.save_screenshot (REAL): {filepath}")
            region_tuple = None
            if region:
                region_tuple = (region.x, region.y, region.w, region.h)
            return self.hal_capture.save_screenshot(filepath, monitor, region_tuple)

    def _record_screenshot(self, screenshot: Image.Image):
        """Record a screenshot if recording is enabled.

        Args:
            screenshot: Screenshot image (PIL Image)
        """
        # Get controller and check if recording
        from .controller import get_controller

        controller = get_controller()

        if not controller.is_recording():
            return

        # Record screenshot
        try:
            controller.recorder.record_screenshot(screenshot)
            logger.debug("Recorded screenshot")
        except Exception as e:
            logger.error(f"Failed to record screenshot: {e}", exc_info=True)
