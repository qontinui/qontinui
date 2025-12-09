"""Screenshot provider implementation using PureActions.

Captures screenshots using the HAL abstraction through PureActions.
"""

from PIL import Image

from ...hal.interfaces import IScreenCapture
from ...model.element import Region
from .screenshot_provider import ScreenshotProvider


class PureActionsScreenshotProvider(ScreenshotProvider):
    """Screenshot provider using PureActions for capture.

    Uses the HAL screen capture interface through PureActions to capture
    screenshots. Supports capturing the entire screen or specific regions.
    """

    def __init__(self, screen_capture: IScreenCapture | None = None) -> None:
        """Initialize provider with screen capture interface.

        Args:
            screen_capture: Optional screen capture interface.
                           If None, uses HAL factory default.
        """
        if screen_capture is None:
            from ...hal.factory import HALFactory

            screen_capture = HALFactory.get_screen_capture()
        self.screen = screen_capture

    def capture(self, region: Region | None = None, monitor: int | None = None) -> Image.Image:
        """Capture screenshot using PureActions.

        Args:
            region: Optional region to capture. If None, captures entire screen.
            monitor: Optional monitor index (0-based). If None, captures all monitors
                    (virtual desktop) when multi-monitor is enabled.

        Returns:
            PIL Image of the captured screenshot.

        Raises:
            RuntimeError: If screenshot capture fails.
        """
        try:
            if region is not None:
                # Capture specific region (monitor offset is applied by the HAL)
                image = self.screen.capture_region(
                    region.x, region.y, region.width, region.height, monitor=monitor
                )
            else:
                # Capture screen - specific monitor or all monitors
                image = self.screen.capture_screen(monitor=monitor)

            if image is None:
                raise RuntimeError("Screenshot capture returned None")

            return image

        except Exception as e:
            raise RuntimeError(f"Screenshot capture failed: {e}") from e
