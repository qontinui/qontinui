"""Physical screen management - ported from Qontinui framework.

Screen implementation that captures at physical resolution regardless of DPI scaling.
"""

import logging
import tkinter as tk
from dataclasses import dataclass

from PIL import Image, ImageGrab

from ..model.element.region import Region

logger = logging.getLogger(__name__)


@dataclass
class PhysicalScreen:
    """Screen implementation that captures at physical resolution.

    Port of PhysicalScreen from Qontinui framework class.

    This ensures Qontinui captures screenshots at the same resolution as the actual display,
    solving pattern matching issues caused by DPI scaling.
    """

    physical_width: int
    """Physical screen width in pixels."""

    physical_height: int
    """Physical screen height in pixels."""

    needs_scaling: bool
    """Whether DPI scaling compensation is needed."""

    scale_factor: float
    """Scale factor for DPI compensation."""

    def __init__(self) -> None:
        """Initialize physical screen with resolution detection."""
        # Try to get physical resolution
        self.physical_width, self.physical_height = self._get_physical_resolution()

        # Get logical resolution for comparison
        logical_width, logical_height = self._get_logical_resolution()

        # Check if scaling is needed
        self.needs_scaling = (
            self.physical_width != logical_width
            or self.physical_height != logical_height
        )

        if self.needs_scaling:
            self.scale_factor = self.physical_width / logical_width
            logger.info("PhysicalScreen: Compensating for DPI scaling")
            logger.info(f"  Physical: {self.physical_width}x{self.physical_height}")
            logger.info(f"  Logical:  {logical_width}x{logical_height}")
            logger.info(f"  Scale Factor: {self.scale_factor}")
        else:
            self.scale_factor = 1.0

    def _get_physical_resolution(self) -> tuple[int, int]:
        """Get physical screen resolution.

        Returns:
            (width, height) tuple
        """
        try:
            # Try platform-specific methods
            import platform

            if platform.system() == "Windows":
                import ctypes

                user32 = ctypes.windll.user32  # type: ignore[attr-defined]
                user32.SetProcessDPIAware()
                width = user32.GetSystemMetrics(0)
                height = user32.GetSystemMetrics(1)
                return width, height
            elif platform.system() == "Darwin":  # macOS
                # macOS typically reports physical resolution correctly
                root = tk.Tk()
                root.withdraw()
                width = root.winfo_screenwidth()
                height = root.winfo_screenheight()
                root.destroy()
                return width, height
            else:  # Linux
                # Try xrandr or similar
                root = tk.Tk()
                root.withdraw()
                width = root.winfo_screenwidth()
                height = root.winfo_screenheight()
                root.destroy()
                return width, height
        except (OSError, RuntimeError, ValueError, ImportError, AttributeError) as e:
            logger.warning(f"Failed to get physical resolution: {e}")
            return self._get_logical_resolution()

    def _get_logical_resolution(self) -> tuple[int, int]:
        """Get logical screen resolution (DPI-scaled).

        Returns:
            (width, height) tuple
        """
        try:
            root = tk.Tk()
            root.withdraw()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            return width, height
        except (OSError, RuntimeError, ValueError, ImportError):
            # Fallback to common resolution
            return 1920, 1080

    def capture(self, region: Region | None = None) -> Image.Image:
        """Capture screenshot at physical resolution.

        Args:
            region: Optional region to capture

        Returns:
            PIL Image of the captured region
        """
        if region is None:
            return self._capture_physical_resolution(
                0, 0, self.physical_width, self.physical_height
            )

        if self.needs_scaling:
            # Scale logical coordinates to physical
            x = int(region.x * self.scale_factor)
            y = int(region.y * self.scale_factor)
            w = int(region.width * self.scale_factor)
            h = int(region.height * self.scale_factor)
        else:
            x, y, w, h = region.x, region.y, region.width, region.height

        return self._capture_physical_resolution(x, y, w, h)

    def _capture_physical_resolution(
        self, x: int, y: int, w: int, h: int
    ) -> Image.Image:
        """Capture at physical resolution.

        Args:
            x: X coordinate
            y: Y coordinate
            w: Width
            h: Height

        Returns:
            PIL Image of the captured region
        """
        try:
            # Ensure bounds are within screen
            x = max(0, min(x, self.physical_width - 1))
            y = max(0, min(y, self.physical_height - 1))
            w = min(w, self.physical_width - x)
            h = min(h, self.physical_height - y)

            if w <= 0 or h <= 0:
                logger.warning(f"Invalid capture region: {x},{y} {w}x{h}")
                return Image.new("RGB", (1, 1))

            # Capture using PIL
            bbox = (x, y, x + w, y + h)
            screenshot = ImageGrab.grab(bbox=bbox)

            return screenshot

        except (OSError, RuntimeError, ValueError, MemoryError) as e:
            logger.error(f"Failed to capture screen: {e}")
            return Image.new("RGB", (w if w > 0 else 1, h if h > 0 else 1))

    def get_bounds(self) -> Region:
        """Get the screen bounds as a Region.

        Returns:
            Region representing the full screen
        """
        return Region(x=0, y=0, width=self.physical_width, height=self.physical_height)
