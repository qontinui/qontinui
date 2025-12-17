"""Screen capture interface definition."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass
class Monitor:
    """Monitor information."""

    index: int
    x: int
    y: int
    width: int
    height: int
    scale: float = 1.0
    is_primary: bool = False
    name: str | None = None

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Get monitor bounds as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)

    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is within monitor bounds."""
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height


class IScreenCapture(ABC):
    """Interface for screen capture operations."""

    @abstractmethod
    def capture_screen(self, monitor: int | None = None) -> Image.Image:
        """Capture entire screen or specific monitor.

        Args:
            monitor: Monitor index (0-based), None for all monitors

        Returns:
            PIL Image of screenshot
        """
        pass

    @abstractmethod
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
        """
        pass

    @abstractmethod
    def get_monitors(self) -> list[Monitor]:
        """Get list of available monitors.

        Returns:
            List of Monitor objects
        """
        pass

    @abstractmethod
    def get_primary_monitor(self) -> Monitor:
        """Get primary monitor.

        Returns:
            Primary Monitor object
        """
        pass

    @abstractmethod
    def get_screen_size(self) -> tuple[int, int]:
        """Get screen size.

        Returns:
            Tuple of (width, height) in pixels
        """
        pass

    @abstractmethod
    def get_pixel_color(
        self, x: int, y: int, monitor: int | None = None
    ) -> tuple[int, int, int]:
        """Get color of pixel at coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            monitor: Optional monitor index

        Returns:
            RGB color tuple
        """
        pass

    @abstractmethod
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
        """
        pass
