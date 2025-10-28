"""Abstract base interface for screenshot capture.

Defines the contract for screenshot capture implementations.
"""

from abc import ABC, abstractmethod

from PIL import Image

from ...model.element import Region


class ScreenshotProvider(ABC):
    """Abstract base for screenshot capture implementations.

    Provides a clean interface for capturing screenshots of the entire screen
    or specific regions. Implementations may use different capture mechanisms
    or add performance optimizations like caching.
    """

    @abstractmethod
    def capture(self, region: Region | None = None) -> Image.Image:
        """Capture screenshot of entire screen or region.

        Args:
            region: Optional region to capture. If None, captures entire screen.

        Returns:
            PIL Image of the captured screenshot.

        Raises:
            RuntimeError: If screenshot capture fails.
        """
        pass
