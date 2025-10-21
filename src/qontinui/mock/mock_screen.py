"""Mock screen capture for testing without actual screen operations.

Based on Brobot's mock pattern - simulates screen capture instantly.
"""

import hashlib
import logging
from datetime import datetime
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from ..hal.interfaces.screen_capture import IScreenCapture, Monitor

logger = logging.getLogger(__name__)


class MockScreen(IScreenCapture):
    """Mock implementation of screen capture for testing.

    All operations complete instantly without actual screen capture.
    Returns synthetic images for pattern matching in mock mode.
    """

    def __init__(self, width: int = 1920, height: int = 1080):
        """Initialize mock screen capture.

        Args:
            width: Mock screen width
            height: Mock screen height
        """
        self._width = width
        self._height = height
        self._capture_count = 0
        self._mock_screens: dict[str, np.ndarray] = {}
        logger.debug(f"MockScreen initialized ({width}x{height})")

    def capture_screen(self, monitor_index: int = 0) -> np.ndarray:
        """Mock capture entire screen (instant).

        Returns:
            Synthetic screen image as numpy array
        """
        self._capture_count += 1
        logger.debug(f"[MOCK] Screen capture #{self._capture_count} (monitor {monitor_index})")

        # Return synthetic blank screen
        # In real mock mode, this would be populated with expected patterns
        screen = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        return screen

    def capture_region(
        self, x: int, y: int, width: int, height: int, monitor_index: int = 0
    ) -> np.ndarray:
        """Mock capture screen region (instant).

        Args:
            x: Region X coordinate
            y: Region Y coordinate
            width: Region width
            height: Region height
            monitor_index: Monitor index

        Returns:
            Synthetic region image as numpy array
        """
        self._capture_count += 1
        logger.debug(
            f"[MOCK] Region capture #{self._capture_count} "
            f"({x}, {y}, {width}x{height}) on monitor {monitor_index}"
        )

        # Return synthetic blank region
        region = np.zeros((height, width, 3), dtype=np.uint8)
        return region

    def save_screenshot(
        self, file_path: str, monitor_index: int = 0, region: tuple[int, int, int, int] | None = None
    ) -> bool:
        """Mock save screenshot to file (instant).

        Args:
            file_path: Output file path
            monitor_index: Monitor index
            region: Optional region (x, y, width, height)

        Returns:
            True (always succeeds in mock mode)
        """
        logger.debug(f"[MOCK] Save screenshot to {file_path}")

        if region:
            x, y, width, height = region
            image = self.capture_region(x, y, width, height, monitor_index)
        else:
            image = self.capture_screen(monitor_index)

        # Actually save the mock image
        pil_image = Image.fromarray(image)
        pil_image.save(file_path)

        return True

    def get_screen_size(self, monitor_index: int = 0) -> tuple[int, int]:
        """Get mock screen size.

        Args:
            monitor_index: Monitor index

        Returns:
            (width, height) tuple
        """
        return (self._width, self._height)

    def get_monitor_count(self) -> int:
        """Get number of mock monitors.

        Returns:
            Always returns 1 in mock mode
        """
        return 1

    def get_monitors(self) -> list[Monitor]:
        """Get list of available monitors.

        Returns:
            List with single mock monitor
        """
        monitor = Monitor(
            index=0,
            x=0,
            y=0,
            width=self._width,
            height=self._height,
            scale=1.0,
            is_primary=True,
            name="Mock Monitor"
        )
        return [monitor]

    def get_primary_monitor(self) -> Monitor:
        """Get primary monitor.

        Returns:
            Primary mock monitor
        """
        return Monitor(
            index=0,
            x=0,
            y=0,
            width=self._width,
            height=self._height,
            scale=1.0,
            is_primary=True,
            name="Mock Monitor"
        )

    def get_pixel_color(self, x: int, y: int, monitor: int | None = None) -> tuple[int, int, int]:
        """Get color of pixel at coordinates.

        Args:
            x: X coordinate
            y: Y coordinate
            monitor: Optional monitor index

        Returns:
            RGB color tuple (0, 0, 0) for mock mode
        """
        logger.debug(f"[MOCK] Get pixel color at ({x}, {y})")
        return (0, 0, 0)

    def close(self) -> None:
        """Close mock screen capture (no-op)."""
        logger.debug("MockScreen closed")

    # Mock-specific methods

    def set_mock_screen(self, screen_id: str, image: np.ndarray | Image.Image | str) -> None:
        """Set a mock screen state for testing.

        Args:
            screen_id: Identifier for this screen state
            image: Image as numpy array, PIL Image, or file path
        """
        if isinstance(image, str):
            # Load from file
            pil_image = Image.open(image)
            np_image = np.array(pil_image)
        elif isinstance(image, Image.Image):
            np_image = np.array(image)
        else:
            np_image = image

        self._mock_screens[screen_id] = np_image
        logger.debug(f"Mock screen state '{screen_id}' set ({np_image.shape})")

    def get_mock_screen(self, screen_id: str) -> np.ndarray | None:
        """Get a mock screen state.

        Args:
            screen_id: Identifier for the screen state

        Returns:
            Mock screen image or None if not found
        """
        return self._mock_screens.get(screen_id)

    def set_screen_size(self, width: int, height: int) -> None:
        """Set mock screen size.

        Args:
            width: Screen width
            height: Screen height
        """
        self._width = width
        self._height = height
        logger.debug(f"Mock screen size set to {width}x{height}")

    def reset(self) -> None:
        """Reset mock screen state (for test cleanup)."""
        self._capture_count = 0
        self._mock_screens.clear()
        logger.debug("MockScreen reset")

    def get_capture_count(self) -> int:
        """Get number of captures performed.

        Returns:
            Capture count
        """
        return self._capture_count
