"""Screen wrapper for mock/live automation switching.

Based on Brobot's wrapper pattern - provides stable API that routes
to mock or live implementation based on execution mode.
"""

import logging

import numpy as np

from ..hal.factory import HALFactory
from ..mock.mock_mode_manager import MockModeManager
from ..mock.mock_screen import MockScreen

logger = logging.getLogger(__name__)


class Screen:
    """Wrapper for screen operations that routes to mock or live implementation.

    This wrapper provides a stable API for screen operations while allowing
    the underlying implementation to switch between mock and live modes.

    In mock mode:
    - All operations complete instantly (no real screen capture)
    - Returns synthetic images for testing

    In live mode:
    - Uses HAL screen capture for real screen operations
    - Actual system screen capture

    Example usage:
        screenshot = Screen.capture()  # Capture screen (real or simulated)
        region = Screen.capture_region(0, 0, 100, 100)  # Capture region
    """

    _mock_screen = MockScreen()

    @classmethod
    def capture(cls, monitor_index: int = 0) -> np.ndarray:
        """Capture entire screen.

        In mock mode: Returns synthetic screen image instantly
        In live mode: Captures actual screen

        Args:
            monitor_index: Monitor index to capture

        Returns:
            Screen image as numpy array
        """
        if MockModeManager.is_mock_mode():
            result = cls._mock_screen.capture_screen(monitor_index)
            logger.debug(f"[MOCK] Screen captured (monitor {monitor_index})")
            return result
        else:
            capture = HALFactory.get_screen_capture()
            result = capture.capture_screen(monitor_index)
            logger.debug(f"[LIVE] Screen captured (monitor {monitor_index})")
            return result

    @classmethod
    def capture_region(
        cls, x: int, y: int, width: int, height: int, monitor_index: int = 0
    ) -> np.ndarray:
        """Capture screen region.

        Args:
            x: Region X coordinate
            y: Region Y coordinate
            width: Region width
            height: Region height
            monitor_index: Monitor index

        Returns:
            Region image as numpy array
        """
        if MockModeManager.is_mock_mode():
            result = cls._mock_screen.capture_region(x, y, width, height, monitor_index)
            logger.debug(f"[MOCK] Region captured ({x}, {y}, {width}x{height})")
            return result
        else:
            capture = HALFactory.get_screen_capture()
            result = capture.capture_region(x, y, width, height, monitor_index)
            logger.debug(f"[LIVE] Region captured ({x}, {y}, {width}x{height})")
            return result

    @classmethod
    def save(
        cls,
        file_path: str,
        monitor_index: int = 0,
        region: tuple[int, int, int, int] | None = None,
    ) -> bool:
        """Save screenshot to file.

        Args:
            file_path: Output file path
            monitor_index: Monitor index
            region: Optional region (x, y, width, height)

        Returns:
            True if successful
        """
        if MockModeManager.is_mock_mode():
            result = cls._mock_screen.save_screenshot(file_path, monitor_index, region)
            logger.debug(f"[MOCK] Screenshot saved to {file_path}")
            return result
        else:
            capture = HALFactory.get_screen_capture()
            result = capture.save_screenshot(file_path, monitor_index, region)
            logger.debug(f"[LIVE] Screenshot saved to {file_path}")
            return result

    @classmethod
    def size(cls, monitor_index: int = 0) -> tuple[int, int]:
        """Get screen size.

        Args:
            monitor_index: Monitor index

        Returns:
            (width, height) tuple
        """
        if MockModeManager.is_mock_mode():
            return cls._mock_screen.get_screen_size(monitor_index)
        else:
            capture = HALFactory.get_screen_capture()
            return capture.get_screen_size(monitor_index)

    @classmethod
    def monitor_count(cls) -> int:
        """Get number of monitors.

        Returns:
            Monitor count
        """
        if MockModeManager.is_mock_mode():
            return cls._mock_screen.get_monitor_count()
        else:
            capture = HALFactory.get_screen_capture()
            return capture.get_monitor_count()

    @classmethod
    def reset_mock(cls) -> None:
        """Reset mock screen state (for test cleanup).

        Only affects mock mode.
        """
        cls._mock_screen.reset()
        logger.debug("Mock screen reset")

    @classmethod
    def set_mock_screen(cls, screen_id: str, image: np.ndarray) -> None:
        """Set a mock screen state for testing.

        Only affects mock mode. Allows tests to set expected screen states.

        Args:
            screen_id: Identifier for this screen state
            image: Image as numpy array
        """
        cls._mock_screen.set_mock_screen(screen_id, image)
