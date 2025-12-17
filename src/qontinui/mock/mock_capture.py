"""MockCapture - Simulates screen capture using cached screenshots (Brobot pattern).

Provides mock screen capture for testing without actual screen access.
Uses pre-captured screenshots from a cache directory or generates simple
mock images on demand.

This enables:
- Fast testing (no real screen capture overhead)
- Headless CI/CD execution (no display needed)
- Deterministic testing (same screenshots every time)
- Screenshot-based testing (pre-record screenshots, replay in tests)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from ..hal.interfaces.screen_capture import Monitor

if TYPE_CHECKING:
    from ..model.element import Region

logger = logging.getLogger(__name__)


class MockCapture:
    """Mock screen capture implementation.

    Simulates screen capture operations by:
    1. Using pre-captured screenshots from cache directory (if configured)
    2. Generating simple mock images (if no cache)

    Example:
        # With screenshot cache
        capture = MockCapture()
        screenshot = capture.capture()  # Returns cached screenshot

        # No cache - generates mock
        screenshot = capture.capture()  # Returns generated mock image
    """

    def __init__(self, screenshot_dir: str | None = None) -> None:
        """Initialize MockCapture.

        Args:
            screenshot_dir: Optional directory containing cached screenshots
        """
        self.screenshot_dir = Path(screenshot_dir) if screenshot_dir else None
        self.screenshot_cache: dict[str, Image.Image] = {}

        # Default mock screen dimensions
        self.mock_width = 1920
        self.mock_height = 1080

        # Mock monitors
        self._monitors = [
            Monitor(
                index=0,
                x=0,
                y=0,
                width=self.mock_width,
                height=self.mock_height,
                scale=1.0,
                is_primary=True,
                name="Mock Primary Display",
            ),
        ]

        logger.debug(f"MockCapture initialized (screenshot_dir={screenshot_dir})")

    def capture(self, monitor: int | None = None) -> Image.Image:
        """Capture entire screen (mock).

        Args:
            monitor: Monitor index (ignored in mock mode)

        Returns:
            PIL Image (cached or generated)

        Example:
            capture = MockCapture()
            screenshot = capture.capture()
        """
        logger.debug(f"MockCapture.capture: monitor={monitor}")

        # Try to load from cache
        if self.screenshot_dir:
            cached = self._load_from_cache("fullscreen.png")
            if cached:
                return cached

        # Generate mock screenshot
        return self._generate_mock_screenshot()

    def capture_region(
        self,
        region: Region,
        monitor: int | None = None,
    ) -> Image.Image:
        """Capture specific region (mock).

        Args:
            region: Region to capture
            monitor: Monitor index (ignored)

        Returns:
            PIL Image of region

        Example:
            from qontinui.model.element.region import Region
            capture = MockCapture()
            region = Region(100, 200, 300, 400)
            screenshot = capture.capture_region(region)
        """
        logger.debug(f"MockCapture.capture_region: {region}")

        # Try to load full screenshot and crop
        if self.screenshot_dir:
            cached = self._load_from_cache("fullscreen.png")
            if cached:
                return cached.crop(
                    (region.x, region.y, region.x + region.w, region.y + region.h)
                )

        # Generate mock region
        return self._generate_mock_region(region.w, region.h)

    def get_monitors(self) -> list[Monitor]:
        """Get list of monitors (mock).

        Returns:
            List with single mock monitor

        Example:
            capture = MockCapture()
            monitors = capture.get_monitors()
        """
        logger.debug("MockCapture.get_monitors")
        return self._monitors

    def get_primary_monitor(self) -> Monitor:
        """Get primary monitor (mock).

        Returns:
            Mock primary monitor

        Example:
            capture = MockCapture()
            primary = capture.get_primary_monitor()
        """
        logger.debug("MockCapture.get_primary_monitor")
        return self._monitors[0]

    def get_screen_size(self) -> tuple[int, int]:
        """Get screen size (mock).

        Returns:
            Tuple of (width, height)

        Example:
            capture = MockCapture()
            width, height = capture.get_screen_size()
        """
        logger.debug("MockCapture.get_screen_size")
        return (self.mock_width, self.mock_height)

    def get_pixel_color(
        self,
        x: int,
        y: int,
        monitor: int | None = None,
    ) -> tuple[int, int, int]:
        """Get pixel color at coordinates (mock).

        Args:
            x: X coordinate
            y: Y coordinate
            monitor: Monitor index (ignored)

        Returns:
            RGB color tuple

        Example:
            capture = MockCapture()
            color = capture.get_pixel_color(100, 200)
        """
        logger.debug(f"MockCapture.get_pixel_color: ({x}, {y})")

        # Try to get from cached screenshot
        if self.screenshot_dir:
            cached = self._load_from_cache("fullscreen.png")
            if cached and 0 <= x < cached.width and 0 <= y < cached.height:
                return cached.getpixel((x, y))[:3]  # type: ignore[return-value,index]

        # Return mock color (light gray)
        return (200, 200, 200)

    def save_screenshot(
        self,
        filepath: str,
        monitor: int | None = None,
        region: Region | None = None,
    ) -> str:
        """Save screenshot to file (mock).

        Args:
            filepath: Path to save screenshot
            monitor: Monitor index (ignored)
            region: Optional region to capture

        Returns:
            Path where screenshot was saved

        Example:
            capture = MockCapture()
            path = capture.save_screenshot("test.png")
        """
        logger.debug(f"MockCapture.save_screenshot: {filepath}")

        if region:
            screenshot = self.capture_region(region, monitor)
        else:
            screenshot = self.capture(monitor)

        screenshot.save(filepath)
        logger.info(f"Mock screenshot saved to: {filepath}")
        return filepath

    # Private helper methods

    def _load_from_cache(self, filename: str) -> Image.Image | None:
        """Load screenshot from cache.

        Args:
            filename: Filename in cache directory

        Returns:
            PIL Image if found, None otherwise
        """
        if not self.screenshot_dir:
            return None

        # Check memory cache first
        if filename in self.screenshot_cache:
            logger.debug(f"Screenshot loaded from memory cache: {filename}")
            return self.screenshot_cache[filename]

        # Try to load from disk
        filepath = self.screenshot_dir / filename
        if filepath.exists():
            try:
                image = Image.open(filepath)
                self.screenshot_cache[filename] = image
                logger.debug(f"Screenshot loaded from disk cache: {filepath}")
                return image
            except Exception as e:
                logger.warning(f"Failed to load cached screenshot {filepath}: {e}")

        return None

    def _generate_mock_screenshot(self) -> Image.Image:
        """Generate a mock full screenshot.

        Returns:
            PIL Image with mock content
        """
        logger.debug("Generating mock fullscreen screenshot")

        # Create a simple gradient image
        image = Image.new(
            "RGB", (self.mock_width, self.mock_height), color=(240, 240, 240)
        )

        # Add some visual elements to make it look more realistic
        from PIL import ImageDraw

        draw = ImageDraw.Draw(image)

        # Draw a mock taskbar at bottom
        draw.rectangle(
            [(0, self.mock_height - 40), (self.mock_width, self.mock_height)],
            fill=(50, 50, 50),
        )

        # Draw a mock title bar at top
        draw.rectangle(
            [(0, 0), (self.mock_width, 30)],
            fill=(60, 60, 60),
        )

        # Draw some mock window rectangles
        draw.rectangle([(100, 100), (800, 600)], outline=(100, 100, 100), width=2)
        draw.rectangle([(900, 150), (1500, 700)], outline=(100, 100, 100), width=2)

        return image

    def _generate_mock_region(self, width: int, height: int) -> Image.Image:
        """Generate a mock region image.

        Args:
            width: Region width
            height: Region height

        Returns:
            PIL Image with mock content
        """
        logger.debug(f"Generating mock region: {width}x{height}")

        # Create a simple solid color image
        image = Image.new("RGB", (width, height), color=(220, 220, 220))

        # Add border
        from PIL import ImageDraw

        draw = ImageDraw.Draw(image)
        draw.rectangle(
            [(0, 0), (width - 1, height - 1)], outline=(150, 150, 150), width=2
        )

        return image
