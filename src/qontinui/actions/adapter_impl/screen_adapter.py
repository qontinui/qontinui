"""Screen action adapters following Single Responsibility Principle.

Separates screen-specific functionality from other adapter concerns.
"""

from abc import ABC, abstractmethod

from .adapter_result import AdapterResult


class ScreenAdapter(ABC):
    """Abstract base for screen-specific actions."""

    @abstractmethod
    def capture_screen(self, region: tuple[int, int, int, int] | None = None) -> AdapterResult:
        """Capture screenshot."""
        pass

    @abstractmethod
    def get_screen_size(self) -> AdapterResult:
        """Get screen dimensions."""
        pass


class HALScreenAdapter(ScreenAdapter):
    """HAL implementation of screen actions."""

    def __init__(self, screen_capture) -> None:
        """Initialize HAL screen adapter.

        Args:
            screen_capture: HAL screen capture instance
        """
        self.screen_capture = screen_capture

    def capture_screen(self, region: tuple[int, int, int, int] | None = None) -> AdapterResult:
        """Capture screenshot."""
        try:
            if region:
                x, y, width, height = region
                screenshot = self.screen_capture.capture_region(x, y, width, height)
            else:
                screenshot = self.screen_capture.capture_screen()
            return AdapterResult(success=True, data=screenshot)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def get_screen_size(self) -> AdapterResult:
        """Get screen dimensions."""
        try:
            primary = self.screen_capture.get_primary_monitor()
            if primary:
                return AdapterResult(success=True, data=(primary.width, primary.height))
            return AdapterResult(success=False, error="No primary monitor found")
        except Exception as e:
            return AdapterResult(success=False, error=str(e))


class SeleniumScreenAdapter(ScreenAdapter):
    """Selenium implementation of screen actions."""

    def __init__(self, driver) -> None:
        """Initialize Selenium screen adapter.

        Args:
            driver: Selenium WebDriver instance
        """
        self.driver = driver

    def capture_screen(self, region: tuple[int, int, int, int] | None = None) -> AdapterResult:
        """Capture browser screenshot."""
        try:
            import io

            from PIL import Image

            screenshot = self.driver.get_screenshot_as_png()
            image = Image.open(io.BytesIO(screenshot))  # type: ignore[assignment]

            if region:
                x, y, width, height = region
                image = image.crop((x, y, x + width, y + height))  # type: ignore[assignment]

            return AdapterResult(success=True, data=image)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def get_screen_size(self) -> AdapterResult:
        """Get browser window size."""
        try:
            size = self.driver.get_window_size()
            return AdapterResult(success=True, data=(size["width"], size["height"]))
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
