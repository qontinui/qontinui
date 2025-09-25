"""Backend adapters for action execution following Brobot principles.

NO BACKWARD COMPATIBILITY: PyAutoGUI has been completely replaced with HAL.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class AdapterResult:
    """Result from adapter action execution."""

    success: bool
    data: Any | None = None
    error: str | None = None


class ActionAdapter(ABC):
    """Abstract base class for pure action adapters.

    Following Brobot principles:
    - Each method is atomic and does exactly one thing
    - No composite actions at adapter level
    - Clear, predictable behavior
    """

    # Pure Mouse Actions

    @abstractmethod
    def mouse_down(
        self, x: int | None = None, y: int | None = None, button: str = "left"
    ) -> AdapterResult:
        """Press and hold mouse button."""
        pass

    @abstractmethod
    def mouse_up(
        self, x: int | None = None, y: int | None = None, button: str = "left"
    ) -> AdapterResult:
        """Release mouse button."""
        pass

    @abstractmethod
    def mouse_move(self, x: int, y: int, duration: float = 0.0) -> AdapterResult:
        """Move mouse to coordinates."""
        pass

    @abstractmethod
    def mouse_click(self, x: int, y: int, button: str = "left") -> AdapterResult:
        """Single click at position."""
        pass

    @abstractmethod
    def mouse_scroll(
        self, clicks: int, x: int | None = None, y: int | None = None
    ) -> AdapterResult:
        """Scroll mouse wheel."""
        pass

    # Pure Keyboard Actions

    @abstractmethod
    def key_down(self, key: str) -> AdapterResult:
        """Press and hold key."""
        pass

    @abstractmethod
    def key_up(self, key: str) -> AdapterResult:
        """Release key."""
        pass

    @abstractmethod
    def key_press(self, key: str) -> AdapterResult:
        """Press key (down + up)."""
        pass

    @abstractmethod
    def type_character(self, char: str) -> AdapterResult:
        """Type single character."""
        pass

    # Pure Screen Actions

    @abstractmethod
    def capture_screen(self, region: tuple[int, int, int, int] | None = None) -> AdapterResult:
        """Capture screenshot."""
        pass

    @abstractmethod
    def get_mouse_position(self) -> AdapterResult:
        """Get current mouse position."""
        pass

    @abstractmethod
    def get_screen_size(self) -> AdapterResult:
        """Get screen dimensions."""
        pass


class HALAdapter(ActionAdapter):
    """HAL backend adapter with pure actions - REPLACES PyAutoGUIAdapter."""

    def __init__(self):
        """Initialize HAL adapter."""
        from ..hal import HALFactory
        from ..hal.interfaces import MouseButton as HALMouseButton

        # Get HAL components
        self.screen_capture = HALFactory.get_screen_capture()
        self.input_controller = HALFactory.get_input_controller()
        self.pattern_matcher = HALFactory.get_pattern_matcher()

        # Button mapping
        self._button_map = {
            "left": HALMouseButton.LEFT,
            "right": HALMouseButton.RIGHT,
            "middle": HALMouseButton.MIDDLE,
        }

    # Mouse Actions

    def mouse_down(
        self, x: int | None = None, y: int | None = None, button: str = "left"
    ) -> AdapterResult:
        """Press and hold mouse button."""
        try:
            hal_button = self._button_map.get(button.lower(), self._button_map["left"])
            success = self.input_controller.mouse_down(x, y, hal_button)
            if success:
                return AdapterResult(success=True)
            return AdapterResult(success=False, error="Mouse down failed")
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def mouse_up(
        self, x: int | None = None, y: int | None = None, button: str = "left"
    ) -> AdapterResult:
        """Release mouse button."""
        try:
            hal_button = self._button_map.get(button.lower(), self._button_map["left"])
            success = self.input_controller.mouse_up(x, y, hal_button)
            if success:
                return AdapterResult(success=True)
            return AdapterResult(success=False, error="Mouse up failed")
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def mouse_move(self, x: int, y: int, duration: float = 0.0) -> AdapterResult:
        """Move mouse to coordinates."""
        try:
            success = self.input_controller.mouse_move(x, y, duration)
            if success:
                return AdapterResult(success=True, data=(x, y))
            return AdapterResult(success=False, error="Mouse move failed")
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def mouse_click(self, x: int, y: int, button: str = "left") -> AdapterResult:
        """Single click at position."""
        try:
            hal_button = self._button_map.get(button.lower(), self._button_map["left"])
            success = self.input_controller.mouse_click(x, y, hal_button)
            if success:
                return AdapterResult(success=True, data=(x, y))
            return AdapterResult(success=False, error="Mouse click failed")
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def mouse_scroll(
        self, clicks: int, x: int | None = None, y: int | None = None
    ) -> AdapterResult:
        """Scroll mouse wheel."""
        try:
            success = self.input_controller.mouse_scroll(clicks, x, y)
            if success:
                return AdapterResult(success=True, data=clicks)
            return AdapterResult(success=False, error="Mouse scroll failed")
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    # Keyboard Actions

    def key_down(self, key: str) -> AdapterResult:
        """Press and hold key."""
        try:
            success = self.input_controller.key_down(key)
            if success:
                return AdapterResult(success=True, data=key)
            return AdapterResult(success=False, error="Key down failed")
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def key_up(self, key: str) -> AdapterResult:
        """Release key."""
        try:
            success = self.input_controller.key_up(key)
            if success:
                return AdapterResult(success=True, data=key)
            return AdapterResult(success=False, error="Key up failed")
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def key_press(self, key: str) -> AdapterResult:
        """Press key (down + up)."""
        try:
            success = self.input_controller.key_press(key)
            if success:
                return AdapterResult(success=True, data=key)
            return AdapterResult(success=False, error="Key press failed")
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def type_character(self, char: str) -> AdapterResult:
        """Type single character."""
        try:
            if len(char) != 1:
                return AdapterResult(success=False, error="Must be single character")
            success = self.input_controller.type_text(char)
            if success:
                return AdapterResult(success=True, data=char)
            return AdapterResult(success=False, error="Type character failed")
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    # Screen Actions

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

    def get_mouse_position(self) -> AdapterResult:
        """Get current mouse position."""
        try:
            pos = self.input_controller.get_mouse_position()
            return AdapterResult(success=True, data=(pos.x, pos.y))
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


class SeleniumAdapter(ActionAdapter):
    """Selenium WebDriver backend adapter with pure actions."""

    def __init__(self, driver: Any):
        """Initialize Selenium adapter.

        Args:
            driver: Selenium WebDriver instance
        """
        self.driver = driver
        self._init_action_chains()

    def _init_action_chains(self):
        """Initialize Selenium ActionChains."""
        try:
            from selenium.webdriver.common.action_chains import ActionChains

            self.actions = ActionChains(self.driver)
        except ImportError as e:
            raise ImportError("Selenium not installed") from e

    # Mouse Actions

    def mouse_down(
        self, x: int | None = None, y: int | None = None, button: str = "left"
    ) -> AdapterResult:
        """Press and hold mouse button in browser context."""
        try:
            if x is not None and y is not None:
                # Move to position first
                self.actions.move_by_offset(x, y)

            if button == "left":
                self.actions.click_and_hold()
            elif button == "right":
                self.actions.context_click()

            self.actions.perform()
            return AdapterResult(success=True)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def mouse_up(
        self, x: int | None = None, y: int | None = None, button: str = "left"
    ) -> AdapterResult:
        """Release mouse button in browser context."""
        try:
            if x is not None and y is not None:
                self.actions.move_by_offset(x, y)

            self.actions.release()
            self.actions.perform()
            return AdapterResult(success=True)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def mouse_move(self, x: int, y: int, duration: float = 0.0) -> AdapterResult:
        """Move mouse in browser context."""
        try:
            # Selenium doesn't support duration, move instantly
            self.actions.move_by_offset(x, y)
            self.actions.perform()
            return AdapterResult(success=True, data=(x, y))
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def mouse_click(self, x: int, y: int, button: str = "left") -> AdapterResult:
        """Click in browser context."""
        try:
            self.actions.move_by_offset(x, y)

            if button == "left":
                self.actions.click()
            elif button == "right":
                self.actions.context_click()

            self.actions.perform()
            return AdapterResult(success=True, data=(x, y))
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def mouse_scroll(
        self, clicks: int, x: int | None = None, y: int | None = None
    ) -> AdapterResult:
        """Scroll in browser context."""
        try:
            # Execute JavaScript for scrolling
            scroll_amount = clicks * 100  # Approximate pixels per click
            self.driver.execute_script(f"window.scrollBy(0, {-scroll_amount})")
            return AdapterResult(success=True, data=clicks)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    # Keyboard Actions

    def key_down(self, key: str) -> AdapterResult:
        """Press key in browser context."""
        try:
            from selenium.webdriver.common.keys import Keys

            selenium_key = getattr(Keys, key.upper(), key)
            self.actions.key_down(selenium_key)
            self.actions.perform()
            return AdapterResult(success=True, data=key)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def key_up(self, key: str) -> AdapterResult:
        """Release key in browser context."""
        try:
            from selenium.webdriver.common.keys import Keys

            selenium_key = getattr(Keys, key.upper(), key)
            self.actions.key_up(selenium_key)
            self.actions.perform()
            return AdapterResult(success=True, data=key)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def key_press(self, key: str) -> AdapterResult:
        """Press key in browser context."""
        try:
            from selenium.webdriver.common.keys import Keys

            selenium_key = getattr(Keys, key.upper(), key)
            self.actions.send_keys(selenium_key)
            self.actions.perform()
            return AdapterResult(success=True, data=key)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def type_character(self, char: str) -> AdapterResult:
        """Type character in browser context."""
        try:
            if len(char) != 1:
                return AdapterResult(success=False, error="Must be single character")
            self.actions.send_keys(char)
            self.actions.perform()
            return AdapterResult(success=True, data=char)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    # Screen Actions

    def capture_screen(self, region: tuple[int, int, int, int] | None = None) -> AdapterResult:
        """Capture browser screenshot."""
        try:
            import io

            from PIL import Image

            screenshot = self.driver.get_screenshot_as_png()
            image = Image.open(io.BytesIO(screenshot))

            if region:
                x, y, width, height = region
                image = image.crop((x, y, x + width, y + height))

            return AdapterResult(success=True, data=image)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def get_mouse_position(self) -> AdapterResult:
        """Get mouse position (not supported in Selenium)."""
        return AdapterResult(
            success=False, error="Mouse position not available in Selenium context"
        )

    def get_screen_size(self) -> AdapterResult:
        """Get browser window size."""
        try:
            size = self.driver.get_window_size()
            return AdapterResult(success=True, data=(size["width"], size["height"]))
        except Exception as e:
            return AdapterResult(success=False, error=str(e))


# Alias for backward compatibility (will be removed)
# Since we don't need backward compatibility, we can remove this
# PyAutoGUIAdapter = HALAdapter  # REMOVED - no backward compatibility
