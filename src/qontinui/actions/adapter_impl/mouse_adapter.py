"""Mouse action adapters following Single Responsibility Principle.

Separates mouse-specific functionality from other adapter concerns.
"""

from abc import ABC, abstractmethod

from .adapter_result import AdapterResult


class MouseAdapter(ABC):
    """Abstract base for mouse-specific actions."""

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

    @abstractmethod
    def get_mouse_position(self) -> AdapterResult:
        """Get current mouse position."""
        pass


class HALMouseAdapter(MouseAdapter):
    """HAL implementation of mouse actions."""

    def __init__(self, input_controller, button_map: dict) -> None:
        """Initialize HAL mouse adapter.

        Args:
            input_controller: HAL input controller
            button_map: Button name to HAL button mapping
        """
        self.input_controller = input_controller
        self._button_map = button_map

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

    def get_mouse_position(self) -> AdapterResult:
        """Get current mouse position."""
        try:
            pos = self.input_controller.get_mouse_position()
            return AdapterResult(success=True, data=(pos.x, pos.y))
        except Exception as e:
            return AdapterResult(success=False, error=str(e))


class SeleniumMouseAdapter(MouseAdapter):
    """Selenium implementation of mouse actions."""

    def __init__(self, driver, action_chains) -> None:
        """Initialize Selenium mouse adapter.

        Args:
            driver: Selenium WebDriver instance
            action_chains: Selenium ActionChains instance
        """
        self.driver = driver
        self.actions = action_chains

    def mouse_down(
        self, x: int | None = None, y: int | None = None, button: str = "left"
    ) -> AdapterResult:
        """Press and hold mouse button in browser context."""
        try:
            if x is not None and y is not None:
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
            scroll_amount = clicks * 100
            self.driver.execute_script(f"window.scrollBy(0, {-scroll_amount})")
            return AdapterResult(success=True, data=clicks)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))

    def get_mouse_position(self) -> AdapterResult:
        """Get mouse position (not supported in Selenium)."""
        return AdapterResult(
            success=False, error="Mouse position not available in Selenium context"
        )
