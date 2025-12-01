"""Backend adapters for action execution following Brobot principles.

NO BACKWARD COMPATIBILITY: PyAutoGUI has been completely replaced with HAL.

This module uses the Facade pattern with composition, delegating to specialized
adapters for mouse, keyboard, and screen operations following Single Responsibility Principle.
"""

from typing import Any

from .adapter_impl.adapter_result import AdapterResult
from .adapter_impl.keyboard_adapter import KeyboardAdapter
from .adapter_impl.mouse_adapter import MouseAdapter
from .adapter_impl.screen_adapter import ScreenAdapter


class ActionAdapter:
    """Base class for pure action adapters.

    Following Brobot principles:
    - Each method is atomic and does exactly one thing
    - No composite actions at adapter level
    - Clear, predictable behavior

    Uses Facade pattern with composition to delegate to specialized adapters.
    """

    def __init__(
        self,
        mouse_adapter: MouseAdapter,
        keyboard_adapter: KeyboardAdapter,
        screen_adapter: ScreenAdapter,
    ) -> None:
        """Initialize action adapter with specialized sub-adapters.

        Args:
            mouse_adapter: Mouse-specific adapter
            keyboard_adapter: Keyboard-specific adapter
            screen_adapter: Screen-specific adapter
        """
        self._mouse = mouse_adapter
        self._keyboard = keyboard_adapter
        self._screen = screen_adapter

    # Mouse Actions (delegated to MouseAdapter)
    def mouse_down(
        self, x: int | None = None, y: int | None = None, button: str = "left"
    ) -> AdapterResult:
        """Press and hold mouse button."""
        return self._mouse.mouse_down(x, y, button)

    def mouse_up(
        self, x: int | None = None, y: int | None = None, button: str = "left"
    ) -> AdapterResult:
        """Release mouse button."""
        return self._mouse.mouse_up(x, y, button)

    def mouse_move(self, x: int, y: int, duration: float = 0.0) -> AdapterResult:
        """Move mouse to coordinates."""
        return self._mouse.mouse_move(x, y, duration)

    def mouse_click(self, x: int, y: int, button: str = "left") -> AdapterResult:
        """Single click at position."""
        return self._mouse.mouse_click(x, y, button)

    def mouse_scroll(
        self, clicks: int, x: int | None = None, y: int | None = None
    ) -> AdapterResult:
        """Scroll mouse wheel."""
        return self._mouse.mouse_scroll(clicks, x, y)

    def get_mouse_position(self) -> AdapterResult:
        """Get current mouse position."""
        return self._mouse.get_mouse_position()

    # Keyboard Actions (delegated to KeyboardAdapter)
    def key_down(self, key: str) -> AdapterResult:
        """Press and hold key."""
        return self._keyboard.key_down(key)

    def key_up(self, key: str) -> AdapterResult:
        """Release key."""
        return self._keyboard.key_up(key)

    def key_press(self, key: str) -> AdapterResult:
        """Press key (down + up)."""
        return self._keyboard.key_press(key)

    def type_character(self, char: str) -> AdapterResult:
        """Type single character."""
        return self._keyboard.type_character(char)

    # Screen Actions (delegated to ScreenAdapter)
    def capture_screen(
        self, region: tuple[int, int, int, int] | None = None
    ) -> AdapterResult:
        """Capture screenshot."""
        return self._screen.capture_screen(region)

    def get_screen_size(self) -> AdapterResult:
        """Get screen dimensions."""
        return self._screen.get_screen_size()


class HALAdapter(ActionAdapter):
    """HAL backend adapter with pure actions - REPLACES PyAutoGUIAdapter.

    Uses composition to delegate to specialized HAL adapters for each concern.
    """

    def __init__(self) -> None:
        """Initialize HAL adapter with specialized sub-adapters."""
        from ..hal import HALFactory
        from ..hal.interfaces import MouseButton as HALMouseButton
        from .adapter_impl.keyboard_adapter import HALKeyboardAdapter
        from .adapter_impl.mouse_adapter import HALMouseAdapter
        from .adapter_impl.screen_adapter import HALScreenAdapter

        # Get HAL components
        screen_capture = HALFactory.get_screen_capture()
        input_controller = HALFactory.get_input_controller()

        # Button mapping
        button_map = {
            "left": HALMouseButton.LEFT,
            "right": HALMouseButton.RIGHT,
            "middle": HALMouseButton.MIDDLE,
        }

        # Create specialized adapters
        mouse_adapter = HALMouseAdapter(input_controller, button_map)
        keyboard_adapter = HALKeyboardAdapter(input_controller)
        screen_adapter = HALScreenAdapter(screen_capture)

        # Initialize with composition
        super().__init__(mouse_adapter, keyboard_adapter, screen_adapter)


class SeleniumAdapter(ActionAdapter):
    """Selenium WebDriver backend adapter with pure actions.

    Uses composition to delegate to specialized Selenium adapters for each concern.
    """

    def __init__(self, driver: Any) -> None:
        """Initialize Selenium adapter with specialized sub-adapters.

        Args:
            driver: Selenium WebDriver instance
        """
        from .adapter_impl.keyboard_adapter import SeleniumKeyboardAdapter
        from .adapter_impl.mouse_adapter import SeleniumMouseAdapter
        from .adapter_impl.screen_adapter import SeleniumScreenAdapter

        try:
            from selenium.webdriver.common.action_chains import ActionChains

            action_chains = ActionChains(driver)
        except ImportError as e:
            raise ImportError("Selenium not installed") from e

        # Create specialized adapters
        mouse_adapter = SeleniumMouseAdapter(driver, action_chains)
        keyboard_adapter = SeleniumKeyboardAdapter(action_chains)
        screen_adapter = SeleniumScreenAdapter(driver)

        # Initialize with composition
        super().__init__(mouse_adapter, keyboard_adapter, screen_adapter)
