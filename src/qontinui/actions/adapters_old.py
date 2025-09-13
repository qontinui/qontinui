"""Backend adapters for action execution following Brobot principles.

NO BACKWARD COMPATIBILITY: PyAutoGUI has been completely replaced with HAL.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any
from dataclasses import dataclass
import time


@dataclass
class AdapterResult:
    """Result from adapter action execution."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None


class ActionAdapter(ABC):
    """Abstract base class for pure action adapters.
    
    Following Brobot principles:
    - Each method is atomic and does exactly one thing
    - No composite actions at adapter level
    - Clear, predictable behavior
    """
    
    # Pure Mouse Actions
    
    @abstractmethod
    def mouse_down(self, x: Optional[int] = None, y: Optional[int] = None, 
                   button: str = 'left') -> AdapterResult:
        """Press and hold mouse button."""
        pass
    
    @abstractmethod
    def mouse_up(self, x: Optional[int] = None, y: Optional[int] = None,
                 button: str = 'left') -> AdapterResult:
        """Release mouse button."""
        pass
    
    @abstractmethod
    def mouse_move(self, x: int, y: int, duration: float = 0.0) -> AdapterResult:
        """Move mouse to coordinates."""
        pass
    
    @abstractmethod
    def mouse_click(self, x: int, y: int, button: str = 'left') -> AdapterResult:
        """Single click at position."""
        pass
    
    @abstractmethod
    def mouse_scroll(self, clicks: int, x: Optional[int] = None,
                    y: Optional[int] = None) -> AdapterResult:
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
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> AdapterResult:
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
            'left': HALMouseButton.LEFT,
            'right': HALMouseButton.RIGHT,
            'middle': HALMouseButton.MIDDLE
        }
    
    # Mouse Actions
    
    def mouse_down(self, x: Optional[int] = None, y: Optional[int] = None,
                   button: str = 'left') -> AdapterResult:
        """Press and hold mouse button."""
        try:
            hal_button = self._button_map.get(button.lower(), self._button_map['left'])
            success = self.input_controller.mouse_down(x, y, hal_button)
            if success:
                return AdapterResult(success=True)
            return AdapterResult(success=False, error="Mouse down failed")
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    def mouse_up(self, x: Optional[int] = None, y: Optional[int] = None,
                 button: str = 'left') -> AdapterResult:
        """Release mouse button."""
        try:
            if x is not None and y is not None:
                self.pg.mouseUp(x, y, button=button)
            else:
                self.pg.mouseUp(button=button)
            return AdapterResult(success=True)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    def mouse_move(self, x: int, y: int, duration: float = 0.0) -> AdapterResult:
        """Move mouse to coordinates."""
        try:
            self.pg.moveTo(x, y, duration=duration)
            return AdapterResult(success=True, data=(x, y))
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    def mouse_click(self, x: int, y: int, button: str = 'left') -> AdapterResult:
        """Single click at position."""
        try:
            self.pg.click(x, y, button=button)
            return AdapterResult(success=True, data=(x, y))
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    def mouse_scroll(self, clicks: int, x: Optional[int] = None,
                    y: Optional[int] = None) -> AdapterResult:
        """Scroll mouse wheel."""
        try:
            if x is not None and y is not None:
                self.pg.scroll(clicks, x, y)
            else:
                self.pg.scroll(clicks)
            return AdapterResult(success=True, data=clicks)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    # Keyboard Actions
    
    def key_down(self, key: str) -> AdapterResult:
        """Press and hold key."""
        try:
            self.pg.keyDown(key)
            return AdapterResult(success=True, data=key)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    def key_up(self, key: str) -> AdapterResult:
        """Release key."""
        try:
            self.pg.keyUp(key)
            return AdapterResult(success=True, data=key)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    def key_press(self, key: str) -> AdapterResult:
        """Press key (down + up)."""
        try:
            self.pg.press(key)
            return AdapterResult(success=True, data=key)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    def type_character(self, char: str) -> AdapterResult:
        """Type single character."""
        try:
            if len(char) != 1:
                return AdapterResult(success=False, error="Must be single character")
            self.pg.typewrite(char)
            return AdapterResult(success=True, data=char)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    # Screen Actions
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> AdapterResult:
        """Capture screenshot."""
        try:
            screenshot = self.pg.screenshot(region=region)
            return AdapterResult(success=True, data=screenshot)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    def get_mouse_position(self) -> AdapterResult:
        """Get current mouse position."""
        try:
            pos = self.pg.position()
            return AdapterResult(success=True, data=(pos.x, pos.y))
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    def get_screen_size(self) -> AdapterResult:
        """Get screen dimensions."""
        try:
            size = self.pg.size()
            return AdapterResult(success=True, data=(size.width, size.height))
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
        except ImportError:
            raise ImportError("Selenium not installed")
    
    # Mouse Actions
    
    def mouse_down(self, x: Optional[int] = None, y: Optional[int] = None,
                   button: str = 'left') -> AdapterResult:
        """Press and hold mouse button."""
        try:
            if x is not None and y is not None:
                # Move to position first
                self.actions.move_by_offset(x, y)
            
            if button == 'left':
                self.actions.click_and_hold()
            elif button == 'right':
                self.actions.context_click()
            
            self.actions.perform()
            return AdapterResult(success=True)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    def mouse_up(self, x: Optional[int] = None, y: Optional[int] = None,
                 button: str = 'left') -> AdapterResult:
        """Release mouse button."""
        try:
            if x is not None and y is not None:
                self.actions.move_by_offset(x, y)
            
            self.actions.release()
            self.actions.perform()
            return AdapterResult(success=True)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    def mouse_move(self, x: int, y: int, duration: float = 0.0) -> AdapterResult:
        """Move mouse to coordinates."""
        try:
            # Selenium doesn't support duration directly
            if duration > 0:
                time.sleep(duration)
            
            self.actions.move_by_offset(x, y)
            self.actions.perform()
            return AdapterResult(success=True, data=(x, y))
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    def mouse_click(self, x: int, y: int, button: str = 'left') -> AdapterResult:
        """Single click at position."""
        try:
            self.actions.move_by_offset(x, y)
            
            if button == 'left':
                self.actions.click()
            elif button == 'right':
                self.actions.context_click()
            
            self.actions.perform()
            return AdapterResult(success=True, data=(x, y))
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    def mouse_scroll(self, clicks: int, x: Optional[int] = None,
                    y: Optional[int] = None) -> AdapterResult:
        """Scroll mouse wheel."""
        try:
            # Selenium scrolls differently
            self.driver.execute_script(f"window.scrollBy(0, {clicks * 100})")
            return AdapterResult(success=True, data=clicks)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    # Keyboard Actions
    
    def key_down(self, key: str) -> AdapterResult:
        """Press and hold key."""
        try:
            from selenium.webdriver.common.keys import Keys
            key_value = getattr(Keys, key.upper(), key)
            self.actions.key_down(key_value)
            self.actions.perform()
            return AdapterResult(success=True, data=key)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    def key_up(self, key: str) -> AdapterResult:
        """Release key."""
        try:
            from selenium.webdriver.common.keys import Keys
            key_value = getattr(Keys, key.upper(), key)
            self.actions.key_up(key_value)
            self.actions.perform()
            return AdapterResult(success=True, data=key)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    def key_press(self, key: str) -> AdapterResult:
        """Press key (down + up)."""
        try:
            from selenium.webdriver.common.keys import Keys
            key_value = getattr(Keys, key.upper(), key)
            self.actions.send_keys(key_value)
            self.actions.perform()
            return AdapterResult(success=True, data=key)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    def type_character(self, char: str) -> AdapterResult:
        """Type single character."""
        try:
            if len(char) != 1:
                return AdapterResult(success=False, error="Must be single character")
            self.actions.send_keys(char)
            self.actions.perform()
            return AdapterResult(success=True, data=char)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    # Screen Actions
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> AdapterResult:
        """Capture screenshot."""
        try:
            screenshot = self.driver.get_screenshot_as_png()
            return AdapterResult(success=True, data=screenshot)
        except Exception as e:
            return AdapterResult(success=False, error=str(e))
    
    def get_mouse_position(self) -> AdapterResult:
        """Get current mouse position (not supported in Selenium)."""
        return AdapterResult(
            success=False, 
            error="Mouse position not available in Selenium"
        )
    
    def get_screen_size(self) -> AdapterResult:
        """Get screen dimensions."""
        try:
            size = self.driver.get_window_size()
            return AdapterResult(success=True, data=(size['width'], size['height']))
        except Exception as e:
            return AdapterResult(success=False, error=str(e))


class AdapterFactory:
    """Factory for creating action adapters."""
    
    @staticmethod
    def create_adapter(backend: str = "pyautogui", **kwargs) -> ActionAdapter:
        """Create an action adapter for the specified backend.
        
        Args:
            backend: Backend name ('pyautogui', 'selenium')
            **kwargs: Backend-specific configuration
            
        Returns:
            ActionAdapter instance
            
        Raises:
            ValueError: If backend is not supported
        """
        adapters = {
            "pyautogui": PyAutoGUIAdapter,
            "selenium": SeleniumAdapter,
        }
        
        if backend not in adapters:
            raise ValueError(f"Unsupported backend: {backend}. Choose from: {list(adapters.keys())}")
        
        adapter_class = adapters[backend]
        return adapter_class(**kwargs)