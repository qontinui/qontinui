"""Pure atomic actions following Brobot principles."""

from typing import Optional, Tuple, Any
from dataclasses import dataclass
import time

# Handle pyautogui import for headless environments
try:
    import pyautogui
except Exception:
    # Create mock for headless environments
    import sys
    from unittest.mock import MagicMock
    pyautogui = MagicMock()
    sys.modules['pyautogui'] = pyautogui


@dataclass
class ActionResult:
    """Result of an action execution."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class PureActions:
    """Pure, atomic actions that do one thing only.
    
    Following Brobot principles:
    - Each action is atomic and does exactly one thing
    - No composite actions (like drag = mouseDown + move + mouseUp)
    - Actions return results for chaining
    - No retry logic in base actions
    """
    
    def __init__(self):
        """Initialize pure actions."""
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
    
    # Mouse Actions (Atomic)
    
    def mouse_down(self, x: Optional[int] = None, y: Optional[int] = None, 
                   button: str = 'left') -> ActionResult:
        """Press and hold mouse button.
        
        Args:
            x: X coordinate (None = current position)
            y: Y coordinate (None = current position)
            button: 'left', 'right', or 'middle'
            
        Returns:
            ActionResult with success status
        """
        try:
            if x is not None and y is not None:
                pyautogui.mouseDown(x, y, button=button)
            else:
                pyautogui.mouseDown(button=button)
            return ActionResult(success=True)
        except Exception as e:
            return ActionResult(success=False, error=str(e))
    
    def mouse_up(self, x: Optional[int] = None, y: Optional[int] = None,
                 button: str = 'left') -> ActionResult:
        """Release mouse button.
        
        Args:
            x: X coordinate (None = current position)
            y: Y coordinate (None = current position) 
            button: 'left', 'right', or 'middle'
            
        Returns:
            ActionResult with success status
        """
        try:
            if x is not None and y is not None:
                pyautogui.mouseUp(x, y, button=button)
            else:
                pyautogui.mouseUp(button=button)
            return ActionResult(success=True)
        except Exception as e:
            return ActionResult(success=False, error=str(e))
    
    def mouse_move(self, x: int, y: int, duration: float = 0.0) -> ActionResult:
        """Move mouse to coordinates.
        
        Args:
            x: Target X coordinate
            y: Target Y coordinate
            duration: Duration of movement in seconds
            
        Returns:
            ActionResult with success status
        """
        try:
            pyautogui.moveTo(x, y, duration=duration)
            return ActionResult(success=True, data=(x, y))
        except Exception as e:
            return ActionResult(success=False, error=str(e))
    
    def mouse_click(self, x: int, y: int, button: str = 'left') -> ActionResult:
        """Single atomic click at position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            button: 'left', 'right', or 'middle'
            
        Returns:
            ActionResult with success status
        """
        try:
            pyautogui.click(x, y, button=button)
            return ActionResult(success=True, data=(x, y))
        except Exception as e:
            return ActionResult(success=False, error=str(e))
    
    def mouse_scroll(self, clicks: int, x: Optional[int] = None,
                    y: Optional[int] = None) -> ActionResult:
        """Scroll mouse wheel.
        
        Args:
            clicks: Number of scroll clicks (positive=up, negative=down)
            x: Optional X coordinate
            y: Optional Y coordinate
            
        Returns:
            ActionResult with success status
        """
        try:
            if x is not None and y is not None:
                pyautogui.scroll(clicks, x, y)
            else:
                pyautogui.scroll(clicks)
            return ActionResult(success=True, data=clicks)
        except Exception as e:
            return ActionResult(success=False, error=str(e))
    
    # Keyboard Actions (Atomic)
    
    def key_down(self, key: str) -> ActionResult:
        """Press and hold a key.
        
        Args:
            key: Key to press down
            
        Returns:
            ActionResult with success status
        """
        try:
            pyautogui.keyDown(key)
            return ActionResult(success=True, data=key)
        except Exception as e:
            return ActionResult(success=False, error=str(e))
    
    def key_up(self, key: str) -> ActionResult:
        """Release a key.
        
        Args:
            key: Key to release
            
        Returns:
            ActionResult with success status
        """
        try:
            pyautogui.keyUp(key)
            return ActionResult(success=True, data=key)
        except Exception as e:
            return ActionResult(success=False, error=str(e))
    
    def key_press(self, key: str) -> ActionResult:
        """Press a single key (down + up).
        
        Args:
            key: Key to press
            
        Returns:
            ActionResult with success status
        """
        try:
            pyautogui.press(key)
            return ActionResult(success=True, data=key)
        except Exception as e:
            return ActionResult(success=False, error=str(e))
    
    def type_character(self, char: str) -> ActionResult:
        """Type a single character.
        
        Args:
            char: Single character to type
            
        Returns:
            ActionResult with success status
        """
        try:
            if len(char) != 1:
                return ActionResult(success=False, error="Must be single character")
            pyautogui.typewrite(char)
            return ActionResult(success=True, data=char)
        except Exception as e:
            return ActionResult(success=False, error=str(e))
    
    # Screen Actions (Atomic)
    
    def capture_screen(self, region: Optional[Tuple[int, int, int, int]] = None) -> ActionResult:
        """Capture screenshot.
        
        Args:
            region: Optional (x, y, width, height) tuple
            
        Returns:
            ActionResult with screenshot data
        """
        try:
            screenshot = pyautogui.screenshot(region=region)
            return ActionResult(success=True, data=screenshot)
        except Exception as e:
            return ActionResult(success=False, error=str(e))
    
    def get_mouse_position(self) -> ActionResult:
        """Get current mouse position.
        
        Returns:
            ActionResult with (x, y) position
        """
        try:
            position = pyautogui.position()
            return ActionResult(success=True, data=(position.x, position.y))
        except Exception as e:
            return ActionResult(success=False, error=str(e))
    
    def get_screen_size(self) -> ActionResult:
        """Get screen dimensions.
        
        Returns:
            ActionResult with (width, height)
        """
        try:
            size = pyautogui.size()
            return ActionResult(success=True, data=(size.width, size.height))
        except Exception as e:
            return ActionResult(success=False, error=str(e))
    
    # Timing Actions (Atomic)
    
    def wait(self, seconds: float) -> ActionResult:
        """Wait for specified duration.
        
        Args:
            seconds: Seconds to wait
            
        Returns:
            ActionResult with success status
        """
        try:
            time.sleep(seconds)
            return ActionResult(success=True, data=seconds)
        except Exception as e:
            return ActionResult(success=False, error=str(e))
    
    def pause(self, milliseconds: int) -> ActionResult:
        """Pause for milliseconds.
        
        Args:
            milliseconds: Milliseconds to pause
            
        Returns:
            ActionResult with success status
        """
        try:
            time.sleep(milliseconds / 1000.0)
            return ActionResult(success=True, data=milliseconds)
        except Exception as e:
            return ActionResult(success=False, error=str(e))