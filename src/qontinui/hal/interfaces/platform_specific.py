"""Platform-specific interface definition."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class Window:
    """Represents a window/application."""
    handle: Any
    title: str
    class_name: Optional[str] = None
    process_id: Optional[int] = None
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    is_visible: bool = True
    is_minimized: bool = False
    is_maximized: bool = False
    
    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Get window bounds as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)


@dataclass
class UIElement:
    """Represents a UI element (button, textbox, etc.)."""
    handle: Any
    type: str
    name: Optional[str] = None
    text: Optional[str] = None
    class_name: Optional[str] = None
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    is_enabled: bool = True
    is_visible: bool = True
    is_focused: bool = False
    
    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Get element bounds as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of element."""
        return (self.x + self.width // 2, self.y + self.height // 2)


class IPlatformSpecific(ABC):
    """Interface for platform-specific operations."""
    
    # Window management
    
    @abstractmethod
    def get_all_windows(self) -> List[Window]:
        """Get list of all windows.
        
        Returns:
            List of Window objects
        """
        pass
    
    @abstractmethod
    def get_window_by_title(self, title: str, 
                           partial: bool = False) -> Optional[Window]:
        """Find window by title.
        
        Args:
            title: Window title to search for
            partial: Allow partial title match
            
        Returns:
            Window if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_window_by_process(self, process_name: str) -> List[Window]:
        """Get windows belonging to a process.
        
        Args:
            process_name: Process name (e.g., 'chrome.exe')
            
        Returns:
            List of Window objects
        """
        pass
    
    @abstractmethod
    def get_active_window(self) -> Optional[Window]:
        """Get currently active/focused window.
        
        Returns:
            Active Window or None
        """
        pass
    
    @abstractmethod
    def set_window_focus(self, window: Window) -> bool:
        """Set focus to window.
        
        Args:
            window: Window to focus
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def minimize_window(self, window: Window) -> bool:
        """Minimize window.
        
        Args:
            window: Window to minimize
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def maximize_window(self, window: Window) -> bool:
        """Maximize window.
        
        Args:
            window: Window to maximize
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def restore_window(self, window: Window) -> bool:
        """Restore window to normal size.
        
        Args:
            window: Window to restore
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def move_window(self, window: Window, x: int, y: int) -> bool:
        """Move window to position.
        
        Args:
            window: Window to move
            x: Target X coordinate
            y: Target Y coordinate
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def resize_window(self, window: Window, 
                     width: int, height: int) -> bool:
        """Resize window.
        
        Args:
            window: Window to resize
            width: New width
            height: New height
            
        Returns:
            True if successful
        """
        pass
    
    # UI Automation
    
    @abstractmethod
    def get_ui_elements(self, window: Window) -> List[UIElement]:
        """Get all UI elements in window.
        
        Args:
            window: Window to analyze
            
        Returns:
            List of UIElement objects
        """
        pass
    
    @abstractmethod
    def find_ui_element(self, window: Window,
                       name: Optional[str] = None,
                       type: Optional[str] = None,
                       text: Optional[str] = None) -> Optional[UIElement]:
        """Find UI element by properties.
        
        Args:
            window: Window to search in
            name: Element name/ID
            type: Element type (button, textbox, etc.)
            text: Element text content
            
        Returns:
            UIElement if found, None otherwise
        """
        pass
    
    @abstractmethod
    def click_ui_element(self, element: UIElement) -> bool:
        """Click UI element using platform API.
        
        Args:
            element: Element to click
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def set_ui_text(self, element: UIElement, text: str) -> bool:
        """Set text in UI element.
        
        Args:
            element: Text input element
            text: Text to set
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def get_ui_text(self, element: UIElement) -> Optional[str]:
        """Get text from UI element.
        
        Args:
            element: Element to read from
            
        Returns:
            Text content or None
        """
        pass
    
    # System information
    
    @abstractmethod
    def get_platform_name(self) -> str:
        """Get platform name.
        
        Returns:
            Platform name (e.g., 'Windows', 'macOS', 'Linux')
        """
        pass
    
    @abstractmethod
    def get_platform_version(self) -> str:
        """Get platform version.
        
        Returns:
            Version string (e.g., '10.0.19043', '12.6', 'Ubuntu 22.04')
        """
        pass
    
    @abstractmethod
    def get_screen_resolution(self) -> Tuple[int, int]:
        """Get primary screen resolution.
        
        Returns:
            Resolution as (width, height)
        """
        pass
    
    @abstractmethod
    def get_dpi_scaling(self) -> float:
        """Get DPI scaling factor.
        
        Returns:
            DPI scale (e.g., 1.0, 1.25, 1.5, 2.0)
        """
        pass
    
    @abstractmethod
    def is_dark_mode(self) -> bool:
        """Check if system is in dark mode.
        
        Returns:
            True if dark mode is enabled
        """
        pass