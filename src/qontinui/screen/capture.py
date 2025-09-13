"""Enhanced screen capture using HAL (Hardware Abstraction Layer).

This module has been refactored to use the HAL instead of PyAutoGUI,
providing better performance and flexibility.
"""

from typing import Optional, Tuple, List, Union
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

from ..hal import HALFactory
from ..hal.interfaces.screen_capture import Monitor
from ..logging import get_logger
from ..exceptions import ScreenCaptureException
from ..config import get_settings


logger = get_logger(__name__)


class ScreenCapture:
    """Enhanced screen capture using HAL.
    
    Features:
        - Fast screenshot capture using mss
        - Multi-monitor support
        - Region capture
        - Template matching via OpenCV
        - Screenshot caching
    """
    
    def __init__(self):
        """Initialize screen capture using HAL."""
        self.settings = get_settings()
        
        # Get HAL components
        self.hal_capture = HALFactory.get_screen_capture()
        self.hal_matcher = HALFactory.get_pattern_matcher()
        
        logger.info(
            "screen_capture_initialized",
            backend="HAL",
            multi_monitor=self.settings.multi_monitor
        )
    
    @property
    def monitors(self) -> List[Monitor]:
        """Get list of available monitors."""
        return self.hal_capture.get_monitors()
    
    @property
    def primary_monitor(self) -> Monitor:
        """Get primary monitor."""
        return self.hal_capture.get_primary_monitor()
    
    def get_monitor(self, index: Optional[int] = None) -> Monitor:
        """Get specific monitor by index.
        
        Args:
            index: Monitor index (0-based), None for primary
            
        Returns:
            Monitor object
            
        Raises:
            ScreenCaptureException: If monitor not found
        """
        monitors = self.hal_capture.get_monitors()
        
        if index is None:
            return self.hal_capture.get_primary_monitor()
            
        if 0 <= index < len(monitors):
            return monitors[index]
        
        raise ScreenCaptureException(
            f"Monitor {index} not found",
            monitor=index
        )
    
    def capture_screen(
        self, 
        monitor: Optional[int] = None,
        cache: bool = True
    ) -> Image.Image:
        """Capture entire screen or specific monitor.
        
        Args:
            monitor: Monitor index, None for all monitors
            cache: Whether to use caching (handled by HAL)
            
        Returns:
            PIL Image of screenshot
            
        Raises:
            ScreenCaptureException: If capture fails
        """
        try:
            return self.hal_capture.capture_screen(monitor)
        except Exception as e:
            raise ScreenCaptureException(
                f"Failed to capture screen: {e}",
                monitor=monitor
            )
    
    def capture_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        monitor: Optional[int] = None,
        cache: bool = False
    ) -> Image.Image:
        """Capture specific region.
        
        Args:
            x: X coordinate of top-left corner
            y: Y coordinate of top-left corner  
            width: Region width
            height: Region height
            monitor: Optional monitor index
            cache: Whether to use caching (handled by HAL)
            
        Returns:
            PIL Image of region
            
        Raises:
            ScreenCaptureException: If capture fails
        """
        try:
            return self.hal_capture.capture_region(x, y, width, height, monitor)
        except Exception as e:
            raise ScreenCaptureException(
                f"Failed to capture region: {e}",
                monitor=monitor
            )
    
    def find_on_screen(
        self,
        image: Union[Image.Image, str, Path],
        confidence: float = 0.9,
        monitor: Optional[int] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
        grayscale: bool = False
    ) -> Optional[Tuple[int, int, int, int]]:
        """Find image on screen using template matching.
        
        Args:
            image: Image to find (PIL Image or path)
            confidence: Match confidence threshold
            monitor: Optional monitor to search
            region: Optional region to search (x, y, width, height)
            grayscale: Convert to grayscale for matching
            
        Returns:
            Location as (x, y, width, height) or None if not found
        """
        try:
            # Load image if path provided
            if isinstance(image, (str, Path)):
                needle = Image.open(image)
            else:
                needle = image
            
            # Get screenshot of search area
            if region:
                x, y, width, height = region
                haystack = self.capture_region(x, y, width, height, monitor)
                offset_x, offset_y = x, y
            elif monitor is not None:
                haystack = self.capture_screen(monitor)
                mon = self.get_monitor(monitor)
                offset_x, offset_y = mon.x, mon.y
            else:
                haystack = self.capture_screen()
                offset_x, offset_y = 0, 0
            
            # Use HAL pattern matcher
            match = self.hal_matcher.find_pattern(
                haystack, needle, confidence, grayscale
            )
            
            if match:
                # Adjust coordinates for offset
                return (
                    match.x + offset_x,
                    match.y + offset_y,
                    match.width,
                    match.height
                )
            
            logger.debug(
                "image_not_found",
                confidence=confidence,
                region=region
            )
            return None
            
        except Exception as e:
            logger.error(
                "find_on_screen_failed",
                error=str(e)
            )
            return None
    
    def find_all_on_screen(
        self,
        image: Union[Image.Image, str, Path],
        confidence: float = 0.9,
        monitor: Optional[int] = None,
        region: Optional[Tuple[int, int, int, int]] = None,
        grayscale: bool = False
    ) -> List[Tuple[int, int, int, int]]:
        """Find all occurrences of image on screen.
        
        Args:
            image: Image to find (PIL Image or path)
            confidence: Match confidence threshold
            monitor: Optional monitor to search
            region: Optional region to search
            grayscale: Convert to grayscale for matching
            
        Returns:
            List of locations as (x, y, width, height)
        """
        try:
            # Load image if path provided
            if isinstance(image, (str, Path)):
                needle = Image.open(image)
            else:
                needle = image
            
            # Get screenshot of search area
            if region:
                x, y, width, height = region
                haystack = self.capture_region(x, y, width, height, monitor)
                offset_x, offset_y = x, y
            elif monitor is not None:
                haystack = self.capture_screen(monitor)
                mon = self.get_monitor(monitor)
                offset_x, offset_y = mon.x, mon.y
            else:
                haystack = self.capture_screen()
                offset_x, offset_y = 0, 0
            
            # Use HAL pattern matcher
            matches = self.hal_matcher.find_all_patterns(
                haystack, needle, confidence, grayscale
            )
            
            # Adjust coordinates for offset
            locations = []
            for match in matches:
                locations.append((
                    match.x + offset_x,
                    match.y + offset_y,
                    match.width,
                    match.height
                ))
            
            logger.debug(
                "images_found",
                count=len(locations),
                confidence=confidence
            )
            
            return locations
            
        except Exception as e:
            logger.error(
                "find_all_on_screen_failed",
                error=str(e)
            )
            return []
    
    def save_screenshot(
        self,
        path: Union[str, Path],
        monitor: Optional[int] = None,
        region: Optional[Tuple[int, int, int, int]] = None
    ) -> Path:
        """Save screenshot to file.
        
        Args:
            path: File path to save to
            monitor: Optional monitor to capture
            region: Optional region to capture
            
        Returns:
            Path where screenshot was saved
            
        Raises:
            ScreenCaptureException: If save fails
        """
        try:
            path_str = str(path)
            saved_path = self.hal_capture.save_screenshot(path_str, monitor, region)
            return Path(saved_path)
        except Exception as e:
            raise ScreenCaptureException(
                f"Failed to save screenshot: {e}",
                monitor=monitor
            )
    
    def get_pixel_color(
        self,
        x: int,
        y: int,
        monitor: Optional[int] = None
    ) -> Tuple[int, int, int]:
        """Get color of pixel at coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            monitor: Optional monitor index
            
        Returns:
            RGB color tuple
        """
        try:
            return self.hal_capture.get_pixel_color(x, y, monitor)
        except Exception as e:
            logger.error(
                "get_pixel_color_failed",
                x=x, y=y,
                error=str(e)
            )
            return (0, 0, 0)
    
    def clear_cache(self) -> None:
        """Clear screenshot cache."""
        # Cache is managed by HAL implementation
        if hasattr(self.hal_capture, 'clear_cache'):
            self.hal_capture.clear_cache()
        logger.debug("screenshot_cache_cleared")
    
    def close(self) -> None:
        """Close screen capture resources."""
        if hasattr(self.hal_capture, 'close'):
            self.hal_capture.close()
        logger.debug("screen_capture_closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Global instance for convenience
_screen_capture: Optional[ScreenCapture] = None


def get_screen_capture() -> ScreenCapture:
    """Get global screen capture instance.
    
    Returns:
        ScreenCapture instance
    """
    global _screen_capture
    if _screen_capture is None:
        _screen_capture = ScreenCapture()
    return _screen_capture