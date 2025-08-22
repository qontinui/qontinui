"""Enhanced screen capture with multi-monitor support.

This replaces Brobot's screen package with a Python solution using mss for fast
screenshots and pyautogui for compatibility.
"""
import pyautogui
import mss
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import time

from ..logging import get_logger
from ..exceptions import ScreenCaptureException, InvalidImageException
from ..config import get_settings


logger = get_logger(__name__)


@dataclass
class Monitor:
    """Monitor information.
    
    Attributes:
        index: Monitor index (0-based)
        x: X coordinate of top-left corner
        y: Y coordinate of top-left corner  
        width: Monitor width in pixels
        height: Monitor height in pixels
        scale: DPI scaling factor
        is_primary: Whether this is the primary monitor
    """
    index: int
    x: int
    y: int
    width: int
    height: int
    scale: float = 1.0
    is_primary: bool = False
    
    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Get monitor bounds as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is within monitor bounds.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if point is within bounds
        """
        return (self.x <= x < self.x + self.width and
                self.y <= y < self.y + self.height)


class ScreenCapture:
    """Enhanced screen capture with multi-monitor support.
    
    Features:
        - Fast screenshot capture using mss
        - Multi-monitor support
        - Region capture
        - Template matching
        - Screenshot caching
    """
    
    def __init__(self):
        """Initialize screen capture."""
        self.settings = get_settings()
        self.sct = mss.mss()
        self._monitors = self._detect_monitors()
        self._cache: Dict[str, Image.Image] = {}
        self._cache_timeout = 1.0  # Cache timeout in seconds
        self._cache_timestamps: Dict[str, float] = {}
        
        logger.info(
            "screen_capture_initialized",
            monitor_count=len(self._monitors),
            multi_monitor=self.settings.multi_monitor
        )
        
    def _detect_monitors(self) -> List[Monitor]:
        """Detect all available monitors.
        
        Returns:
            List of Monitor objects
        """
        monitors = []
        
        # Skip index 0 as it's the combined virtual monitor
        for i, mon in enumerate(self.sct.monitors[1:], 1):
            monitor = Monitor(
                index=i - 1,  # 0-based index
                x=mon["left"],
                y=mon["top"],
                width=mon["width"],
                height=mon["height"],
                scale=self.settings.dpi_scaling,
                is_primary=(i == 1)  # First monitor is usually primary
            )
            monitors.append(monitor)
            
            logger.debug(
                "monitor_detected",
                index=monitor.index,
                bounds=monitor.bounds,
                is_primary=monitor.is_primary
            )
            
        return monitors
    
    @property
    def monitors(self) -> List[Monitor]:
        """Get list of available monitors."""
        return self._monitors
    
    @property
    def primary_monitor(self) -> Monitor:
        """Get primary monitor."""
        for monitor in self._monitors:
            if monitor.is_primary:
                return monitor
        return self._monitors[0] if self._monitors else None
    
    def get_monitor(self, index: Optional[int] = None) -> Monitor:
        """Get specific monitor by index.
        
        Args:
            index: Monitor index (0-based), None for primary
            
        Returns:
            Monitor object
            
        Raises:
            ScreenCaptureException: If monitor not found
        """
        if index is None:
            index = self.settings.default_monitor or 0
            
        if 0 <= index < len(self._monitors):
            return self._monitors[index]
        
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
            cache: Whether to use caching
            
        Returns:
            PIL Image of screenshot
            
        Raises:
            ScreenCaptureException: If capture fails
        """
        try:
            # Check cache
            cache_key = f"screen_{monitor}"
            if cache and self._is_cached(cache_key):
                return self._cache[cache_key]
            
            # Capture screenshot
            if monitor is None and not self.settings.multi_monitor:
                # Use default monitor
                monitor = self.settings.default_monitor or 0
                
            if monitor is not None:
                # Capture specific monitor
                mon = self.sct.monitors[monitor + 1]  # +1 for mss indexing
                sct_img = self.sct.grab(mon)
            else:
                # Capture all monitors
                sct_img = self.sct.grab(self.sct.monitors[0])
                
            # Convert to PIL Image
            image = Image.frombytes(
                'RGB',
                (sct_img.width, sct_img.height),
                sct_img.bgra,
                'raw',
                'BGRX'
            )
            
            # Apply screenshot delay
            if self.settings.screenshot_delay > 0:
                time.sleep(self.settings.screenshot_delay)
                
            # Update cache
            if cache:
                self._update_cache(cache_key, image)
                
            logger.debug(
                "screen_captured",
                monitor=monitor,
                size=(image.width, image.height)
            )
            
            return image
            
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
            cache: Whether to use caching
            
        Returns:
            PIL Image of region
            
        Raises:
            ScreenCaptureException: If capture fails
        """
        try:
            # Build region dict
            region = {
                "left": x,
                "top": y,
                "width": width,
                "height": height
            }
            
            # Adjust for monitor offset if specified
            if monitor is not None:
                mon = self.sct.monitors[monitor + 1]
                region["left"] += mon["left"]
                region["top"] += mon["top"]
                
            # Check cache
            cache_key = f"region_{x}_{y}_{width}_{height}_{monitor}"
            if cache and self._is_cached(cache_key):
                return self._cache[cache_key]
                
            # Capture region
            sct_img = self.sct.grab(region)
            
            # Convert to PIL Image
            image = Image.frombytes(
                'RGB',
                (sct_img.width, sct_img.height),
                sct_img.bgra,
                'raw',
                'BGRX'
            )
            
            # Update cache
            if cache:
                self._update_cache(cache_key, image)
                
            logger.debug(
                "region_captured",
                region=(x, y, width, height),
                monitor=monitor
            )
            
            return image
            
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
                image = Image.open(image)
                
            # Prepare search region
            search_region = None
            if region:
                search_region = region
            elif monitor is not None:
                mon = self._monitors[monitor]
                search_region = mon.bounds
                
            # Use pyautogui for template matching
            location = pyautogui.locateOnScreen(
                image,
                confidence=confidence,
                region=search_region,
                grayscale=grayscale
            )
            
            if location:
                logger.debug(
                    "image_found",
                    location=location,
                    confidence=confidence
                )
            else:
                logger.debug(
                    "image_not_found",
                    confidence=confidence,
                    region=search_region
                )
                
            return location
            
        except pyautogui.ImageNotFoundException:
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
                image = Image.open(image)
                
            # Prepare search region
            search_region = None
            if region:
                search_region = region
            elif monitor is not None:
                mon = self._monitors[monitor]
                search_region = mon.bounds
                
            # Find all matches
            locations = list(pyautogui.locateAllOnScreen(
                image,
                confidence=confidence,
                region=search_region,
                grayscale=grayscale
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
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Capture image
            if region:
                image = self.capture_region(*region, monitor=monitor)
            else:
                image = self.capture_screen(monitor=monitor)
                
            # Save image
            image.save(path)
            
            logger.info(
                "screenshot_saved",
                path=str(path),
                size=(image.width, image.height)
            )
            
            return path
            
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
            # Adjust coordinates for monitor
            if monitor is not None:
                mon = self._monitors[monitor]
                x += mon.x
                y += mon.y
                
            # Get pixel color using pyautogui
            return pyautogui.pixel(x, y)
            
        except Exception as e:
            logger.error(
                "get_pixel_color_failed",
                x=x, y=y,
                error=str(e)
            )
            return (0, 0, 0)
    
    def _is_cached(self, key: str) -> bool:
        """Check if cache entry is valid.
        
        Args:
            key: Cache key
            
        Returns:
            True if cached and valid
        """
        if key not in self._cache:
            return False
            
        timestamp = self._cache_timestamps.get(key, 0)
        age = time.time() - timestamp
        
        if age > self._cache_timeout:
            # Expired, remove from cache
            del self._cache[key]
            del self._cache_timestamps[key]
            return False
            
        return True
    
    def _update_cache(self, key: str, image: Image.Image) -> None:
        """Update cache entry.
        
        Args:
            key: Cache key
            image: Image to cache
        """
        self._cache[key] = image
        self._cache_timestamps[key] = time.time()
        
        # Limit cache size
        max_cache_size = 50
        if len(self._cache) > max_cache_size:
            # Remove oldest entries
            sorted_keys = sorted(
                self._cache_timestamps.keys(),
                key=lambda k: self._cache_timestamps[k]
            )
            for old_key in sorted_keys[:len(self._cache) - max_cache_size]:
                del self._cache[old_key]
                del self._cache_timestamps[old_key]
    
    def clear_cache(self) -> None:
        """Clear screenshot cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.debug("screenshot_cache_cleared")
    
    def close(self) -> None:
        """Close screen capture resources."""
        if hasattr(self, 'sct'):
            self.sct.close()
        self.clear_cache()
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