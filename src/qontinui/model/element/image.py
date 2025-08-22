"""Image model - ported from Qontinui framework.

Physical representation of an image in the framework.
"""

from typing import Optional, Union
from dataclasses import dataclass, field
import numpy as np
from PIL import Image as PILImage
import cv2
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class Image:
    """Physical representation of an image in the GUI automation framework.
    
    Port of Image from Qontinui framework class.
    
    Image serves as the core container for visual data in Qontinui, providing a unified 
    interface for working with images across different formats and libraries. It acts as 
    the bridge between PIL Images, NumPy arrays (OpenCV), and other image representations.
    
    Key features:
    - Multi-format Support: Stores images as PIL Image internally with conversions to 
      NumPy arrays (BGR/HSV) and other formats
    - Database Persistence: Serializable to byte arrays for storage
    - Color Space Conversions: Built-in BGR and HSV representations for 
      advanced color-based matching
    - Flexible Construction: Can be created from files, PIL Images, NumPy arrays, 
      or Patterns
    
    Use cases in model-based automation:
    - Storing screenshots captured during automation execution
    - Holding pattern templates for visual matching
    - Providing image data for color analysis and profiling
    - Enabling image manipulation and processing operations
    
    The Image class abstracts away the complexity of working with multiple image 
    libraries, providing a consistent API that supports the framework's cross-platform 
    and technology-agnostic approach to GUI automation.
    """
    
    name: Optional[str] = None
    """Optional name for this image."""
    
    _pil_image: Optional[PILImage.Image] = field(default=None, repr=False)
    """Internal PIL Image storage."""
    
    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> 'Image':
        """Create Image from file.
        
        Args:
            filename: Path to image file
            
        Returns:
            Image instance
        """
        path = Path(filename)
        try:
            pil_image = PILImage.open(path)
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Extract name without extension
            name = path.stem
            
            return cls(name=name, _pil_image=pil_image)
        except Exception as e:
            logger.error(f"Failed to load image from {filename}: {e}")
            return cls(name=str(path))
    
    @classmethod
    def from_pil(cls, pil_image: PILImage.Image, name: Optional[str] = None) -> 'Image':
        """Create Image from PIL Image.
        
        Args:
            pil_image: PIL Image object
            name: Optional name
            
        Returns:
            Image instance
        """
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return cls(name=name, _pil_image=pil_image)
    
    @classmethod
    def from_numpy(cls, numpy_array: np.ndarray, name: Optional[str] = None) -> 'Image':
        """Create Image from NumPy array (OpenCV format).
        
        Args:
            numpy_array: NumPy array in BGR format
            name: Optional name
            
        Returns:
            Image instance
        """
        # Convert BGR to RGB
        rgb_array = cv2.cvtColor(numpy_array, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_array)
        return cls(name=name, _pil_image=pil_image)
    
    @classmethod
    def from_pattern(cls, pattern: 'Pattern') -> 'Image':
        """Create Image from Pattern.
        
        Args:
            pattern: Pattern object
            
        Returns:
            Image instance
        """
        return cls(name=pattern.name, _pil_image=pattern.get_image())
    
    @classmethod
    def get_empty_image(cls) -> 'Image':
        """Create an empty image.
        
        Returns:
            Empty Image instance
        """
        # Create a blank 1920x1080 image (default screen size)
        pil_image = PILImage.new('RGB', (1920, 1080), color='black')
        return cls(name="empty scene", _pil_image=pil_image)
    
    @property
    def pil_image(self) -> Optional[PILImage.Image]:
        """Get PIL Image.
        
        Returns:
            PIL Image or None
        """
        return self._pil_image
    
    def get_mat_bgr(self) -> Optional[np.ndarray]:
        """Get BGR representation as NumPy array.
        
        Returns:
            BGR NumPy array or None
        """
        if self._pil_image is None:
            logger.error(f"Cannot convert to BGR - PIL image is null for image: {self.name}")
            return None
        
        try:
            # Convert PIL to NumPy array
            rgb_array = np.array(self._pil_image)
            # Convert RGB to BGR
            bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            
            logger.debug(f"Successfully converted to BGR - dimensions: {bgr_array.shape} for image: {self.name}")
            return bgr_array
        except Exception as e:
            logger.error(f"BGR conversion failed for image {self.name}: {e}")
            return None
    
    def get_mat_hsv(self) -> Optional[np.ndarray]:
        """Get HSV representation as NumPy array.
        
        Returns:
            HSV NumPy array or None
        """
        if self._pil_image is None:
            logger.error(f"Cannot convert to HSV - PIL image is null for image: {self.name}")
            return None
        
        try:
            # Convert PIL to NumPy array
            rgb_array = np.array(self._pil_image)
            # Convert RGB to HSV
            hsv_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
            
            logger.debug(f"Successfully converted to HSV - dimensions: {hsv_array.shape} for image: {self.name}")
            return hsv_array
        except Exception as e:
            logger.error(f"HSV conversion failed for image {self.name}: {e}")
            return None
    
    def is_empty(self) -> bool:
        """Check if image is empty.
        
        Returns:
            True if no image data
        """
        return self._pil_image is None
    
    @property
    def width(self) -> int:
        """Get image width.
        
        Returns:
            Width in pixels or 0
        """
        if self._pil_image is None:
            logger.warning(f"PIL image is null for image: {self.name}")
            return 0
        return self._pil_image.width
    
    @property
    def w(self) -> int:
        """Alias for width (Brobot compatibility).
        
        Returns:
            Width in pixels or 0
        """
        return self.width
    
    @property
    def height(self) -> int:
        """Get image height.
        
        Returns:
            Height in pixels or 0
        """
        if self._pil_image is None:
            logger.warning(f"PIL image is null for image: {self.name}")
            return 0
        return self._pil_image.height
    
    @property
    def h(self) -> int:
        """Alias for height (Brobot compatibility).
        
        Returns:
            Height in pixels or 0
        """
        return self.height
    
    def save(self, filename: Union[str, Path]) -> bool:
        """Save image to file.
        
        Args:
            filename: Output file path
            
        Returns:
            True if successful
        """
        if self._pil_image is None:
            logger.error(f"Cannot save - PIL image is null for image: {self.name}")
            return False
        
        try:
            self._pil_image.save(filename)
            logger.debug(f"Saved image to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save image to {filename}: {e}")
            return False
    
    def to_bytes(self) -> Optional[bytes]:
        """Convert to bytes for storage.
        
        Returns:
            Image as bytes or None
        """
        if self._pil_image is None:
            return None
        
        try:
            from io import BytesIO
            buffer = BytesIO()
            self._pil_image.save(buffer, format='PNG')
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Failed to convert image to bytes: {e}")
            return None
    
    @classmethod
    def from_bytes(cls, data: bytes, name: Optional[str] = None) -> 'Image':
        """Create Image from bytes.
        
        Args:
            data: Image byte data
            name: Optional name
            
        Returns:
            Image instance
        """
        try:
            from io import BytesIO
            buffer = BytesIO(data)
            pil_image = PILImage.open(buffer)
            return cls(name=name, _pil_image=pil_image)
        except Exception as e:
            logger.error(f"Failed to create image from bytes: {e}")
            return cls(name=name)
    
    def __str__(self) -> str:
        """String representation."""
        width = self._pil_image.width if self._pil_image else "N/A"
        height = self._pil_image.height if self._pil_image else "N/A"
        return f"Image(name='{self.name}', width={width}, height={height})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()