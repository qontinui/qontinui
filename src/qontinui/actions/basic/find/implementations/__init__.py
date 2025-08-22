"""Find implementations - ported from Qontinui framework.

Various strategies for finding elements on screen.
"""

from .find_image import FindImage, ImageFinder
from .find_text import FindText, OCRResult
from .find_all import FindAll, FindAllBuilder

__all__ = [
    # Image finding
    'FindImage',
    'ImageFinder',
    
    # Text finding
    'FindText',
    'OCRResult',
    
    # Exhaustive search
    'FindAll',
    'FindAllBuilder',
]