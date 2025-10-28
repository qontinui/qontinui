"""Find implementations - ported from Qontinui framework.

Various strategies for finding elements on screen.
"""

from .find_all import FindAll, FindAllBuilder
from .find_image import FindImage, ImageFinder

# Text finding via OCR is available but not yet integrated into the main Find API.
# See find_text/ directory for OCRResult and FindTextOrchestrator implementations.
# Integration tracked in: https://github.com/jspinak/qontinui/issues/[TBD]
# from .find_text import FindTextOrchestrator, OCRResult

__all__ = [
    # Image finding
    "FindImage",
    "ImageFinder",
    # Text finding (commented out until imports fixed)
    # "FindTextOrchestrator",
    # "OCRResult",
    # Exhaustive search
    "FindAll",
    "FindAllBuilder",
]
