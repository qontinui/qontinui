"""Find text implementation - modular OCR and matching components.

Refactored architecture with separated concerns:
- OCR engines: Strategy pattern for different OCR backends
- Text matchers: Strategy pattern for text matching algorithms
- Image preprocessor: Centralized image preprocessing
- Orchestrator: Coordinates all components
"""

from .find_text_orchestrator import FindTextOrchestrator
from .image_preprocessor import ImagePreprocessor
from .ocr_engines import BaseOCREngine, EasyOCREngine, OCRResult, PaddleOCREngine, TesseractEngine
from .text_matchers import (
    BaseMatcher,
    ContainsMatcher,
    EndsWithMatcher,
    ExactMatcher,
    FuzzyMatcher,
    RegexMatcher,
    StartsWithMatcher,
)

__all__ = [
    # Orchestrator
    "FindTextOrchestrator",
    # OCR engines
    "BaseOCREngine",
    "OCRResult",
    "TesseractEngine",
    "EasyOCREngine",
    "PaddleOCREngine",
    # Text matchers
    "BaseMatcher",
    "ExactMatcher",
    "ContainsMatcher",
    "StartsWithMatcher",
    "EndsWithMatcher",
    "FuzzyMatcher",
    "RegexMatcher",
    # Preprocessor
    "ImagePreprocessor",
]
