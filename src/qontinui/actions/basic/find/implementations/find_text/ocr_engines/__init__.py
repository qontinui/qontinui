"""OCR engine implementations.

Strategy pattern implementations for different OCR backends.
"""

from .base_ocr_engine import BaseOCREngine, OCRResult
from .easyocr_engine import EasyOCREngine
from .paddleocr_engine import PaddleOCREngine
from .tesseract_engine import TesseractEngine

__all__ = [
    "BaseOCREngine",
    "OCRResult",
    "TesseractEngine",
    "EasyOCREngine",
    "PaddleOCREngine",
]
