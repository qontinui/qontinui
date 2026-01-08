"""Detection engines for vision verification.

Provides specialized detection engines that integrate with the
discovered GUI environment for improved accuracy:

- OCR engine with typography hints
- Template matching with learned patterns
- State detection using visual signatures
"""

from qontinui.vision.verification.detection.ocr import (
    OCREngine,
    OCRResult,
    get_ocr_engine,
)
from qontinui.vision.verification.detection.template import (
    TemplateEngine,
    TemplateMatch,
    get_template_engine,
)

__all__ = [
    "OCREngine",
    "OCRResult",
    "get_ocr_engine",
    "TemplateEngine",
    "TemplateMatch",
    "get_template_engine",
]
