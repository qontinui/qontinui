"""
Vision-based extraction for StateImage candidate discovery.

This module provides computer vision and machine learning techniques
for detecting UI elements in screenshots. StateImage is the primary
extractable element - categories like button, input, link are for
description only and don't have functional significance.

Detection Techniques:
    - Edge Detection: Canny edges + contour analysis (OpenCV)
    - SAM3 Segmentation: Segment Anything Model for precise boundaries
    - OCR Detection: Text detection and recognition (EasyOCR/Tesseract)

Usage:
    >>> from qontinui.extraction.vision import UnifiedVisionExtractor
    >>> extractor = UnifiedVisionExtractor()
    >>> result = await extractor.extract(screenshot_path, screenshot_id)
    >>> print(f"Found {len(result.candidates)} StateImage candidates")
"""

from .edge import EdgeDetector
from .fusion import ResultMerger
from .models import (
    BoundingBox,
    ContourResult,
    EdgeDetectionConfig,
    EdgeDetectionResult,
    ExtractedStateImageCandidate,
    FusionConfig,
    OCRConfig,
    OCRResult,
    SAM3Config,
    SAM3SegmentResult,
    ScreenshotInfo,
    TemplateConfig,
    VisionExtractionConfig,
    VisionExtractionResult,
)
from .ocr import OCRDetector
from .sam3 import SAM3Segmenter
from .unified_extractor import UnifiedVisionExtractor

# Keep backward compatibility with old VisionExtractor
from .vision_extractor import VisionExtractor

__all__ = [
    # Main extractor
    "UnifiedVisionExtractor",
    "VisionExtractor",  # Legacy
    # Detection modules
    "EdgeDetector",
    "SAM3Segmenter",
    "OCRDetector",
    "ResultMerger",
    # Models
    "BoundingBox",
    "ExtractedStateImageCandidate",
    "VisionExtractionResult",
    "EdgeDetectionResult",
    "SAM3SegmentResult",
    "OCRResult",
    "ContourResult",
    "ScreenshotInfo",
    # Config
    "VisionExtractionConfig",
    "EdgeDetectionConfig",
    "SAM3Config",
    "OCRConfig",
    "TemplateConfig",
    "FusionConfig",
]
