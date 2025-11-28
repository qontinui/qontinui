"""
Web extraction using Playwright.

This module provides data models and utilities for extracting GUI elements,
states, and transitions from web applications.
"""

from qontinui.extraction.web.config import ExtractionConfig
from qontinui.extraction.web.extractor import WebExtractor
from qontinui.extraction.web.models import (
    BoundingBox,
    ElementType,
    ExtractedElement,
    ExtractedState,
    ExtractedTransition,
    ExtractionResult,
    PageExtraction,
    StateType,
    TransitionType,
)

__all__ = [
    "BoundingBox",
    "ElementType",
    "ExtractionConfig",
    "ExtractionResult",
    "ExtractedElement",
    "ExtractedState",
    "ExtractedTransition",
    "PageExtraction",
    "StateType",
    "TransitionType",
    "WebExtractor",
]
