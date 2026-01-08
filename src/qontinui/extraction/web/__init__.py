"""
Web extraction using Playwright.

This module provides data models and utilities for extracting interactive GUI elements
from web applications using DOM-based analysis.

Key components:
- SafePlaywrightStateCollector: Safe web crawler for State Machine data collection
- SafetyConfig: Configuration for preventing dangerous clicks
- ClickableVerifier: Verification of extracted elements using pattern matching
"""

from qontinui.extraction.web.config import ExtractionConfig
from qontinui.extraction.web.models import (
    BoundingBox,
    ElementType,
    ExtractedElement,
    ExtractedState,
    ExtractedTransition,
    ExtractionResult,
    InteractiveElement,
    PageExtraction,
    StateType,
    TransitionType,
)

# Safety module exports
from qontinui.extraction.web.safety import (
    ActionRisk,
    ConfirmationDialogHandler,
    ElementRiskAssessment,
    ElementSafetyAnalyzer,
    SafetyConfig,
)

# Verification module exports
from qontinui.extraction.web.verification import (
    BatchVerifier,
    ClickableVerifier,
    ExtractedClickable,
    VerificationMetrics,
    VerificationResult,
)


def __getattr__(name: str):
    """Lazy import for Playwright-dependent classes."""
    if name == "InteractiveElementExtractor":
        from qontinui.extraction.web.interactive_element_extractor import (
            InteractiveElementExtractor,
        )

        return InteractiveElementExtractor
    if name == "SafePlaywrightStateCollector":
        from qontinui.extraction.web.playwright_collector import (
            SafePlaywrightStateCollector,
        )

        return SafePlaywrightStateCollector
    if name == "CollectionResult":
        from qontinui.extraction.web.playwright_collector import CollectionResult

        return CollectionResult
    if name == "collect_web_states":
        from qontinui.extraction.web.playwright_collector import collect_web_states

        return collect_web_states
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Configuration
    "ExtractionConfig",
    # Interactive element extraction (lazy import)
    "InteractiveElementExtractor",
    "InteractiveElement",
    # Core models
    "BoundingBox",
    "ElementType",
    "ExtractionResult",
    "ExtractedElement",
    "ExtractedState",
    "ExtractedTransition",
    "PageExtraction",
    "StateType",
    "TransitionType",
    # Safety
    "ActionRisk",
    "SafetyConfig",
    "ElementSafetyAnalyzer",
    "ElementRiskAssessment",
    "ConfirmationDialogHandler",
    # Verification
    "ExtractedClickable",
    "ClickableVerifier",
    "BatchVerifier",
    "VerificationResult",
    "VerificationMetrics",
    # Playwright collector (lazy import)
    "SafePlaywrightStateCollector",
    "CollectionResult",
    "collect_web_states",
]
