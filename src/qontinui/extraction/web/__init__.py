"""
Web extraction module using Playwright.

This module provides comprehensive tools for extracting interactive GUI elements
from web applications using DOM-based analysis, accessibility trees, and AI-powered
element selection. It enables robust web automation that survives DOM changes.

Architecture Overview
---------------------
The module is organized into several capability layers:

**Core Extraction**
    - `InteractiveElementExtractor`: Extract buttons, links, inputs from DOM
    - `ExtractionOptions`: Configure extraction behavior
    - `InteractiveElement`: Data model for extracted elements

**Multi-Frame Support**
    - `FrameManager`: Enumerate and track iframes
    - `FrameAwareElement`: Elements with frame context
    - `extract_across_frames()`: Extract from all frames

**Accessibility Integration**
    - `AccessibilityExtractor`: CDP-based a11y tree extraction
    - `A11yTree`, `A11yNode`: Tree data structures
    - `EnrichedElement`: Elements with a11y data merged

**AI-Powered Selection**
    - `NaturalLanguageSelector`: Find elements by description
    - `SelectorHealer`: Auto-repair broken selectors
    - `llm_clients`: Anthropic, OpenAI, LiteLLM adapters

**Multimodal Context**
    - `HybridExtractor`: Combined DOM + screenshot context
    - `HybridContext`: Container for LLM consumption
    - `StateTracker`: Track page state changes

**Dynamic Content**
    - `DOMStabilityWaiter`: Wait for DOM to settle
    - `LazyContentLoader`: Handle infinite scroll, load more
    - `ContentChangeDetector`: Detect significant changes

**Safety & Verification**
    - `SafetyConfig`: Prevent dangerous interactions
    - `ClickableVerifier`: Verify element clickability
    - `SafePlaywrightStateCollector`: Safe web crawling

Quick Start Examples
--------------------
Basic element extraction::

    from qontinui.extraction.web import InteractiveElementExtractor

    extractor = InteractiveElementExtractor()
    elements = await extractor.extract_interactive_elements(page, "ss_001")
    print(f"Found {len(elements)} interactive elements")

Full extraction with stability and frames::

    result = await extractor.extract_full(
        page,
        screenshot_id="ss_001",
        include_iframes=True,
        wait_for_stability=True,
    )

AI-powered element selection::

    from qontinui.extraction.web import NaturalLanguageSelector, AnthropicClient

    selector = NaturalLanguageSelector(AnthropicClient())
    result = await selector.find_element("the blue submit button", elements)
    print(f"Found: {result.element.text} (confidence: {result.confidence})")

Multimodal context for LLMs::

    from qontinui.extraction.web import extract_hybrid_context

    context = await extract_hybrid_context(page)
    message = context.to_llm_message(include_screenshot=True)

Selector healing::

    from qontinui.extraction.web import SelectorHealer

    healer = SelectorHealer()
    result = await healer.heal_selector("#broken", saved_element, page)

Module Index
------------
- `interactive_element_extractor`: Core DOM extraction
- `frame_manager`: Multi-frame support
- `accessibility_extractor`: A11y tree extraction
- `llm_clients`: LLM adapters (Anthropic, OpenAI, LiteLLM)
- `natural_language_selector`: AI element selection
- `selector_healer`: Automatic selector repair
- `hybrid_extractor`: Multimodal context
- `dom_stability`: Dynamic content handling
- `llm_formatter`: LLM-friendly formatting
- `deep_locator`: Cross-frame selectors
- `safety`: Safety configuration
- `verification`: Element verification
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
    if name == "CollectorConfig":
        from qontinui.extraction.web.playwright_collector import CollectorConfig

        return CollectorConfig
    if name == "collect_with_enhanced_extraction":
        from qontinui.extraction.web.playwright_collector import (
            collect_with_enhanced_extraction,
        )

        return collect_with_enhanced_extraction
    if name == "ExtractionOptions":
        from qontinui.extraction.web.interactive_element_extractor import (
            ExtractionOptions,
        )

        return ExtractionOptions

    # Frame manager (multi-frame extraction)
    if name == "FrameManager":
        from qontinui.extraction.web.frame_manager import FrameManager

        return FrameManager
    if name == "FrameAwareElement":
        from qontinui.extraction.web.frame_manager import FrameAwareElement

        return FrameAwareElement
    if name == "FrameInfo":
        from qontinui.extraction.web.frame_manager import FrameInfo

        return FrameInfo
    if name == "FrameExtractionResult":
        from qontinui.extraction.web.frame_manager import FrameExtractionResult

        return FrameExtractionResult
    if name == "extract_across_frames":
        from qontinui.extraction.web.frame_manager import extract_across_frames

        return extract_across_frames

    # LLM formatter
    if name == "LLMFormatter":
        from qontinui.extraction.web.llm_formatter import LLMFormatter

        return LLMFormatter
    if name == "format_for_llm":
        from qontinui.extraction.web.llm_formatter import format_for_llm

        return format_for_llm
    if name == "FormattedElementList":
        from qontinui.extraction.web.llm_formatter import FormattedElementList

        return FormattedElementList
    if name == "IndexedElement":
        from qontinui.extraction.web.llm_formatter import IndexedElement

        return IndexedElement

    # Accessibility extractor
    if name == "AccessibilityExtractor":
        from qontinui.extraction.web.accessibility_extractor import (
            AccessibilityExtractor,
        )

        return AccessibilityExtractor
    if name == "A11yTree":
        from qontinui.extraction.web.accessibility_extractor import A11yTree

        return A11yTree
    if name == "A11yNode":
        from qontinui.extraction.web.accessibility_extractor import A11yNode

        return A11yNode
    if name == "EnrichedElement":
        from qontinui.extraction.web.accessibility_extractor import EnrichedElement

        return EnrichedElement
    if name == "extract_accessibility_tree":
        from qontinui.extraction.web.accessibility_extractor import (
            extract_accessibility_tree,
        )

        return extract_accessibility_tree
    if name == "enrich_with_accessibility":
        from qontinui.extraction.web.accessibility_extractor import (
            enrich_with_accessibility,
        )

        return enrich_with_accessibility

    # Deep locator resolver
    if name == "DeepLocatorResolver":
        from qontinui.extraction.web.deep_locator import DeepLocatorResolver

        return DeepLocatorResolver
    if name == "resolve_deep_locator":
        from qontinui.extraction.web.deep_locator import resolve_deep_locator

        return resolve_deep_locator
    if name == "click_deep":
        from qontinui.extraction.web.deep_locator import click_deep

        return click_deep
    if name == "fill_deep":
        from qontinui.extraction.web.deep_locator import fill_deep

        return fill_deep

    # Natural language selector (Phase 2)
    if name == "NaturalLanguageSelector":
        from qontinui.extraction.web.natural_language_selector import (
            NaturalLanguageSelector,
        )

        return NaturalLanguageSelector
    if name == "SelectionResult":
        from qontinui.extraction.web.natural_language_selector import SelectionResult

        return SelectionResult
    if name == "FallbackSelector":
        from qontinui.extraction.web.natural_language_selector import FallbackSelector

        return FallbackSelector
    if name == "find_element_by_description":
        from qontinui.extraction.web.natural_language_selector import (
            find_element_by_description,
        )

        return find_element_by_description

    # Selector healer (Phase 2)
    if name == "SelectorHealer":
        from qontinui.extraction.web.selector_healer import SelectorHealer

        return SelectorHealer
    if name == "HealingResult":
        from qontinui.extraction.web.selector_healer import HealingResult

        return HealingResult
    if name == "heal_broken_selector":
        from qontinui.extraction.web.selector_healer import heal_broken_selector

        return heal_broken_selector

    # Hybrid extractor (Phase 2)
    if name == "HybridExtractor":
        from qontinui.extraction.web.hybrid_extractor import HybridExtractor

        return HybridExtractor
    if name == "HybridContext":
        from qontinui.extraction.web.hybrid_extractor import HybridContext

        return HybridContext
    if name == "StateTracker":
        from qontinui.extraction.web.hybrid_extractor import StateTracker

        return StateTracker
    if name == "extract_hybrid_context":
        from qontinui.extraction.web.hybrid_extractor import extract_hybrid_context

        return extract_hybrid_context
    if name == "build_llm_prompt":
        from qontinui.extraction.web.hybrid_extractor import build_llm_prompt

        return build_llm_prompt

    # DOM stability (Phase 3)
    if name == "DOMStabilityWaiter":
        from qontinui.extraction.web.dom_stability import DOMStabilityWaiter

        return DOMStabilityWaiter
    if name == "LazyContentLoader":
        from qontinui.extraction.web.dom_stability import LazyContentLoader

        return LazyContentLoader
    if name == "ContentChangeDetector":
        from qontinui.extraction.web.dom_stability import ContentChangeDetector

        return ContentChangeDetector
    if name == "DOMSnapshot":
        from qontinui.extraction.web.dom_stability import DOMSnapshot

        return DOMSnapshot
    if name == "StabilityResult":
        from qontinui.extraction.web.dom_stability import StabilityResult

        return StabilityResult
    if name == "wait_for_stable_extraction":
        from qontinui.extraction.web.dom_stability import wait_for_stable_extraction

        return wait_for_stable_extraction
    if name == "load_lazy_content":
        from qontinui.extraction.web.dom_stability import load_lazy_content

        return load_lazy_content

    # Healing history (Phase 3)
    if name == "HealingHistory":
        from qontinui.extraction.web.selector_healer import HealingHistory

        return HealingHistory
    if name == "HealingRecord":
        from qontinui.extraction.web.selector_healer import HealingRecord

        return HealingRecord

    # LLM clients (for natural language selection)
    if name == "AnthropicClient":
        from qontinui.extraction.web.llm_clients import AnthropicClient

        return AnthropicClient
    if name == "OpenAIClient":
        from qontinui.extraction.web.llm_clients import OpenAIClient

        return OpenAIClient
    if name == "LiteLLMClient":
        from qontinui.extraction.web.llm_clients import LiteLLMClient

        return LiteLLMClient
    if name == "MockLLMClient":
        from qontinui.extraction.web.llm_clients import MockLLMClient

        return MockLLMClient
    if name == "create_llm_client":
        from qontinui.extraction.web.llm_clients import create_llm_client

        return create_llm_client
    if name == "LLMConfig":
        from qontinui.extraction.web.llm_clients import LLMConfig

        return LLMConfig
    if name == "LLMConfigValidationError":
        from qontinui.extraction.web.llm_clients import LLMConfigValidationError

        return LLMConfigValidationError

    # Exception types (lazy import)
    if name == "WebExtractionError":
        from qontinui.extraction.web.exceptions import WebExtractionError

        return WebExtractionError
    if name == "ExtractionTimeoutError":
        from qontinui.extraction.web.exceptions import ExtractionTimeoutError

        return ExtractionTimeoutError
    if name == "CDPError":
        from qontinui.extraction.web.exceptions import CDPError

        return CDPError
    if name == "FrameExtractionError":
        from qontinui.extraction.web.exceptions import FrameExtractionError

        return FrameExtractionError
    if name == "ShadowDOMError":
        from qontinui.extraction.web.exceptions import ShadowDOMError

        return ShadowDOMError
    if name == "ElementExtractionError":
        from qontinui.extraction.web.exceptions import ElementExtractionError

        return ElementExtractionError
    if name == "ValidationError":
        from qontinui.extraction.web.exceptions import ValidationError

        return ValidationError

    # Retry utilities (lazy import)
    if name == "with_retry":
        from qontinui.extraction.web.exceptions import with_retry

        return with_retry
    if name == "with_timeout":
        from qontinui.extraction.web.exceptions import with_timeout

        return with_timeout

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
    "CollectorConfig",
    "collect_web_states",
    "collect_with_enhanced_extraction",
    # Extractor options
    "ExtractionOptions",
    # Frame manager - multi-frame extraction (lazy import)
    "FrameManager",
    "FrameAwareElement",
    "FrameInfo",
    "FrameExtractionResult",
    "extract_across_frames",
    # LLM formatter - LLM-friendly element format (lazy import)
    "LLMFormatter",
    "format_for_llm",
    "FormattedElementList",
    "IndexedElement",
    # Accessibility extractor - A11y tree integration (lazy import)
    "AccessibilityExtractor",
    "A11yTree",
    "A11yNode",
    "EnrichedElement",
    "extract_accessibility_tree",
    "enrich_with_accessibility",
    # Deep locator - cross-frame selector resolution (lazy import)
    "DeepLocatorResolver",
    "resolve_deep_locator",
    "click_deep",
    "fill_deep",
    # Natural language selector - AI-driven element selection (lazy import)
    "NaturalLanguageSelector",
    "SelectionResult",
    "FallbackSelector",
    "find_element_by_description",
    # Selector healer - automatic selector repair (lazy import)
    "SelectorHealer",
    "HealingResult",
    "heal_broken_selector",
    # Hybrid extractor - combined DOM + screenshot context (lazy import)
    "HybridExtractor",
    "HybridContext",
    "StateTracker",
    "extract_hybrid_context",
    "build_llm_prompt",
    # DOM stability - dynamic content handling (Phase 3, lazy import)
    "DOMStabilityWaiter",
    "LazyContentLoader",
    "ContentChangeDetector",
    "DOMSnapshot",
    "StabilityResult",
    "wait_for_stable_extraction",
    "load_lazy_content",
    # Healing history - learning from past repairs (Phase 3, lazy import)
    "HealingHistory",
    "HealingRecord",
    # LLM clients - for natural language selection (lazy import)
    "AnthropicClient",
    "OpenAIClient",
    "LiteLLMClient",
    "MockLLMClient",
    "create_llm_client",
    "LLMConfig",
    "LLMConfigValidationError",
    # Exception types (lazy import)
    "WebExtractionError",
    "ExtractionTimeoutError",
    "CDPError",
    "FrameExtractionError",
    "ShadowDOMError",
    "ElementExtractionError",
    "ValidationError",
    # Retry utilities (lazy import)
    "with_retry",
    "with_timeout",
]
