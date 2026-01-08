"""
Extraction framework for qontinui.

This module provides tools for extracting GUI elements, states, and transitions
from applications through static code analysis, runtime observation, and
correlation between the two approaches.

The framework supports multiple extraction backends:
1. DOM Extraction: Playwright-based extraction for web applications
2. Vision Extraction: CV/ML-based extraction for any GUI
3. Accessibility Extraction: OS API-based extraction for native apps
4. Static Analysis: Extracts structure from source code
5. State Matching: Correlates static and runtime data

The ExtractionOrchestrator coordinates these components and manages
the complete extraction pipeline, with automatic backend selection.

Usage:
    >>> from qontinui.extraction import ExtractionOrchestrator, ExtractorConfig
    >>> from qontinui.extraction.abstract_extractor import ExtractionContext
    >>>
    >>> orchestrator = ExtractionOrchestrator()
    >>> config = ExtractorConfig()
    >>> context = ExtractionContext(url="http://localhost:3000")
    >>> result = await orchestrator.extract(context, config)
"""

# Unified extraction architecture (new)
from .abstract_extractor import (
    AbstractExtractor,
    ExtractionContext,
    ExtractionError,
    ScreenshotError,
)
from .abstract_extractor import ExtractedElement as UnifiedExtractedElement
from .abstract_extractor import ExtractedState as UnifiedExtractedState
from .abstract_extractor import ExtractedTransition as UnifiedExtractedTransition
from .abstract_extractor import ExtractionResult as UnifiedExtractionResult
from .extractor_config import (
    AccessibilityConfig,
    ConfidenceThreshold,
    DOMConfig,
    ElementFilter,
    ExtractionBackend,
    ExtractorConfig,
    HybridConfig,
    HybridStrategy,
    SizeThreshold,
    VisionConfig,
)

# Existing extraction components
from .matching import StateMatcher
from .models.base import (
    BoundingBox,
    ConfigError,
    CorrelatedState,
    ExtractionConfig,
    ExtractionMode,
    ExtractionResult,
    ExtractionTarget,
    FrameworkType,
    InferredTransition,
    RuntimeExtractionResult,
    Screenshot,
    StaticAnalysisResult,
    Viewport,
)
from .models.composite import (
    ApplicationStateStructure,
    CompositeStateStructure,
    StateStructure,
)
from .models.correlated import EvidenceType, MatchingEvidence, VerifiedTransition
from .orchestrator import ExtractionOrchestrator
from .runtime import RuntimeExtractor
from .static import StaticAnalyzer

__all__ = [
    # === New unified extraction architecture ===
    # Abstract base
    "AbstractExtractor",
    "ExtractionContext",
    "ExtractionError",
    "ScreenshotError",
    # Unified models (prefixed to avoid conflict with existing models)
    "UnifiedExtractedElement",
    "UnifiedExtractedState",
    "UnifiedExtractedTransition",
    "UnifiedExtractionResult",
    # Configuration
    "ExtractorConfig",
    "ExtractionBackend",
    "ElementFilter",
    "HybridStrategy",
    "SizeThreshold",
    "ConfidenceThreshold",
    "VisionConfig",
    "DOMConfig",
    "AccessibilityConfig",
    "HybridConfig",
    # === Existing extraction components ===
    # Core state structure (unified model)
    "StateStructure",
    # Backward compatibility aliases
    "ApplicationStateStructure",
    "CompositeStateStructure",
    # Core types
    "BoundingBox",
    "ConfigError",
    "CorrelatedState",
    "EvidenceType",
    "ExtractionConfig",
    "ExtractionMode",
    "ExtractionOrchestrator",
    "ExtractionResult",
    "ExtractionTarget",
    "FrameworkType",
    "InferredTransition",
    "MatchingEvidence",
    "RuntimeExtractor",
    "RuntimeExtractionResult",
    "Screenshot",
    "StateMatcher",
    "StaticAnalysisResult",
    "StaticAnalyzer",
    "VerifiedTransition",
    "Viewport",
]
