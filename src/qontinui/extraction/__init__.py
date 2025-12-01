"""
Extraction framework for qontinui.

This module provides tools for extracting GUI elements, states, and transitions
from applications through static code analysis, runtime observation, and
correlation between the two approaches.

The framework consists of three main components:
1. Static Analysis: Extracts structure from source code
2. Runtime Extraction: Observes actual UI behavior
3. State Matching: Correlates static and runtime data

The ExtractionOrchestrator coordinates these components and manages
the complete extraction pipeline.
"""

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
from .models.correlated import (
    EvidenceType,
    MatchingEvidence,
    VerifiedTransition,
)
from .orchestrator import ExtractionOrchestrator
from .runtime import RuntimeExtractor
from .static import StaticAnalyzer

__all__ = [
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
