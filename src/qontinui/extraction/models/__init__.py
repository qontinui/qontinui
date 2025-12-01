"""
Data models for the extraction orchestrator.

These models define the contracts for extraction components across different
platforms and frameworks.
"""

# Base models and ABCs
from .base import (
    BoundingBox,
    ConfigError,
    ExtractionConfig,
    ExtractionMode,
    ExtractionTarget,
    FrameworkType,
    OutputFormat,
    RuntimeExtractor,
    Screenshot,
    StateMatcher,
    StaticAnalyzer,
    Viewport,
)

# Composite models for multi-application environments
from .composite import (
    ApplicationStateStructure,
    CompositeStateStructure,
)

# Correlated/output models
from .correlated import (
    CorrelatedState,
    EvidenceType,
    ExtractionResult,
    InferredTransition,
    MatchingEvidence,
    VerificationDiscrepancy,
    VerifiedTransition,
)

# Runtime extraction models
from .runtime import (
    DetectedRegion,
    ElementType,
    ExtractedElement,
    InteractionAction,
    ObservedTransition,
    RuntimeExtractionResult,
    RuntimeStateCapture,
    StateType,
    TransitionType,
)

# Static analysis models
from .static import (
    APICallDefinition,
    APICallType,
    ComponentDefinition,
    ComponentType,
    ConditionalPattern,
    ConditionalRender,
    EventHandler,
    RouteDefinition,
    RouteParam,
    RouteType,
    SearchParam,
    StateScope,
    StateSourceType,
    StateVariable,
    StaticAnalysisResult,
)

__all__ = [
    # Base
    "BoundingBox",
    "ConfigError",
    "ExtractionConfig",
    "ExtractionMode",
    "ExtractionTarget",
    "FrameworkType",
    "OutputFormat",
    "RuntimeExtractor",
    "Screenshot",
    "StateMatcher",
    "StaticAnalyzer",
    "Viewport",
    # Static
    "APICallDefinition",
    "APICallType",
    "ComponentDefinition",
    "ComponentType",
    "ConditionalPattern",
    "ConditionalRender",
    "EventHandler",
    "RouteDefinition",
    "RouteParam",
    "RouteType",
    "SearchParam",
    "StateScope",
    "StateSourceType",
    "StateVariable",
    "StaticAnalysisResult",
    # Runtime
    "DetectedRegion",
    "ElementType",
    "ExtractedElement",
    "InteractionAction",
    "ObservedTransition",
    "RuntimeExtractionResult",
    "RuntimeStateCapture",
    "StateType",
    "TransitionType",
    # Correlated
    "CorrelatedState",
    "EvidenceType",
    "ExtractionResult",
    "InferredTransition",
    "MatchingEvidence",
    "VerificationDiscrepancy",
    "VerifiedTransition",
    # Composite
    "ApplicationStateStructure",
    "CompositeStateStructure",
]
