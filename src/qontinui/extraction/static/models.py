"""
Re-export static analysis models from centralized location.

This module provides convenient access to static analysis models for
analyzers in the static/ directory.
"""

# Re-export for backwards compatibility
# Import StaticConfig from the config module
from qontinui.extraction.config import StaticConfig
from qontinui.extraction.models.static import (
    APICallDefinition,  # New hint models for state discovery
)
from qontinui.extraction.models.static import (
    APICallType,
    ComponentCategory,
    ComponentDefinition,
    ComponentType,
    ConditionalPattern,
    ConditionalRender,
    EventHandler,
    RouteDefinition,
    RouteParam,
    RouteType,
    SearchParam,
    StateHint,
    StateImageHint,
    StateScope,
    StateSourceType,
    StateVariable,
    StaticAnalysisResult,
    TransitionHint,
    VisibilityState,
)

__all__ = [
    # Enums
    "ComponentCategory",
    "ComponentType",
    "StateSourceType",
    "StateScope",
    "ConditionalPattern",
    "RouteType",
    "APICallType",
    # Dataclasses
    "ComponentDefinition",
    "StateVariable",
    "ConditionalRender",
    "RouteParam",
    "SearchParam",
    "RouteDefinition",
    "EventHandler",
    "APICallDefinition",
    "StaticAnalysisResult",
    "VisibilityState",
    # Hint models (for guiding runtime state discovery)
    "StateHint",
    "StateImageHint",
    "TransitionHint",
    # Config
    "StaticConfig",
]
