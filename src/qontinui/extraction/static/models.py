"""
Re-export static analysis models from centralized location.

This module provides convenient access to static analysis models for
analyzers in the static/ directory.
"""

# Re-export for backwards compatibility
# Import StaticConfig from the config module
from qontinui.extraction.config import StaticConfig
from qontinui.extraction.models.static import (
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
    # Enums
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
    # Config
    "StaticConfig",
]
