"""
Hybrid extraction module for combining static analysis with runtime extraction.

This module provides a framework for extracting States, StateImages, and Transitions
by combining:
1. Static code analysis (component structure, event handlers, routing)
2. Runtime extraction (screenshots, bounding boxes, actual UI state)

The hybrid approach provides precise bounding boxes for StateImages because:
- Static analysis tells us WHAT elements exist and their semantic meaning
- Runtime extraction tells us WHERE they are rendered on screen
- Combining both gives us rich, accurate State/StateImage/Transition data
"""

from .base import (
    HybridExtractionConfig,
    HybridExtractionResult,
    HybridExtractor,
    State,
    StateImage,
    StateTransition,
    TechStackExtractor,
)
from .registry import TechStackRegistry

__all__ = [
    "HybridExtractor",
    "TechStackExtractor",
    "TechStackRegistry",
    "HybridExtractionConfig",
    "State",
    "StateImage",
    "StateTransition",
    "HybridExtractionResult",
]
