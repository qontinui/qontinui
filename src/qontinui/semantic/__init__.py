"""Semantic discovery system for GUI automation.

This package provides a modular system for semantic analysis of screenshots,
enabling AI-powered object detection and scene understanding alongside
traditional pattern matching.
"""

from .core.pixel_location import PixelLocation
from .core.scene_analyzer import SceneAnalyzer
from .core.scene_object_store import SceneObjectStore
from .core.scene_query_service import SceneQueryService
from .core.semantic_object import SemanticObject
from .core.semantic_scene import SemanticScene
from .processors.base import ProcessingHints, ProcessorConfig, SemanticProcessor
from .processors.manager import ProcessorManager

__all__ = [
    "PixelLocation",
    "SemanticObject",
    "SemanticScene",
    "SceneObjectStore",
    "SceneQueryService",
    "SceneAnalyzer",
    "SemanticProcessor",
    "ProcessorConfig",
    "ProcessingHints",
    "ProcessorManager",
]
