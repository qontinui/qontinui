"""
Qontinui RAG (Retrieval-Augmented Generation) module.

Provides vector database storage and retrieval for GUI elements with multimodal embeddings.
"""

from .embeddings import (
    CLIPEmbedder,
    DINOv2Embedder,
    HybridImageEmbedder,
    TextDescriptionGenerator,
    TextEmbedder,
    colors_to_semantic,
)
from .export import (
    ConfigExportPipeline,
    ConfigExportResult,
    ConfigMetadata,
    ReembeddingRecommendation,
    check_model_compatibility,
    load_config_metadata,
    prompt_reembedding_if_needed,
)
from .filters import PredictedFilters, SearchQuery, build_filter_query
from .models import (
    BoundingBox,
    ElementType,
    EmbeddedElement,
    ExportResult,
    GUIElementChunk,
    SearchResult,
)
from .runtime import (
    FindOptions,
    FoundElement,
    OCRConfig,
    OCRFilter,
    OCRMatchMode,
    ProjectDefaults,
    RuntimeElementFinder,
    RuntimeEmbedder,
    ScreenSegment,
    ScreenSegmenter,
    SearchSession,
)
from .segment_vectorizer import (
    MatchingStrategy,
    RAGMatch,
    SegmentVector,
    SegmentVectorizer,
)
from .vector_db import QdrantLocalDB, RAGIndex

__all__ = [
    # Models
    "BoundingBox",
    "ElementType",
    "GUIElementChunk",
    "EmbeddedElement",
    "SearchResult",
    "ExportResult",
    # Embeddings - Text
    "TextEmbedder",
    "TextDescriptionGenerator",
    "colors_to_semantic",
    # Embeddings - Image
    "CLIPEmbedder",
    "DINOv2Embedder",
    "HybridImageEmbedder",
    # Export
    "ConfigExportPipeline",
    "ConfigExportResult",
    "ConfigMetadata",
    "ReembeddingRecommendation",
    "check_model_compatibility",
    "load_config_metadata",
    "prompt_reembedding_if_needed",
    # Filters
    "PredictedFilters",
    "SearchQuery",
    "build_filter_query",
    # Vector DB
    "QdrantLocalDB",
    "RAGIndex",
    # Runtime
    "FindOptions",
    "FoundElement",
    "OCRConfig",
    "OCRFilter",
    "OCRMatchMode",
    "ProjectDefaults",
    "RuntimeEmbedder",
    "RuntimeElementFinder",
    "ScreenSegment",
    "ScreenSegmenter",
    "SearchSession",
    # Segment Vectorizer
    "SegmentVectorizer",
    "SegmentVector",
    "RAGMatch",
    "MatchingStrategy",
]
