"""Runtime element finder for Qontinui RAG system.

This module provides runtime element discovery by combining indexed knowledge
with real-time screen segmentation and matching. It integrates SAM3 segmentation
with CLIP embeddings for accurate element location.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..logging import get_logger
from .embeddings import CLIPEmbedder, DINOv2Embedder
from .embeddings.text import TextEmbedder
from .models import BoundingBox, SearchResult
from .segment_vectorizer import MatchingStrategy, RAGMatch, SegmentVector, SegmentVectorizer
from .vector_db import QdrantLocalDB, RAGIndex

logger = get_logger(__name__)


# ============================================================================
# OCR Configuration Types
# ============================================================================


class OCRMatchMode(str, Enum):
    """Mode for OCR text matching."""

    EXACT = "exact"  # Text must match exactly
    CONTAINS = "contains"  # Text must contain the filter
    REGEX = "regex"  # Text must match regex pattern


@dataclass
class OCRFilter:
    """Filter for OCR text matching.

    Used to filter elements by their text content.

    Attributes:
        text: Text to match against OCR results
        match_mode: How to match the text
        similarity: Minimum similarity for fuzzy matching (0.0-1.0)
    """

    text: str | None = None
    match_mode: OCRMatchMode = OCRMatchMode.CONTAINS
    similarity: float = 0.8


@dataclass
class OCRConfig:
    """Configuration for OCR-based matching.

    Attributes:
        enabled: Whether OCR is enabled
        weight: Weight of OCR score in combined score (0.0-1.0)
        as_filter: If True, OCR must match to be included in results
        filter_threshold: Minimum OCR similarity to pass filter
    """

    enabled: bool = True
    weight: float = 0.3
    as_filter: bool = True
    filter_threshold: float = 0.6


@dataclass
class FindOptions:
    """Options for element finding that can override StateImage settings.

    These options have the highest priority in the similarity cascade:
    Project default (lowest) < StateImage (optional) < FindOptions (highest)

    Attributes:
        similarity_threshold: Override similarity threshold (None = use cascade)
        matching_strategy: Override matching strategy (None = use cascade)
        ocr_filter: OCR text filter for the search
        ocr_config: OCR configuration override
        max_results: Maximum number of results to return
        include_metadata: Include additional metadata in results
    """

    similarity_threshold: float | None = None
    matching_strategy: MatchingStrategy | None = None
    ocr_filter: OCRFilter | None = None
    ocr_config: OCRConfig | None = None
    max_results: int = 5
    include_metadata: bool = True


@dataclass
class ProjectDefaults:
    """Project-level default settings for element finding.

    These are the lowest priority in the similarity cascade.

    Attributes:
        similarity_threshold: Default similarity threshold
        matching_strategy: Default matching strategy
        ocr_config: Default OCR configuration
    """

    similarity_threshold: float = 0.7
    matching_strategy: MatchingStrategy = MatchingStrategy.AVERAGE
    ocr_config: OCRConfig = field(default_factory=lambda: OCRConfig(enabled=False))


@dataclass
class FoundElement:
    """Result from runtime element finding.

    Contains detailed scoring information for the match,
    including visual, text, and OCR components.

    Attributes:
        element_id: Unique identifier for the element
        score: Overall match score (0.0-1.0)
        bounding_box: Location of element on screen
        visual_similarity: Visual (image) similarity score
        text_similarity: Text embedding similarity (if available)
        ocr_similarity: OCR text match ratio (if enabled)
        ocr_text: Extracted OCR text from the matched segment
        metadata: Additional metadata about the match
    """

    element_id: str
    score: float
    bounding_box: BoundingBox
    visual_similarity: float
    text_similarity: float | None = None
    ocr_similarity: float | None = None
    ocr_text: str | None = None
    metadata: dict[str, Any] | None = None

    # Legacy alias for backward compatibility
    @property
    def segment_similarity(self) -> float:
        """Alias for visual_similarity (backward compatibility)."""
        return self.visual_similarity


@dataclass
class ScreenSegment:
    """A candidate region from screen segmentation.

    Attributes:
        bounding_box: Location of segment on screen
        image: Cropped image of the segment
        metadata: Additional metadata about the segment
    """

    bounding_box: BoundingBox
    image: Image.Image
    metadata: dict[str, Any] = field(default_factory=dict)


class ScreenSegmenter:
    """Segment screen into candidate regions for element matching.

    This is a placeholder implementation using grid-based segmentation.
    Will be replaced with more sophisticated region proposal in the future.
    """

    def __init__(
        self,
        grid_size: int = 16,
        overlap: float = 0.2,
        min_size: int = 32,
    ) -> None:
        """Initialize screen segmenter.

        Args:
            grid_size: Number of grid cells in each dimension
            overlap: Overlap between adjacent segments (0.0-0.5)
            min_size: Minimum size in pixels for a segment
        """
        self.grid_size = grid_size
        self.overlap = overlap
        self.min_size = min_size
        logger.info(
            "initialized_screen_segmenter",
            grid_size=grid_size,
            overlap=overlap,
            min_size=min_size,
        )

    async def segment(self, screenshot: Image.Image) -> list[ScreenSegment]:
        """Segment screen into candidate regions.

        Currently uses simple grid-based segmentation as placeholder.
        Future versions will use adaptive segmentation based on visual features.

        Args:
            screenshot: Screenshot to segment

        Returns:
            List of screen segments
        """
        width, height = screenshot.size
        segments: list[ScreenSegment] = []

        # Calculate step size with overlap
        step_x = int(width / self.grid_size * (1 - self.overlap))
        step_y = int(height / self.grid_size * (1 - self.overlap))
        segment_width = int(width / self.grid_size)
        segment_height = int(height / self.grid_size)

        # Ensure minimum size
        if segment_width < self.min_size or segment_height < self.min_size:
            logger.warning(
                "segment_too_small",
                segment_width=segment_width,
                segment_height=segment_height,
                min_size=self.min_size,
            )
            # Fall back to single segment
            segments.append(
                ScreenSegment(
                    bounding_box=BoundingBox(x=0, y=0, width=width, height=height),
                    image=screenshot,
                    metadata={"type": "full_screen"},
                )
            )
            return segments

        # Generate grid segments
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                x = col * step_x
                y = row * step_y

                # Ensure we don't go out of bounds
                x = min(x, width - segment_width)
                y = min(y, height - segment_height)

                # Crop segment
                segment_img = screenshot.crop((x, y, x + segment_width, y + segment_height))

                segments.append(
                    ScreenSegment(
                        bounding_box=BoundingBox(
                            x=x, y=y, width=segment_width, height=segment_height
                        ),
                        image=segment_img,
                        metadata={"grid_row": row, "grid_col": col},
                    )
                )

        logger.debug("screen_segmented", segment_count=len(segments))
        return segments


class RuntimeEmbedder:
    """Lightweight embedder wrapper optimized for runtime speed.

    Uses cached models and provides batch encoding for efficiency.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Initialize runtime embedder with cached models.

        Args:
            cache_dir: Directory for cached model weights
        """
        self.cache_dir = cache_dir
        logger.info(
            "initializing_runtime_embedder",
            cache_dir=str(cache_dir) if cache_dir else "default",
        )

        # Initialize embedders lazily
        self._text_embedder: TextEmbedder | None = None
        self._clip_embedder: CLIPEmbedder | None = None
        self._dinov2_embedder: DINOv2Embedder | None = None

    @property
    def text_embedder(self) -> TextEmbedder:
        """Get text embedder, initializing if needed."""
        if self._text_embedder is None:
            logger.info("loading_text_embedder")
            self._text_embedder = TextEmbedder(cache_dir=self.cache_dir)
        return self._text_embedder

    @property
    def clip_embedder(self) -> CLIPEmbedder:
        """Get CLIP embedder, initializing if needed."""
        if self._clip_embedder is None:
            logger.info("loading_clip_embedder")
            self._clip_embedder = CLIPEmbedder(cache_dir=self.cache_dir)
        return self._clip_embedder

    @property
    def dinov2_embedder(self) -> DINOv2Embedder:
        """Get DINOv2 embedder, initializing if needed."""
        if self._dinov2_embedder is None:
            logger.info("loading_dinov2_embedder")
            self._dinov2_embedder = DINOv2Embedder(cache_dir=self.cache_dir)
        return self._dinov2_embedder

    def encode_query(self, query: str) -> list[float]:
        """Encode text query to embedding.

        Args:
            query: Text query to encode

        Returns:
            Text embedding vector
        """
        return self.text_embedder.encode(query)

    def batch_encode_segments(self, segments: list[ScreenSegment]) -> list[list[float]]:
        """Batch encode segment images using DINOv2.

        Args:
            segments: List of screen segments to encode

        Returns:
            List of embedding vectors
        """
        images = [seg.image for seg in segments]
        return self.dinov2_embedder.batch_encode(images, batch_size=16)


class SearchSession:
    """Manages temporal context for element search.

    Tracks recent interactions to provide recency boosts for frequently
    accessed elements, states, and applications.
    """

    def __init__(self, max_history: int = 50) -> None:
        """Initialize search session.

        Args:
            max_history: Maximum number of interactions to track
        """
        self.max_history = max_history
        self.recent_elements: deque[str] = deque(maxlen=max_history)
        self.recent_states: deque[str] = deque(maxlen=max_history)
        self.recent_apps: deque[str] = deque(maxlen=max_history)
        logger.info("initialized_search_session", max_history=max_history)

    def record_interaction(self, element_id: str, state_id: str, app_id: str) -> None:
        """Record an element interaction.

        Args:
            element_id: ID of the element interacted with
            state_id: ID of the state containing the element
            app_id: ID of the application
        """
        self.recent_elements.append(element_id)
        self.recent_states.append(state_id)
        self.recent_apps.append(app_id)
        logger.debug(
            "recorded_interaction",
            element_id=element_id,
            state_id=state_id,
            app_id=app_id,
        )

    def get_temporal_boost(self, element_id: str, state_id: str, app_id: str) -> float:
        """Calculate recency boost for an element.

        Uses exponential decay based on position in recent history.

        Args:
            element_id: Element ID to check
            state_id: State ID to check
            app_id: Application ID to check

        Returns:
            Boost factor (0.0-1.0) where higher = more recent
        """
        boost = 0.0

        # Element boost (highest priority)
        element_positions = [
            i for i, eid in enumerate(reversed(self.recent_elements)) if eid == element_id
        ]
        if element_positions:
            # Most recent position gets highest boost
            recency = 1.0 - (element_positions[0] / len(self.recent_elements))
            boost += recency * 0.5  # Element boost up to 0.5

        # State boost (medium priority)
        state_positions = [
            i for i, sid in enumerate(reversed(self.recent_states)) if sid == state_id
        ]
        if state_positions:
            recency = 1.0 - (state_positions[0] / len(self.recent_states))
            boost += recency * 0.3  # State boost up to 0.3

        # App boost (lower priority)
        app_positions = [i for i, aid in enumerate(reversed(self.recent_apps)) if aid == app_id]
        if app_positions:
            recency = 1.0 - (app_positions[0] / len(self.recent_apps))
            boost += recency * 0.2  # App boost up to 0.2

        return min(boost, 1.0)  # Cap at 1.0


class RuntimeElementFinder:
    """Find elements at runtime by combining indexed knowledge with screen analysis.

    This class bridges the gap between indexed element knowledge (from RAG)
    and runtime screen analysis. It uses SAM3 segmentation with CLIP embeddings
    to find and match elements on screen.

    Features:
    - SAM3-based screenshot segmentation (with grid fallback)
    - Similarity threshold cascade: Project < StateImage < FindOptions
    - Two matching strategies: AVERAGE (default) or ANY_MATCH
    - OCR text filtering and scoring
    - Temporal context for recency boosting
    """

    def __init__(
        self,
        config_dir: Path,
        project_id: str,
        project_defaults: ProjectDefaults | None = None,
        cache_dir: Path | None = None,
        enable_ocr: bool = False,
    ) -> None:
        """Initialize runtime element finder.

        Args:
            config_dir: Directory containing project configuration and vector DB
            project_id: Project identifier
            project_defaults: Project-level default settings (lowest priority)
            cache_dir: Optional cache directory for model weights
            enable_ocr: Whether to enable OCR text extraction
        """
        self.config_dir = Path(config_dir)
        self.project_id = project_id
        self.cache_dir = cache_dir
        self.project_defaults = project_defaults or ProjectDefaults()

        # Initialize vector database
        db_path = self.config_dir / f"{project_id}.qvdb"
        self.db = QdrantLocalDB(db_path)
        self.index = RAGIndex(self.db)

        # Initialize segment vectorizer (replaces old segmenter + embedder)
        self._segment_vectorizer: SegmentVectorizer | None = None
        self._enable_ocr = enable_ocr

        # Keep legacy embedder for backward compatibility
        self.embedder = RuntimeEmbedder(cache_dir=cache_dir)

        # Optional session for temporal context
        self.session: SearchSession | None = None

        # Cache for segment vectors (cleared on each find call)
        self._cached_segments: list[SegmentVector] | None = None

        logger.info(
            "initialized_runtime_finder",
            project_id=project_id,
            db_path=str(db_path),
            ocr_enabled=enable_ocr,
            default_threshold=self.project_defaults.similarity_threshold,
            default_strategy=self.project_defaults.matching_strategy.value,
        )

    @property
    def segment_vectorizer(self) -> SegmentVectorizer:
        """Lazy-load segment vectorizer."""
        if self._segment_vectorizer is None:
            logger.info("loading_segment_vectorizer")
            self._segment_vectorizer = SegmentVectorizer(
                enable_ocr=self._enable_ocr,
            )
        return self._segment_vectorizer

    def _resolve_threshold(
        self,
        element_threshold: float | None,
        find_options: FindOptions | None,
    ) -> float:
        """Resolve effective similarity threshold using priority cascade.

        Priority order (highest to lowest):
        1. FindOptions.similarity_threshold (if provided)
        2. Element.similarity_threshold (if provided)
        3. ProjectDefaults.similarity_threshold (always present)

        Args:
            element_threshold: Element-level threshold (may be None)
            find_options: Find options override (may be None)

        Returns:
            Effective similarity threshold
        """
        # Highest priority: FindOptions
        if find_options and find_options.similarity_threshold is not None:
            return find_options.similarity_threshold

        # Medium priority: Element-level
        if element_threshold is not None:
            return element_threshold

        # Lowest priority: Project defaults
        return self.project_defaults.similarity_threshold

    def _resolve_strategy(
        self,
        element_strategy: MatchingStrategy | None,
        find_options: FindOptions | None,
    ) -> MatchingStrategy:
        """Resolve effective matching strategy using priority cascade.

        Priority order (highest to lowest):
        1. FindOptions.matching_strategy (if provided)
        2. Element.matching_strategy (if provided)
        3. ProjectDefaults.matching_strategy (always present)

        Args:
            element_strategy: Element-level strategy (may be None)
            find_options: Find options override (may be None)

        Returns:
            Effective matching strategy
        """
        # Highest priority: FindOptions
        if find_options and find_options.matching_strategy is not None:
            return find_options.matching_strategy

        # Medium priority: Element-level
        if element_strategy is not None:
            return element_strategy

        # Lowest priority: Project defaults
        return self.project_defaults.matching_strategy

    def _resolve_ocr_config(
        self,
        element_ocr_config: OCRConfig | None,
        find_options: FindOptions | None,
    ) -> OCRConfig | None:
        """Resolve effective OCR configuration using priority cascade.

        Args:
            element_ocr_config: Element-level OCR config (may be None)
            find_options: Find options override (may be None)

        Returns:
            Effective OCR configuration, or None if disabled
        """
        # Highest priority: FindOptions
        if find_options and find_options.ocr_config is not None:
            return find_options.ocr_config

        # Medium priority: Element-level
        if element_ocr_config is not None:
            return element_ocr_config

        # Lowest priority: Project defaults
        return self.project_defaults.ocr_config

    async def find_element(
        self,
        query: str,
        screenshot: Image.Image,
        filters: dict[str, Any] | None = None,
        find_options: FindOptions | None = None,
        top_k: int = 5,
    ) -> list[FoundElement]:
        """Find elements matching a text query on the current screen.

        This is the main entry point for runtime element finding. It:
        1. Encodes the text query to an embedding
        2. Searches the vector DB for similar indexed elements
        3. Segments the screenshot using SAM3 (or grid fallback)
        4. Matches indexed elements to screen segments
        5. Applies OCR filtering (if configured)
        6. Returns the best matches with locations

        Args:
            query: Text description of element to find
            screenshot: Current screenshot to search in
            filters: Optional filters for vector search
            find_options: Optional override settings (highest priority)
            top_k: Number of results to return

        Returns:
            List of found elements with locations, sorted by score
        """
        options = find_options or FindOptions()
        effective_top_k = options.max_results if options.max_results else top_k

        logger.info(
            "finding_element",
            query=query,
            filters=filters,
            top_k=effective_top_k,
        )

        # 1. Encode query
        query_embedding = self.embedder.encode_query(query)

        # 2. Search vector DB for candidate elements
        candidates: list[SearchResult] = await self.index.search_by_text(
            query_embedding=query_embedding,
            filters=filters,
            limit=effective_top_k * 3,  # Get more candidates for better matching
        )

        if not candidates:
            logger.warning("no_candidates_found", query=query)
            return []

        logger.debug("found_candidates", count=len(candidates))

        # 3. Segment and vectorize the screenshot using SAM3
        segment_vectors = self.segment_vectorizer.vectorize_screenshot(
            screenshot,
            max_segments=100,
            min_confidence=0.5,
        )

        if not segment_vectors:
            logger.warning("no_segments_found")
            return []

        logger.debug("screenshot_segmented", segment_count=len(segment_vectors))

        # 4. Match candidates to segments
        found_elements: list[FoundElement] = []

        for candidate in candidates:
            element = candidate.element

            # Get indexed embedding (prefer image embedding for visual matching)
            indexed_embedding = element.image_embedding
            if not indexed_embedding:
                logger.debug(
                    "skipping_element_no_embedding",
                    element_id=element.id,
                )
                continue

            # Resolve effective settings using cascade
            threshold = self._resolve_threshold(element.similarity_threshold, options)
            strategy = self._resolve_strategy(None, options)  # Element strategy from metadata
            ocr_config = self._resolve_ocr_config(None, options)

            # Get OCR filter text
            ocr_filter_text: str | None = None
            if options.ocr_filter and options.ocr_filter.text:
                ocr_filter_text = options.ocr_filter.text
            elif element.ocr_text:
                ocr_filter_text = element.ocr_text

            # Find matches based on strategy
            if strategy == MatchingStrategy.ANY_MATCH:
                # For ANY_MATCH, we'd need multiple pattern embeddings
                # Here we just use the single embedding as if it were one pattern
                match = self.segment_vectorizer.find_any_match(
                    pattern_embeddings=[indexed_embedding],
                    segments=segment_vectors,
                    threshold=threshold,
                )
                matches = [match] if match else []
            else:
                # AVERAGE strategy (default)
                matches = self.segment_vectorizer.find_matches(
                    query_embedding=indexed_embedding,
                    segments=segment_vectors,
                    threshold=threshold,
                    strategy=strategy,
                    ocr_filter=ocr_filter_text,
                )

            # Apply OCR filtering if configured
            if ocr_config and ocr_config.enabled and ocr_config.as_filter:
                matches = self._apply_ocr_filter(matches, options.ocr_filter, ocr_config)

            # Get best match for this element
            if matches:
                best_match = matches[0]

                # Calculate overall score
                overall_score = self._calculate_score(
                    candidate.score,
                    best_match,
                    ocr_config,
                )

                # Apply temporal boost if session exists
                if self.session:
                    temporal_boost = self.session.get_temporal_boost(
                        element.id,
                        element.state_id or "",
                        element.source_app or "",
                    )
                    overall_score = overall_score * (1.0 + temporal_boost * 0.3)

                # Create bounding box from segment
                x, y, w, h = best_match.segment.bbox
                bbox = BoundingBox(x=x, y=y, width=w, height=h)

                found_elements.append(
                    FoundElement(
                        element_id=element.id,
                        score=overall_score,
                        bounding_box=bbox,
                        visual_similarity=best_match.visual_similarity,
                        text_similarity=best_match.text_similarity,
                        ocr_similarity=best_match.ocr_similarity,
                        ocr_text=best_match.ocr_text,
                        metadata=(
                            {
                                "element_type": (
                                    element.element_type.value
                                    if element.element_type
                                    else "unknown"
                                ),
                                "text_description": element.text_description,
                                "expected_ocr_text": element.ocr_text,
                                "vector_score": candidate.score,
                                "threshold_used": threshold,
                                "strategy_used": strategy.value,
                                "segment_confidence": best_match.segment.confidence,
                            }
                            if options.include_metadata
                            else None
                        ),
                    )
                )

        # Sort by overall score
        found_elements.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            "element_finding_complete",
            query=query,
            found_count=len(found_elements),
        )

        return found_elements[:effective_top_k]

    async def find_by_element(
        self,
        element_id: str,
        screenshot: Image.Image,
        find_options: FindOptions | None = None,
    ) -> list[FoundElement]:
        """Find a specific element by its ID on the current screen.

        This method retrieves the element from the index and searches
        for it on the screenshot using its stored embeddings.

        Args:
            element_id: ID of the element to find
            screenshot: Current screenshot to search in
            find_options: Optional override settings

        Returns:
            List of found matches for this element
        """
        # Get element from index
        element = await self.index.get_element(element_id)
        if not element:
            logger.warning("element_not_found", element_id=element_id)
            return []

        # Use element's text description as query
        query = element.text_description or f"Element {element_id}"
        return await self.find_element(
            query=query,
            screenshot=screenshot,
            filters={"id": element_id},
            find_options=find_options,
        )

    def _apply_ocr_filter(
        self,
        matches: list[RAGMatch],
        ocr_filter: OCRFilter | None,
        ocr_config: OCRConfig,
    ) -> list[RAGMatch]:
        """Apply OCR-based filtering to matches.

        Args:
            matches: List of candidate matches
            ocr_filter: OCR filter criteria
            ocr_config: OCR configuration

        Returns:
            Filtered matches
        """
        if not ocr_filter or not ocr_filter.text:
            return matches

        filtered: list[RAGMatch] = []
        filter_text = ocr_filter.text.lower()

        for match in matches:
            if not match.segment.ocr_text:
                continue

            segment_text = match.segment.ocr_text.lower()

            # Check based on match mode
            if ocr_filter.match_mode == OCRMatchMode.EXACT:
                if segment_text == filter_text:
                    filtered.append(match)
            elif ocr_filter.match_mode == OCRMatchMode.CONTAINS:
                if filter_text in segment_text:
                    filtered.append(match)
            elif ocr_filter.match_mode == OCRMatchMode.REGEX:
                import re

                try:
                    if re.search(filter_text, segment_text):
                        filtered.append(match)
                except re.error:
                    logger.warning("invalid_regex_pattern", pattern=filter_text)

        return filtered

    def _calculate_score(
        self,
        vector_score: float,
        match: RAGMatch,
        ocr_config: OCRConfig | None,
    ) -> float:
        """Calculate overall score combining vector search and visual matching.

        The score combines:
        - Vector search score (from RAG index): 40%
        - Visual similarity: 40%
        - Text similarity: 15% (if available)
        - OCR similarity: 5% (if configured)

        Args:
            vector_score: Score from vector DB search
            match: The RAGMatch result
            ocr_config: OCR configuration (affects OCR weight)

        Returns:
            Combined score (0.0-1.0)
        """
        score = vector_score * 0.4 + match.visual_similarity * 0.4

        # Add text similarity if available
        if match.text_similarity is not None:
            score += match.text_similarity * 0.15
        else:
            # Redistribute to visual
            score += match.visual_similarity * 0.15

        # Add OCR similarity if available and configured
        if ocr_config and ocr_config.enabled and match.ocr_similarity is not None:
            ocr_weight = min(ocr_config.weight, 0.05)  # Cap at 5%
            score += match.ocr_similarity * ocr_weight
        else:
            # Redistribute to visual
            score += match.visual_similarity * 0.05

        return min(score, 1.0)

    def _extract_metadata(self, segment: ScreenSegment, screenshot: Image.Image) -> dict[str, Any]:
        """Extract metadata from a screen segment.

        Currently extracts basic visual features. Can be extended to include
        OCR, color analysis, edge detection, etc.

        Args:
            segment: Screen segment to analyze
            screenshot: Full screenshot for context

        Returns:
            Metadata dictionary
        """
        metadata: dict[str, Any] = {
            "position": {
                "x": segment.bounding_box.x,
                "y": segment.bounding_box.y,
                "width": segment.bounding_box.width,
                "height": segment.bounding_box.height,
            },
            "area": segment.bounding_box.width * segment.bounding_box.height,
        }

        # Calculate relative position (normalized to 0-1)
        screen_width, screen_height = screenshot.size
        metadata["relative_position"] = {
            "x": segment.bounding_box.x / screen_width,
            "y": segment.bounding_box.y / screen_height,
        }

        # Add grid metadata if available
        if "grid_row" in segment.metadata:
            metadata["grid_row"] = segment.metadata["grid_row"]
            metadata["grid_col"] = segment.metadata["grid_col"]

        return metadata

    def _apply_filters(
        self,
        segments: list[ScreenSegment],
        metadata: list[dict[str, Any]],
        filters: dict[str, Any],
    ) -> tuple[list[ScreenSegment], list[dict[str, Any]]]:
        """Apply spatial or visual filters to segments.

        Currently supports basic position filters. Can be extended for
        color, size, or other visual filters.

        Args:
            segments: List of screen segments
            metadata: List of metadata dicts (parallel to segments)
            filters: Filter criteria

        Returns:
            Filtered segments and metadata
        """
        filtered_segments: list[ScreenSegment] = []
        filtered_metadata: list[dict[str, Any]] = []

        for seg, meta in zip(segments, metadata, strict=False):
            keep = True

            # Position filters
            if "min_x" in filters and seg.bounding_box.x < filters["min_x"]:
                keep = False
            if "max_x" in filters and seg.bounding_box.x > filters["max_x"]:
                keep = False
            if "min_y" in filters and seg.bounding_box.y < filters["min_y"]:
                keep = False
            if "max_y" in filters and seg.bounding_box.y > filters["max_y"]:
                keep = False

            # Size filters
            if "min_area" in filters and meta["area"] < filters["min_area"]:
                keep = False
            if "max_area" in filters and meta["area"] > filters["max_area"]:
                keep = False

            if keep:
                filtered_segments.append(seg)
                filtered_metadata.append(meta)

        logger.debug(
            "applied_filters",
            original_count=len(segments),
            filtered_count=len(filtered_segments),
        )

        return filtered_segments, filtered_metadata

    def _find_matching_segment(
        self,
        indexed_embedding: list[float],
        segment_embeddings: list[list[float]],
        segments: list[ScreenSegment],
    ) -> ScreenSegment | None:
        """Find the segment that best matches the indexed element.

        Uses cosine similarity between indexed embedding and segment embeddings.

        Args:
            indexed_embedding: Embedding from indexed element
            segment_embeddings: Embeddings of screen segments
            segments: Screen segments (parallel to segment_embeddings)

        Returns:
            Best matching segment, or None if no good match
        """
        if not segment_embeddings:
            return None

        # Calculate similarities
        similarities = [
            self._cosine_similarity(indexed_embedding, seg_emb) for seg_emb in segment_embeddings
        ]

        # Find best match
        best_idx = int(np.argmax(similarities))
        best_score = similarities[best_idx]

        # Threshold for accepting a match
        threshold = 0.5
        if best_score < threshold:
            logger.debug(
                "no_good_segment_match",
                best_score=best_score,
                threshold=threshold,
            )
            return None

        logger.debug(
            "found_segment_match",
            segment_idx=best_idx,
            similarity=best_score,
        )

        return segments[best_idx]

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0.0-1.0)
        """
        a = np.array(vec1)
        b = np.array(vec2)

        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def enable_temporal_context(self, max_history: int = 50) -> SearchSession:
        """Enable temporal context tracking for this finder.

        Args:
            max_history: Maximum number of interactions to track

        Returns:
            The created SearchSession instance
        """
        self.session = SearchSession(max_history=max_history)
        logger.info("enabled_temporal_context", max_history=max_history)
        return self.session
