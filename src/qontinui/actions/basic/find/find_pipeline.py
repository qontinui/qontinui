"""Find pipeline - ported from Qontinui framework.

Orchestrates the complete find operation pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .strategy_registry import StrategyRegistry

from ....model.element.region import Region
from ....model.match.match import Match
from ...object_collection import ObjectCollection
from .options.base_find_options import BaseFindOptions

logger = logging.getLogger(__name__)


@dataclass
class FindPipeline:
    """Orchestrates complete find operation pipeline.

    Port of FindPipeline from Qontinui framework class.

    FindPipeline manages the execution flow of find operations,
    including preprocessing, strategy selection, execution,
    and post-processing of results.
    """

    # Strategy registry for delegating to implementations
    strategy_registry: StrategyRegistry | None = None

    # Pipeline configuration
    enable_preprocessing: bool = True
    enable_postprocessing: bool = True
    enable_caching: bool = False
    enable_profiling: bool = False

    # Cache for repeated searches
    _cache: dict[str, list[Match]] = field(default_factory=dict)

    # Performance metrics
    _metrics: dict[str, Any] = field(default_factory=dict)

    def execute(
        self, object_collection: ObjectCollection, options: BaseFindOptions
    ) -> list[Match]:
        """Execute the find pipeline.

        Args:
            object_collection: Objects to find
            options: Find configuration

        Returns:
            List of matches found
        """
        start_time = time.time()

        # Check cache if enabled
        if self.enable_caching:
            cache_key = self._generate_cache_key(object_collection, options)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for key: {cache_key}")
                return self._cache[cache_key]

        # Preprocessing
        if self.enable_preprocessing:
            object_collection, options = self._preprocess(object_collection, options)

        # Strategy selection and execution
        matches = self._execute_strategy(object_collection, options)

        # Postprocessing
        if self.enable_postprocessing:
            matches = self._postprocess(matches, options)

        # Update cache
        if self.enable_caching and cache_key:
            self._cache[cache_key] = matches

        # Record metrics
        if self.enable_profiling:
            self._metrics["last_execution_time"] = time.time() - start_time
            self._metrics["last_match_count"] = len(matches)

        return matches

    def _preprocess(
        self, object_collection: ObjectCollection, options: BaseFindOptions
    ) -> tuple[ObjectCollection, BaseFindOptions]:
        """Preprocess inputs before finding.

        Args:
            object_collection: Objects to preprocess
            options: Options to preprocess

        Returns:
            Preprocessed objects and options
        """
        logger.debug("Preprocessing find inputs")

        # Validate options
        if not options.validate():
            logger.warning("Invalid options detected during preprocessing")

        # Set default search region if none specified
        if not options.search_regions:
            # Use full screen as default
            # This would get the screen dimensions from the system
            logger.debug("No search regions specified, using full screen")

        # Filter objects based on options
        # For example, remove objects that don't meet minimum size requirements

        return object_collection, options

    def _execute_strategy(
        self, object_collection: ObjectCollection, options: BaseFindOptions
    ) -> list[Match]:
        """Execute the appropriate find strategy.

        Args:
            object_collection: Objects to find
            options: Find configuration

        Returns:
            Raw matches from strategy
        """
        strategy = options.get_strategy()
        logger.debug(f"Executing strategy: {strategy.name}")

        # Get strategy implementation from registry
        if self.strategy_registry:
            implementation = self.strategy_registry.get_implementation(strategy)
            if implementation:
                return cast(
                    list[Match], implementation.find(object_collection, options)
                )

        # Fallback to basic implementation
        logger.warning(f"No implementation found for strategy: {strategy.name}")
        return self._fallback_find(object_collection, options)

    def _fallback_find(
        self, object_collection: ObjectCollection, options: BaseFindOptions
    ) -> list[Match]:
        """Fallback find implementation.

        Basic implementation when strategy not available.

        Args:
            object_collection: Objects to find
            options: Find configuration

        Returns:
            Empty match list
        """
        logger.warning("Using fallback find implementation")

        # This would contain basic template matching or other fallback
        # For now, return empty list
        return []

    def _postprocess(
        self, matches: list[Match], options: BaseFindOptions
    ) -> list[Match]:
        """Postprocess matches after finding.

        Args:
            matches: Raw matches
            options: Find configuration

        Returns:
            Processed matches
        """
        logger.debug(f"Postprocessing {len(matches)} matches")

        # Filter by similarity threshold
        matches = [m for m in matches if m.similarity >= options.similarity]

        # Apply search type filtering
        matches = self._apply_search_type(matches, options)

        # Apply non-maximum suppression if enabled
        if hasattr(options, "non_max_suppression") and options.non_max_suppression:
            matches = self._apply_nms(matches, getattr(options, "nms_threshold", 0.5))

        # Limit to max_matches
        if len(matches) > options.max_matches:
            matches = matches[: options.max_matches]

        # Sort by similarity
        matches.sort(key=lambda m: m.similarity, reverse=True)

        return matches

    def _apply_search_type(
        self, matches: list[Match], options: BaseFindOptions
    ) -> list[Match]:
        """Apply search type filtering.

        Args:
            matches: Matches to filter
            options: Find configuration

        Returns:
            Filtered matches
        """
        from .options.base_find_options import SearchType

        if not matches:
            return matches

        if options.search_type == SearchType.FIRST:
            return matches[:1]
        elif options.search_type == SearchType.BEST:
            best_match = max(matches, key=lambda m: m.similarity)
            return [best_match]
        elif options.search_type == SearchType.ALL:
            return matches
        elif options.search_type == SearchType.EACH:
            # Return one match per unique pattern
            seen_patterns = set()
            unique_matches = []
            for match in matches:
                if (
                    hasattr(match, "pattern_id")
                    and match.pattern_id not in seen_patterns
                ):
                    seen_patterns.add(match.pattern_id)
                    unique_matches.append(match)
            return unique_matches

        return matches

    def _apply_nms(self, matches: list[Match], threshold: float) -> list[Match]:
        """Apply non-maximum suppression.

        Remove overlapping matches based on IoU threshold.

        Args:
            matches: Matches to filter
            threshold: IoU threshold for suppression

        Returns:
            Filtered matches
        """
        if not matches:
            return matches

        # Sort by similarity (highest first)
        matches = sorted(matches, key=lambda m: m.similarity, reverse=True)

        keep: list[Match] = []
        for match in matches:
            # Check overlap with already kept matches
            should_keep = True
            for kept_match in keep:
                if (
                    match.region
                    and kept_match.region
                    and self._calculate_iou(match.region, kept_match.region) > threshold
                ):
                    should_keep = False
                    break

            if should_keep:
                keep.append(match)

        return keep

    def _calculate_iou(self, region1: Region, region2: Region) -> float:
        """Calculate Intersection over Union of two regions.

        Args:
            region1: First region
            region2: Second region

        Returns:
            IoU value (0.0-1.0)
        """
        if not region1 or not region2:
            return 0.0

        # Calculate intersection
        x1 = max(region1.x, region2.x)
        y1 = max(region1.y, region2.y)
        x2 = min(region1.x + region1.width, region2.x + region2.width)
        y2 = min(region1.y + region1.height, region2.y + region2.height)

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)

        # Calculate union
        area1 = region1.width * region1.height
        area2 = region2.width * region2.height
        union = area1 + area2 - intersection

        if union == 0:
            return 0.0

        return intersection / union

    def _generate_cache_key(
        self, object_collection: ObjectCollection, options: BaseFindOptions
    ) -> str:
        """Generate cache key for find operation.

        Args:
            object_collection: Objects being searched
            options: Find configuration

        Returns:
            Cache key string
        """
        # Simple cache key generation
        # In production, would need more sophisticated hashing
        key_parts = [
            str(options.get_strategy()),
            str(options.similarity),
            str(len(object_collection.state_images)),
            str(len(object_collection.state_strings)),
        ]
        return "_".join(key_parts)

    def clear_cache(self):
        """Clear the find cache."""
        self._cache.clear()
        logger.debug("Find cache cleared")

    def get_metrics(self) -> dict[str, Any]:
        """Get pipeline metrics.

        Returns:
            Performance metrics
        """
        return self._metrics.copy()

    def reset_metrics(self):
        """Reset performance metrics."""
        self._metrics.clear()
        logger.debug("Metrics reset")


class PipelineBuilder:
    """Builder for FindPipeline.

    Provides fluent interface for pipeline configuration.
    """

    def __init__(self) -> None:
        """Initialize builder."""
        self._pipeline = FindPipeline()

    def with_registry(self, registry: StrategyRegistry) -> PipelineBuilder:
        """Set strategy registry.

        Args:
            registry: Strategy registry

        Returns:
            Self for fluent interface
        """
        self._pipeline.strategy_registry = registry
        return self

    def enable_caching(self) -> PipelineBuilder:
        """Enable result caching.

        Returns:
            Self for fluent interface
        """
        self._pipeline.enable_caching = True
        return self

    def enable_profiling(self) -> PipelineBuilder:
        """Enable performance profiling.

        Returns:
            Self for fluent interface
        """
        self._pipeline.enable_profiling = True
        return self

    def disable_preprocessing(self) -> PipelineBuilder:
        """Disable preprocessing.

        Returns:
            Self for fluent interface
        """
        self._pipeline.enable_preprocessing = False
        return self

    def disable_postprocessing(self) -> PipelineBuilder:
        """Disable postprocessing.

        Returns:
            Self for fluent interface
        """
        self._pipeline.enable_postprocessing = False
        return self

    def build(self) -> FindPipeline:
        """Build the pipeline.

        Returns:
            Configured FindPipeline
        """
        return cast(FindPipeline, self._pipeline)
