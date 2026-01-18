"""Cached template matcher wrapper.

Provides caching layer on top of any ImageMatcher implementation.
Checks cache before performing expensive template matching.

This module emits cache hit/miss events for monitoring and metrics.
"""

import logging
import time
from typing import Any

import numpy as np

from ...cache import ActionCache, get_action_cache
from ...model.element import Pattern, Region
from ...reporting.events import EventType, emit_event
from ...reporting.schemas import HealingCacheEventData
from ..match import Match
from .image_matcher import ImageMatcher
from .template_matcher import TemplateMatcher

logger = logging.getLogger(__name__)


class CachedTemplateMatcher(ImageMatcher):
    """Template matcher with caching for deterministic replay.

    Wraps any ImageMatcher (default TemplateMatcher) and adds a caching layer.
    On cache hit, returns cached coordinates without performing template matching.
    On cache miss, performs matching and stores result in cache.

    Attributes:
        matcher: Underlying ImageMatcher implementation.
        cache: ActionCache for storing/retrieving matches.
        cache_enabled: Whether caching is active.
    """

    def __init__(
        self,
        matcher: ImageMatcher | None = None,
        cache: ActionCache | None = None,
        cache_enabled: bool = True,
        state_id: str | None = None,
    ) -> None:
        """Initialize cached matcher.

        Args:
            matcher: Underlying matcher. Defaults to TemplateMatcher().
            cache: ActionCache to use. Defaults to global cache.
            cache_enabled: Whether caching is enabled. Default True.
            state_id: Optional state ID for cache key generation.
        """
        self.matcher = matcher or TemplateMatcher()
        self.cache = cache or get_action_cache()
        self.cache_enabled = cache_enabled
        self.state_id = state_id

        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0

    def find_matches(
        self,
        screenshot: Any,
        pattern: Pattern,
        find_all: bool = False,
        similarity: float = 0.8,
        search_region: tuple[int, int, int, int] | None = None,
    ) -> list[Match]:
        """Find template matches with caching.

        Checks cache first. On hit, returns cached match.
        On miss, delegates to underlying matcher and caches result.

        Args:
            screenshot: Screenshot image.
            pattern: Pattern to search for.
            find_all: If True, find all matches (cache only stores best).
            similarity: Minimum similarity threshold.
            search_region: Optional search region.

        Returns:
            List of Match objects.
        """
        # Convert screenshot for cache validation
        screenshot_arr = self._to_numpy(screenshot)

        # Try cache if enabled and not finding all (cache stores single best)
        if self.cache_enabled and not find_all:
            cache_key = self.cache.build_key(
                pattern=pattern,
                state_id=self.state_id,
                action_type="find",
            )

            result = self.cache.try_get(
                key=cache_key,
                screenshot=screenshot_arr,
                pattern=pattern,
            )

            if result.hit and result.entry:
                self._cache_hits += 1
                coords = result.entry.coordinates
                logger.debug(
                    f"Cache hit for pattern '{pattern.name}' at " f"({coords.x}, {coords.y})"
                )

                # Emit cache hit event
                self._emit_cache_event(pattern, cache_hit=True)

                # Reconstruct Match from cached data
                from ...model.element import Location
                from ...model.match import Match as MatchObject

                match_obj = MatchObject(
                    target=Location(
                        x=coords.x,
                        y=coords.y,
                        region=Region(
                            coords.region_x,
                            coords.region_y,
                            coords.region_width,
                            coords.region_height,
                        ),
                    ),
                    score=result.entry.confidence,
                    name=pattern.name,
                )
                return [Match(match_obj)]

            self._cache_misses += 1
            # Emit cache miss event
            self._emit_cache_event(pattern, cache_hit=False)

        # Perform actual matching
        matches = self.matcher.find_matches(
            screenshot=screenshot,
            pattern=pattern,
            find_all=find_all,
            similarity=similarity,
            search_region=search_region,
        )

        # Cache best result if found (and not finding all)
        if self.cache_enabled and matches and not find_all:
            best_match = matches[0]
            if best_match.region:
                cache_key = self.cache.build_key(
                    pattern=pattern,
                    state_id=self.state_id,
                    action_type="find",
                )

                self.cache.store(
                    key=cache_key,
                    coordinates=(best_match.x, best_match.y),
                    region=best_match.region,
                    confidence=best_match.similarity,
                    pattern=pattern,
                )

        return matches

    def _to_numpy(self, screenshot: Any) -> np.ndarray:
        """Convert screenshot to numpy array.

        Args:
            screenshot: Screenshot in various formats.

        Returns:
            Numpy array (BGR).
        """
        if isinstance(screenshot, np.ndarray):
            return screenshot

        if hasattr(screenshot, "get_mat_bgr"):
            mat_bgr: np.ndarray = screenshot.get_mat_bgr()
            return mat_bgr

        # Try PIL conversion
        try:
            import cv2
            from PIL import Image as PILImage

            if isinstance(screenshot, PILImage.Image):
                rgb_array = np.array(screenshot)
                return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        except ImportError:
            pass

        # Last resort - assume it's array-like
        result: np.ndarray = np.asarray(screenshot)
        return result

    def set_state_id(self, state_id: str | None) -> None:
        """Set the state ID for cache key generation.

        Args:
            state_id: State ID to use in cache keys.
        """
        self.state_id = state_id

    def clear_cache(self) -> int:
        """Clear the action cache.

        Returns:
            Number of entries cleared.
        """
        return self.cache.clear()

    def get_stats(self) -> dict[str, int | float]:
        """Get cache statistics.

        Returns:
            Dictionary with cache hit/miss statistics.
        """
        total = self._cache_hits + self._cache_misses
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": (self._cache_hits / total * 100) if total > 0 else 0.0,
        }

    def _emit_cache_event(self, pattern: Pattern, cache_hit: bool) -> None:
        """Emit a cache hit or miss event.

        Args:
            pattern: The pattern that was looked up in cache.
            cache_hit: Whether the lookup was a hit (True) or miss (False).
        """
        try:
            total = self._cache_hits + self._cache_misses
            hit_rate = (self._cache_hits / total) if total > 0 else 0.0

            event_data = HealingCacheEventData(
                pattern_id=pattern.id,
                pattern_name=pattern.name,
                cache_hit=cache_hit,
                cache_size=len(self.cache.list_keys()) if hasattr(self.cache, "list_keys") else 0,
                hit_rate=hit_rate,
                total_hits=self._cache_hits,
                total_misses=self._cache_misses,
                timestamp=time.time(),
            )

            event_type = EventType.HEALING_CACHE_HIT if cache_hit else EventType.HEALING_CACHE_MISS
            emit_event(event_type, data=event_data.to_dict())
        except Exception as e:
            logger.debug(f"Failed to emit cache event: {e}")
