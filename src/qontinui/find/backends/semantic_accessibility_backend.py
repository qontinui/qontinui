"""Semantic accessibility detection backend.

Extends the AccessibilityBackend with natural-language fuzzy matching.
Accepts a ``description`` needle type and uses semantic search to find
elements in the accessibility tree by name similarity, role hints, and
automation ID matching.

Cost: ~10ms (tree traversal + fuzzy matching, no LLM call).
Falls between AccessibilityBackend (5ms, exact match only) and
TemplateMatchBackend (~20ms, requires image).

Optionally supports LLM-assisted matching for higher accuracy when a
VisionLLMClient or LLMClient is provided (bumps cost to ~200ms).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .base import DetectionBackend, DetectionResult

logger = logging.getLogger(__name__)


class SemanticAccessibilityBackend(DetectionBackend):
    """Detection backend using semantic matching against accessibility tree.

    Accepts ``description`` needle type — a natural-language string like
    "the search bar" or "Submit button". Uses fuzzy matching from
    ``uia_semantic.fuzzy_match_nodes`` to find matching elements.

    Args:
        capture: An ``IAccessibilityCapture`` implementation (UIA, CDP, etc.).
        llm_client: Optional LLM client for higher-accuracy matching.
            When provided, falls back to LLM if fuzzy matching confidence
            is below ``llm_threshold``.
        llm_threshold: Minimum fuzzy score below which LLM is consulted.
        cache_enabled: Whether to cache description→result mappings.
    """

    def __init__(
        self,
        capture: Any,
        *,
        llm_client: Any | None = None,
        llm_threshold: float = 0.7,
        cache_enabled: bool = True,
    ) -> None:
        self._capture = capture
        self._llm_client = llm_client
        self._llm_threshold = llm_threshold
        self._cache_enabled = cache_enabled

        # Lazy-init cache
        self._cache: Any | None = None

    def _get_cache(self) -> Any:
        if self._cache is None:
            from qontinui.hal.implementations.accessibility.uia_semantic import (
                SemanticSearchCache,
            )

            self._cache = SemanticSearchCache()
        return self._cache

    def find(self, needle: Any, haystack: Any, config: dict[str, Any]) -> list[DetectionResult]:
        """Find elements by natural-language description.

        Args:
            needle: Natural-language description string.
            haystack: Ignored (accessibility tree queried directly).
            config: Keys used:
                - ``needle_type``: Must be ``description`` or ``semantic``.
                - ``min_confidence``: Minimum confidence threshold.

        Returns:
            List of DetectionResult for matching elements.
        """
        if not isinstance(needle, str):
            return []

        min_confidence = config.get("min_confidence", 0.6)

        try:
            # Capture tree snapshot
            snapshot = self._run_async(self._capture.capture_tree())
            if snapshot is None:
                return []

            # Check cache
            app_key = snapshot.title or "unknown"
            if self._cache_enabled:
                cache = self._get_cache()
                cached = cache.get(app_key, needle)
                if cached is not None:
                    return self._matches_to_results(cached, min_confidence)

            # Fuzzy match
            from qontinui.hal.implementations.accessibility.uia_semantic import (
                fuzzy_match_nodes,
            )

            matches = fuzzy_match_nodes(
                needle,
                snapshot,
                min_score=min_confidence,
                max_results=5,
            )

            # Cache results
            if self._cache_enabled and matches:
                self._get_cache().put(app_key, needle, matches)

            return self._matches_to_results(matches, min_confidence)

        except Exception:
            logger.exception("SemanticAccessibilityBackend: search failed")
            return []

    def _matches_to_results(
        self, matches: list[Any], min_confidence: float
    ) -> list[DetectionResult]:
        """Convert SemanticMatch list to DetectionResult list."""
        results: list[DetectionResult] = []

        for match in matches:
            if match.score < min_confidence:
                continue

            node = match.node
            bounds = getattr(node, "bounds", None)
            if bounds is None:
                continue

            # Handle both AccessibilityBounds objects and tuples
            if hasattr(bounds, "x"):
                x, y, w, h = bounds.x, bounds.y, bounds.width, bounds.height
            elif isinstance(bounds, (list, tuple)) and len(bounds) >= 4:
                x, y, w, h = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])
            else:
                continue

            results.append(
                DetectionResult(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    confidence=match.score,
                    backend_name=self.name,
                    label=node.name,
                    metadata={
                        "role": getattr(node.role, "value", str(node.role)),
                        "ref": node.ref,
                        "match_type": match.match_type,
                        "matched_term": match.matched_term,
                        "automation_id": node.automation_id,
                    },
                )
            )

        return results

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine from sync context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result(timeout=5.0)
        else:
            return asyncio.run(coro)

    def supports(self, needle_type: str) -> bool:
        return needle_type in ("description", "semantic")

    def estimated_cost_ms(self) -> float:
        return 10.0

    @property
    def name(self) -> str:
        return "semantic_accessibility"

    def is_available(self) -> bool:
        return self._capture is not None and self._capture.is_connected()

    def invalidate_cache(self, app_key: str | None = None) -> None:
        """Clear cached search results."""
        if self._cache is not None:
            self._cache.invalidate(app_key)
