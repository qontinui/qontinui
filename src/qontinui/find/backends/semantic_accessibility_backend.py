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
import re
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
            from qontinui.hal.implementations.accessibility.uia_semantic import SemanticSearchCache

            self._cache = SemanticSearchCache()
        return self._cache

    def find(
        self, needle: Any, haystack: Any, config: dict[str, Any]
    ) -> list[DetectionResult]:
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
            cache = self._get_cache() if self._cache_enabled else None
            if cache is not None:
                cached = cache.get(app_key, needle)
                if cached is not None:
                    return self._matches_to_results(cached, min_confidence)

            # Fuzzy match
            from qontinui.hal.implementations.accessibility.uia_semantic import fuzzy_match_nodes

            matches = fuzzy_match_nodes(
                needle,
                snapshot,
                min_score=min_confidence,
                max_results=5,
                cache=cache,
            )

            # LLM fallback: if best fuzzy score is below llm_threshold
            # and an LLM client is available, use LLM for higher accuracy.
            best_fuzzy_score = matches[0].score if matches else 0.0
            if best_fuzzy_score < self._llm_threshold and self._llm_client is not None:
                llm_results = self._llm_find(needle, snapshot, min_confidence)
                if llm_results:
                    # LLM results are DetectionResults, not SemanticMatches —
                    # don't cache them in the SemanticMatch cache (wrong type).
                    # The next call will re-run fuzzy + LLM if needed.
                    return llm_results

            # Cache results
            if self._cache_enabled and matches:
                self._get_cache().put(app_key, needle, matches)

            return self._matches_to_results(matches, min_confidence)

        except Exception:
            logger.exception("SemanticAccessibilityBackend: search failed")
            return []

    def _llm_find(
        self, description: str, snapshot: Any, min_confidence: float
    ) -> list[DetectionResult]:
        """Use LLM to select the best matching element from the tree.

        Sends the accessibility tree and description to the LLM client
        and parses the response to extract the selected element index.

        Returns:
            List with a single DetectionResult if the LLM found a match,
            empty list otherwise.
        """
        try:
            from qontinui.hal.implementations.accessibility.uia_semantic import (
                _flatten_nodes,
                format_nodes_for_llm,
            )

            indexed_list = format_nodes_for_llm(snapshot)
            if not indexed_list.strip():
                return []

            prompt = (
                "You are an element selector. Given a list of interactive UI "
                "elements from an accessibility tree and a description, find "
                "the element that best matches.\n\n"
                f"Interactive Elements:\n{indexed_list}\n\n"
                f'Description: "{description}"\n\n'
                "Return your answer in this exact format:\n"
                'INDEX: <number or "none">\n'
                "CONFIDENCE: <0.0 to 1.0>"
            )

            assert self._llm_client is not None
            response = self._run_async(self._llm_client.complete(prompt))

            # Parse INDEX
            index_match = re.search(r"INDEX:\s*(\d+|none)", response, re.IGNORECASE)
            if not index_match or index_match.group(1).lower() == "none":
                return []
            index = int(index_match.group(1))

            # Parse CONFIDENCE
            conf_match = re.search(r"CONFIDENCE:\s*([\d.]+)", response, re.IGNORECASE)
            confidence = float(conf_match.group(1)) if conf_match else 0.5

            if confidence < min_confidence:
                return []

            # Look up the node by index in the flattened list
            nodes = _flatten_nodes(snapshot.root, interactive_only=True)
            if index < 0 or index >= len(nodes):
                return []

            node = nodes[index]
            bounds = getattr(node, "bounds", None)
            if bounds is None:
                return []

            if hasattr(bounds, "x"):
                x, y, w, h = bounds.x, bounds.y, bounds.width, bounds.height
            elif isinstance(bounds, list | tuple) and len(bounds) >= 4:
                x, y, w, h = (
                    int(bounds[0]),
                    int(bounds[1]),
                    int(bounds[2]),
                    int(bounds[3]),
                )
            else:
                return []

            return [
                DetectionResult(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    confidence=confidence,
                    backend_name=self.name,
                    label=node.name,
                    metadata={
                        "role": getattr(node.role, "value", str(node.role)),
                        "ref": node.ref,
                        "match_type": "llm",
                        "matched_term": description,
                        "automation_id": node.automation_id,
                    },
                )
            ]

        except Exception:
            logger.exception("SemanticAccessibilityBackend: LLM fallback failed")
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
            elif isinstance(bounds, list | tuple) and len(bounds) >= 4:
                x, y, w, h = (
                    int(bounds[0]),
                    int(bounds[1]),
                    int(bounds[2]),
                    int(bounds[3]),
                )
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
        return 200.0 if self._llm_client is not None else 10.0

    @property
    def name(self) -> str:
        return "semantic_accessibility"

    def is_available(self) -> bool:
        return self._capture is not None and self._capture.is_connected()

    def invalidate_cache(self, app_key: str | None = None) -> None:
        """Clear cached search results."""
        if self._cache is not None:
            self._cache.invalidate(app_key)
