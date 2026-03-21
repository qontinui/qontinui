"""Batch template matching detection backend.

Wraps BatchTemplateMatcher as a DetectionBackend for use in the
CascadeDetector when multiple needles are provided. Amortizes
screenshot processing across all templates.
"""

import logging
from typing import Any

from .base import DetectionBackend, DetectionResult

logger = logging.getLogger(__name__)


class BatchTemplateMatchBackend(DetectionBackend):
    """Detection backend using batch multi-template matching.

    Wraps ``BatchTemplateMatcher`` from ``find.matchers.batch_template_matcher``.
    More efficient than single-template backend when multiple needles are
    searched simultaneously (~25ms amortized vs ~20ms per template).

    Args:
        nms_overlap_threshold: IoU threshold for cross-template NMS (0.0-1.0).
    """

    def __init__(self, nms_overlap_threshold: float = 0.3) -> None:
        self._nms_overlap_threshold = nms_overlap_threshold
        self._matcher: Any = None

    def _get_matcher(self) -> Any:
        if self._matcher is None:
            from ..matchers.batch_template_matcher import BatchTemplateMatcher

            self._matcher = BatchTemplateMatcher(
                nms_overlap_threshold=self._nms_overlap_threshold,
            )
        return self._matcher

    def find(self, needle: Any, haystack: Any, config: dict[str, Any]) -> list[DetectionResult]:
        """Find multiple template needles in haystack screenshot.

        Args:
            needle: A list of ``Pattern`` objects, or a single ``Pattern``.
            haystack: Screenshot as PIL Image, numpy array, or OpenCV mat.
            config: Keys used: ``min_confidence``, ``search_region``.

        Returns:
            List of DetectionResult sorted by confidence.
        """
        from ...model.element import Pattern

        # Accept single pattern or list
        if isinstance(needle, Pattern):
            patterns = [needle]
        elif isinstance(needle, list) and all(isinstance(p, Pattern) for p in needle):
            patterns = needle
        else:
            logger.debug("BatchTemplateMatchBackend: needle is not Pattern(s), skipping")
            return []

        matcher = self._get_matcher()
        similarity = config.get("min_confidence", 0.8)
        search_region = config.get("search_region")

        try:
            batch_results = matcher.find_all_patterns(
                screenshot=haystack,
                patterns=patterns,
                similarity=similarity,
                search_region=search_region,
            )
        except Exception:
            logger.debug("BatchTemplateMatchBackend: matching failed", exc_info=True)
            return []

        # Flatten all pattern results into a single DetectionResult list
        results: list[DetectionResult] = []
        for label, matches in batch_results.items():
            for match in matches:
                region = match.get_region()
                if region is None:
                    continue
                results.append(
                    DetectionResult(
                        x=region.x,
                        y=region.y,
                        width=region.width,
                        height=region.height,
                        confidence=match.similarity,
                        backend_name=self.name,
                        label=label,
                    )
                )

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def supports(self, needle_type: str) -> bool:
        """Supports template and multi_template needle types."""
        return needle_type in ("template", "multi_template")

    def estimated_cost_ms(self) -> float:
        """~25ms amortized across multiple templates."""
        return 25.0

    @property
    def name(self) -> str:
        return "batch_template_match"
