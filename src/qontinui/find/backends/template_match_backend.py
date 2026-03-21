"""Template matching detection backend.

Wraps the existing TemplateMatcher as a DetectionBackend for use in the
CascadeDetector fallback chain. This is typically the fastest vision-based
backend (~20ms).
"""

import logging
from typing import Any

from .base import DetectionBackend, DetectionResult

logger = logging.getLogger(__name__)


class TemplateMatchBackend(DetectionBackend):
    """Detection backend using OpenCV template matching.

    Wraps ``TemplateMatcher`` from ``find.matchers.template_matcher``.
    Fast and reliable for pixel-exact matches but sensitive to scale
    and rotation changes.

    Args:
        matcher: An existing TemplateMatcher instance. If None, a default
                 one is created on first use (lazy import to avoid cycles).
    """

    def __init__(self, matcher: Any | None = None) -> None:
        self._matcher = matcher

    def _get_matcher(self) -> Any:
        if self._matcher is None:
            from ..matchers.template_matcher import TemplateMatcher

            self._matcher = TemplateMatcher()
        return self._matcher

    def find(self, needle: Any, haystack: Any, config: dict[str, Any]) -> list[DetectionResult]:
        """Find needle template in haystack screenshot.

        Args:
            needle: A ``Pattern`` object with ``pixel_data``.
            haystack: Screenshot as PIL Image, numpy array, or OpenCV mat.
            config: Keys used: ``min_confidence``, ``find_all``, ``search_region``.

        Returns:
            List of DetectionResult sorted by confidence.
        """
        from ...model.element import Pattern

        if not isinstance(needle, Pattern):
            logger.debug("TemplateMatchBackend: needle is not a Pattern, skipping")
            return []

        matcher = self._get_matcher()
        similarity = config.get("min_confidence", 0.8)
        find_all = config.get("find_all", False)
        search_region = config.get("search_region")

        try:
            matches = matcher.find_matches(
                screenshot=haystack,
                pattern=needle,
                find_all=find_all,
                similarity=similarity,
                search_region=search_region,
            )
        except Exception:
            logger.exception("TemplateMatchBackend: matching failed")
            return []

        results: list[DetectionResult] = []
        for m in matches:
            region = m.get_region()
            if region is None:
                continue
            results.append(
                DetectionResult(
                    x=region.x,
                    y=region.y,
                    width=region.width,
                    height=region.height,
                    confidence=m.confidence,
                    backend_name=self.name,
                )
            )

        return results

    def supports(self, needle_type: str) -> bool:
        return needle_type == "template"

    def estimated_cost_ms(self) -> float:
        return 20.0

    @property
    def name(self) -> str:
        return "template"
