"""Interactability pre-filter for DetectionBackends.

Wraps any ``DetectionBackend`` and gates its ``DetectionResult`` list through
OmniParser's YOLO interactability head. Candidates whose bboxes don't overlap
a region the head classifies as interactive are dropped.

This is a confidence gate, not a replacement. It exists so vision-based
backends (OmniParser's own semantic detector, Vision LLM / Aria-UI) stop
returning matches that land on non-interactive pixels — canvas interiors,
decorative icons, text labels for disabled controls.

Enabled via ``QONTINUI_OMNIPARSER_PREFILTER=true``. The cascade wraps all
backends whose ``estimated_cost_ms()`` exceeds a configurable threshold
(default 1000ms) so cheap template/feature backends aren't penalised.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from .base import DetectionBackend, DetectionResult

logger = logging.getLogger(__name__)


def prefilter_enabled() -> bool:
    """Check the env flag. Separate helper so cascade can short-circuit."""
    return os.environ.get("QONTINUI_OMNIPARSER_PREFILTER", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class InteractabilityFilter(DetectionBackend):
    """Decorates a wrapped backend with an OmniParser interactability gate.

    On each ``find()`` call:
        1. Delegates to the wrapped backend.
        2. Runs OmniParser's YOLO head once on the haystack.
        3. Drops any DetectionResult whose bbox doesn't overlap an
           interactive region at IoU >= ``iou_threshold``.
        4. Annotates surviving results with
           ``metadata["interactability_filter"]``.

    The wrapped backend's ``name``, ``estimated_cost_ms``, and ``supports``
    are delegated so cascade ordering is unaffected.
    """

    def __init__(
        self,
        wrapped: DetectionBackend,
        iou_threshold: float = 0.3,
        detector: Any | None = None,
    ) -> None:
        self._wrapped = wrapped
        self._iou_threshold = iou_threshold
        self._detector = detector  # lazy; only constructed on first use

    def find(
        self, needle: Any, haystack: Any, config: dict[str, Any]
    ) -> list[DetectionResult]:
        results = self._wrapped.find(needle, haystack, config)
        if not results:
            return results

        if not isinstance(haystack, np.ndarray):
            logger.debug(
                "InteractabilityFilter(%s): haystack not a numpy array, skipping gate",
                self._wrapped.name,
            )
            return results

        detector = self._get_detector()
        if detector is None:
            return results

        try:
            interactive = detector.get_interactive_regions(haystack)
        except Exception:
            logger.warning(
                "InteractabilityFilter(%s): YOLO pre-filter failed, passing through",
                self._wrapped.name,
                exc_info=True,
            )
            return results

        if not interactive:
            logger.debug(
                "InteractabilityFilter(%s): YOLO found no interactive regions, "
                "passing through %d candidates unfiltered",
                self._wrapped.name,
                len(results),
            )
            return results

        kept: list[DetectionResult] = []
        dropped = 0
        for r in results:
            is_interactive, conf = detector.classify_region(
                haystack,
                (r.x, r.y, r.width, r.height),
                iou_threshold=self._iou_threshold,
                interactive_regions=interactive,
            )
            if is_interactive:
                r.metadata.setdefault("interactability_filter", {})
                r.metadata["interactability_filter"] = {
                    "passed": True,
                    "yolo_confidence": conf,
                }
                kept.append(r)
            else:
                dropped += 1

        if dropped:
            logger.info(
                "InteractabilityFilter(%s): kept %d/%d candidates (dropped %d non-interactive)",
                self._wrapped.name,
                len(kept),
                len(results),
                dropped,
            )
        return kept

    def _get_detector(self) -> Any | None:
        if self._detector is not None:
            return self._detector
        try:
            from qontinui.discovery.element_detection.omniparser_detector import OmniParserDetector
            from qontinui.find.backends.omniparser_config import OmniParserSettings

            settings = OmniParserSettings()
            if not settings.enabled:
                logger.debug(
                    "InteractabilityFilter: OmniParserSettings.enabled=false, pre-filter is a no-op"
                )
                return None
            self._detector = OmniParserDetector(settings=settings)
            return self._detector
        except Exception:
            logger.warning(
                "InteractabilityFilter: could not construct OmniParserDetector, "
                "pre-filter disabled",
                exc_info=True,
            )
            return None

    def supports(self, needle_type: str) -> bool:
        return self._wrapped.supports(needle_type)

    def estimated_cost_ms(self) -> float:
        return self._wrapped.estimated_cost_ms()

    @property
    def name(self) -> str:
        return self._wrapped.name

    def is_available(self) -> bool:
        return self._wrapped.is_available()

    @property
    def wrapped(self) -> DetectionBackend:
        """Expose the underlying backend (used by cascade diagnostics)."""
        return self._wrapped
