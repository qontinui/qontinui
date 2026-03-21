"""OmniParser detection backend for the CascadeDetector.

Wraps OmniParserDetector as a DetectionBackend so it can participate
in the graduated fallback chain. Positioned as Tier 2 between
feature matching (~100ms) and Vision LLM (~3000ms).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from qontinui.discovery.element_detection.omniparser_detector import (
    OmniParserDetector,
)
from qontinui.find.semantic_matcher import match_element_by_description

from .base import DetectionBackend, DetectionResult
from .omniparser_config import OmniParserSettings

logger = logging.getLogger(__name__)


class OmniParserBackend(DetectionBackend):
    """CascadeDetector backend that uses OmniParser for zero-shot element detection.

    When the needle is a template image, runs full detection and compares visually.
    When the needle is a text description, matches against OmniParser's semantic
    captions using fuzzy string matching.
    """

    def __init__(self, settings: OmniParserSettings | None = None) -> None:
        self._settings = settings or OmniParserSettings()
        self._detector: OmniParserDetector | None = None

    def _ensure_detector(self) -> OmniParserDetector:
        if self._detector is None:
            self._detector = OmniParserDetector(settings=self._settings)
        return self._detector

    def find(self, needle: Any, haystack: Any, config: dict[str, Any]) -> list[DetectionResult]:
        """Find needle in haystack using OmniParser.

        Args:
            needle: Description string, or numpy array (template image).
            haystack: Screenshot as numpy array (BGR).
            config: Dict with keys like needle_type, min_confidence, description.

        Returns:
            List of DetectionResult sorted by confidence.
        """
        detector = self._ensure_detector()
        needle_type = config.get("needle_type", "template")

        # Run OmniParser detection on the haystack screenshot
        if not isinstance(haystack, np.ndarray):
            logger.warning("OmniParserBackend expects numpy array haystack")
            return []

        elements = detector.detect_from_numpy(haystack)
        if not elements:
            return []

        # Match strategy depends on needle type
        if needle_type in ("description", "semantic", "text"):
            return self._match_by_description(needle, elements, config)
        elif needle_type == "template":
            return self._match_by_template(needle, haystack, elements, config)
        else:
            # Fall back to description matching if needle is a string
            if isinstance(needle, str):
                return self._match_by_description(needle, elements, config)
            return self._convert_all(elements)

    def _match_by_description(
        self, description: Any, elements: list, config: dict[str, Any]
    ) -> list[DetectionResult]:
        """Match a text description against detected element labels."""
        desc_str = str(description)
        min_sim = config.get("min_similarity", 0.4)

        labels = [e.label or "" for e in elements]
        types = [e.element_type for e in elements]

        matches = match_element_by_description(
            desc_str, labels, element_types=types, min_similarity=min_sim
        )

        results: list[DetectionResult] = []
        for m in matches:
            elem = elements[m.element_index]
            bb = elem.bounding_box
            results.append(
                DetectionResult(
                    x=bb.x,
                    y=bb.y,
                    width=bb.width,
                    height=bb.height,
                    confidence=m.score * elem.confidence,
                    backend_name=self.name,
                    label=elem.label,
                    metadata={
                        "element_type": elem.element_type,
                        "match_type": m.match_type,
                        "semantic_score": m.score,
                        "detection_confidence": elem.confidence,
                        **elem.metadata,
                    },
                )
            )

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def _match_by_template(
        self,
        needle: Any,
        haystack: np.ndarray,
        elements: list,
        config: dict[str, Any],
    ) -> list[DetectionResult]:
        """Match a template image against detected element regions.

        Crops each detected region from the haystack and computes template
        similarity against the needle.
        """
        import cv2

        if not isinstance(needle, np.ndarray):
            return self._convert_all(elements)

        needle_gray = cv2.cvtColor(needle, cv2.COLOR_BGR2GRAY) if len(needle.shape) == 3 else needle
        haystack_gray = (
            cv2.cvtColor(haystack, cv2.COLOR_BGR2GRAY) if len(haystack.shape) == 3 else haystack
        )

        results: list[DetectionResult] = []
        for elem in elements:
            bb = elem.bounding_box
            crop = haystack_gray[bb.y : bb.y + bb.height, bb.x : bb.x + bb.width]
            if crop.size == 0:
                continue

            # Resize needle to match crop size for comparison
            try:
                resized_needle = cv2.resize(needle_gray, (crop.shape[1], crop.shape[0]))
                match_result = cv2.matchTemplate(crop, resized_needle, cv2.TM_CCOEFF_NORMED)
                similarity = float(match_result.max())
            except Exception:
                continue

            if similarity > 0.3:
                results.append(
                    DetectionResult(
                        x=bb.x,
                        y=bb.y,
                        width=bb.width,
                        height=bb.height,
                        confidence=similarity * elem.confidence,
                        backend_name=self.name,
                        label=elem.label,
                        metadata={
                            "element_type": elem.element_type,
                            "template_similarity": similarity,
                            "detection_confidence": elem.confidence,
                            **elem.metadata,
                        },
                    )
                )

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def _convert_all(self, elements: list) -> list[DetectionResult]:
        """Convert all detected elements to DetectionResults (no filtering)."""
        return [
            DetectionResult(
                x=e.bounding_box.x,
                y=e.bounding_box.y,
                width=e.bounding_box.width,
                height=e.bounding_box.height,
                confidence=e.confidence,
                backend_name=self.name,
                label=e.label,
                metadata={
                    "element_type": e.element_type,
                    **e.metadata,
                },
            )
            for e in elements
        ]

    def supports(self, needle_type: str) -> bool:
        return needle_type in ("template", "text", "description", "semantic")

    def estimated_cost_ms(self) -> float:
        return 1500.0

    @property
    def name(self) -> str:
        return "omniparser"

    def is_available(self) -> bool:
        return self._settings.enabled
