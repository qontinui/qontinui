"""Scale-and-rotation-invariant template matching detection backend.

Wraps the HAL-level ``find_template_invariant()`` as a DetectionBackend
for use in the CascadeDetector. More expensive than plain template matching
(~120ms vs ~20ms) but handles DPI variance across displays.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import DetectionBackend, DetectionResult

logger = logging.getLogger(__name__)


class InvariantMatchBackend(DetectionBackend):
    """Detection backend using scale/rotation-invariant template matching.

    Sits between the fast template backend (~20ms) and OmniParser (~1500ms)
    in the cascade. Only activates as a fallback when standard matching
    fails, or when explicitly requested via ``MatchSettings(preferred_backend="invariant_template")``.

    Config keys consumed from *config* dict:
        invariant_scales (list[float]): Override default DPI scale factors.
        invariant_rotations (list[float]): Override default rotations (default ``[0]``).
        color_reference (tuple[int,int,int]): If set, apply color-difference post-filter.
        color_tolerance (float): Max mean RGB Euclidean distance (default 50.0).
    """

    def __init__(self) -> None:
        self._matcher: Any | None = None

    def _get_matcher(self) -> Any:
        """Lazy-import the HAL OpenCVMatcher to avoid circular dependencies."""
        if self._matcher is None:
            from ...hal.implementations.opencv_matcher import OpenCVMatcher

            self._matcher = OpenCVMatcher()
        return self._matcher

    def find(self, needle: Any, haystack: Any, config: dict[str, Any]) -> list[DetectionResult]:
        """Find needle template at multiple scales/rotations.

        Args:
            needle: A ``Pattern`` object with ``pixel_data``.
            haystack: Screenshot as PIL Image, numpy array, or OpenCV mat.
            config: Keys used: ``min_confidence``, ``invariant_scales``,
                    ``invariant_rotations``, ``color_reference``, ``color_tolerance``.

        Returns:
            List of DetectionResult (at most one for invariant matching).
        """
        from ...model.element import Pattern

        if not isinstance(needle, Pattern):
            logger.debug("InvariantMatchBackend: needle is not a Pattern, skipping")
            return []

        # Extract the needle and convert to PIL Image
        import cv2
        import numpy as np
        from PIL import Image

        needle_data = needle.pixel_data
        if needle_data is None:
            logger.debug("InvariantMatchBackend: Pattern has no pixel_data")
            return []
        if isinstance(needle_data, np.ndarray) and needle_data.size == 0:
            logger.debug("InvariantMatchBackend: Pattern has no pixel_data")
            return []

        # Pattern.pixel_data is np.ndarray (BGR) — convert to PIL RGB
        if isinstance(needle_data, np.ndarray):
            if len(needle_data.shape) == 3 and needle_data.shape[2] == 3:
                needle_image = Image.fromarray(cv2.cvtColor(needle_data, cv2.COLOR_BGR2RGB))
            else:
                needle_image = Image.fromarray(needle_data)
        elif isinstance(needle_data, Image.Image):
            needle_image = needle_data
        else:
            logger.debug("InvariantMatchBackend: unsupported pixel_data type")
            return []

        # Convert haystack to PIL Image if needed
        if not isinstance(haystack, Image.Image):
            if isinstance(haystack, np.ndarray):
                if len(haystack.shape) == 3 and haystack.shape[2] == 3:
                    haystack = Image.fromarray(cv2.cvtColor(haystack, cv2.COLOR_BGR2RGB))
                else:
                    haystack = Image.fromarray(haystack)
            else:
                logger.debug("InvariantMatchBackend: unsupported haystack type")
                return []

        matcher = self._get_matcher()
        confidence = config.get("min_confidence", 0.8)
        scales = config.get("invariant_scales")
        rotations = config.get("invariant_rotations")
        find_all = config.get("find_all", False)

        # Use DPI-aware scales when no explicit scales are provided
        if scales is None:
            scales = matcher.dpi_aware_scales()

        results: list[DetectionResult] = []

        if find_all:
            hal_matches = matcher.find_all_template_invariant(
                haystack=haystack,
                needle=needle_image,
                scales=scales,
                rotations=rotations,
                confidence=confidence,
            )
            for m in hal_matches:
                results.append(
                    DetectionResult(
                        x=m.x,
                        y=m.y,
                        width=m.width,
                        height=m.height,
                        confidence=m.confidence,
                        backend_name=self.name,
                    )
                )
        else:
            match = matcher.find_template_invariant(
                haystack=haystack,
                needle=needle_image,
                scales=scales,
                rotations=rotations,
                confidence=confidence,
            )
            if match is not None:
                results.append(
                    DetectionResult(
                        x=match.x,
                        y=match.y,
                        width=match.width,
                        height=match.height,
                        confidence=match.confidence,
                        backend_name=self.name,
                    )
                )

        if not results:
            return []

        # Apply color-difference filter if configured
        color_ref = config.get("color_reference")
        if color_ref is not None:
            from ..utils.color_filter import ColorDifferenceFilter

            color_tolerance = config.get("color_tolerance", 50.0)
            color_filter = ColorDifferenceFilter(
                reference_color=color_ref,
                tolerance=color_tolerance,
            )

            haystack_bgr = cv2.cvtColor(np.array(haystack), cv2.COLOR_RGB2BGR)
            results = color_filter.filter_results(results, haystack_bgr)

        return results

    def supports(self, needle_type: str) -> bool:
        """Only handles template-type needles."""
        return needle_type == "template"

    def estimated_cost_ms(self) -> float:
        """~120ms for scale-only (7 scales), up to ~600ms with rotations."""
        return 120.0

    @property
    def name(self) -> str:
        return "invariant_template"
