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

        # Extract the needle as a PIL Image
        from PIL import Image

        needle_image = needle.pixel_data
        if needle_image is None:
            logger.debug("InvariantMatchBackend: Pattern has no pixel_data")
            return []

        # Convert haystack to PIL Image if needed
        if not isinstance(haystack, Image.Image):
            import numpy as np

            if isinstance(haystack, np.ndarray):
                import cv2

                if len(haystack.shape) == 3 and haystack.shape[2] == 3:
                    rgb = cv2.cvtColor(haystack, cv2.COLOR_BGR2RGB)
                    haystack = Image.fromarray(rgb)
                else:
                    haystack = Image.fromarray(haystack)
            else:
                logger.debug("InvariantMatchBackend: unsupported haystack type")
                return []

        matcher = self._get_matcher()
        confidence = config.get("min_confidence", 0.8)
        scales = config.get("invariant_scales")
        rotations = config.get("invariant_rotations")

        match = matcher.find_template_invariant(
            haystack=haystack,
            needle=needle_image,
            scales=scales,
            rotations=rotations,
            confidence=confidence,
        )

        if match is None:
            return []

        result = DetectionResult(
            x=match.x,
            y=match.y,
            width=match.width,
            height=match.height,
            confidence=match.confidence,
            backend_name=self.name,
        )

        # Apply color-difference filter if configured
        color_ref = config.get("color_reference")
        if color_ref is not None:
            import cv2
            import numpy as np

            from ..utils.color_filter import ColorDifferenceFilter

            color_tolerance = config.get("color_tolerance", 50.0)
            color_filter = ColorDifferenceFilter(
                reference_color=color_ref,
                tolerance=color_tolerance,
            )

            # Convert haystack to BGR for color analysis
            haystack_bgr = cv2.cvtColor(np.array(haystack), cv2.COLOR_RGB2BGR)
            filtered = color_filter.filter_results([result], haystack_bgr)
            return filtered

        return [result]

    def supports(self, needle_type: str) -> bool:
        """Only handles template-type needles."""
        return needle_type == "template"

    def estimated_cost_ms(self) -> float:
        """~120ms for scale-only (7 scales), up to ~600ms with rotations."""
        return 120.0

    @property
    def name(self) -> str:
        return "invariant_template"
