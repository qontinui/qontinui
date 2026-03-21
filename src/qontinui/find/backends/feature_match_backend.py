"""Feature-based detection backend.

Wraps the HAL-level OpenCVMatcher feature detection/matching as a
DetectionBackend. More robust to scale and rotation than template
matching but slower (~100ms).
"""

import logging
from typing import Any

from .base import DetectionBackend, DetectionResult

logger = logging.getLogger(__name__)


class FeatureMatchBackend(DetectionBackend):
    """Detection backend using ORB/AKAZE/SIFT feature matching.

    Wraps ``OpenCVMatcher`` from ``hal.implementations.opencv_matcher``.
    Uses feature detection and descriptor matching which is more robust
    to scale/rotation changes than template matching.

    Args:
        matcher: An existing ``OpenCVMatcher`` instance. If None, a default
                 one is created on first use.
        method: Feature detection method (``orb``, ``akaze``, ``sift``).
        min_good_matches: Minimum number of good feature matches to
                          consider a detection successful.
    """

    def __init__(
        self,
        matcher: Any | None = None,
        method: str = "orb",
        min_good_matches: int = 10,
    ) -> None:
        self._matcher = matcher
        self._method = method
        self._min_good_matches = min_good_matches

    def _get_matcher(self) -> Any:
        if self._matcher is None:
            from ...hal.implementations.opencv_matcher import OpenCVMatcher

            self._matcher = OpenCVMatcher()
        return self._matcher

    def find(self, needle: Any, haystack: Any, config: dict[str, Any]) -> list[DetectionResult]:
        """Find needle using feature matching.

        Detects features in both needle and haystack, matches descriptors,
        and computes the bounding region of matched features in the haystack.

        Args:
            needle: PIL Image of the element to find.
            haystack: PIL Image of the screenshot.
            config: Keys used: ``min_confidence``.

        Returns:
            List with at most one DetectionResult (the matched region).
        """
        # Convert needle to PIL Image if it's a Pattern
        needle_image = self._to_pil(needle)
        haystack_image = self._to_pil(haystack)
        if needle_image is None or haystack_image is None:
            return []

        matcher = self._get_matcher()
        min_confidence = config.get("min_confidence", 0.8)

        try:
            needle_features = matcher.find_features(needle_image, method=self._method)
            haystack_features = matcher.find_features(haystack_image, method=self._method)

            if not needle_features or not haystack_features:
                return []

            matched_pairs = matcher.match_features(
                needle_features, haystack_features, threshold=0.7
            )

            if len(matched_pairs) < self._min_good_matches:
                return []

            # Compute bounding box from matched haystack features
            hx = [pair[1].x for pair in matched_pairs]
            hy = [pair[1].y for pair in matched_pairs]
            min_x, max_x = int(min(hx)), int(max(hx))
            min_y, max_y = int(min(hy)), int(max(hy))

            # Confidence based on ratio of good matches to total needle features
            confidence = min(1.0, len(matched_pairs) / max(len(needle_features), 1))

            if confidence < min_confidence:
                return []

            width = max(max_x - min_x, 1)
            height = max(max_y - min_y, 1)

            return [
                DetectionResult(
                    x=min_x,
                    y=min_y,
                    width=width,
                    height=height,
                    confidence=confidence,
                    backend_name=self.name,
                    metadata={"matched_features": len(matched_pairs)},
                )
            ]

        except Exception:
            logger.exception("FeatureMatchBackend: matching failed")
            return []

    def _to_pil(self, image: Any) -> Any:
        """Convert image to PIL Image if possible."""
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            return image

        # Handle Pattern objects
        if hasattr(image, "pixel_data") and image.pixel_data is not None:
            import cv2
            import numpy as np

            data = image.pixel_data
            if isinstance(data, np.ndarray):
                if len(data.shape) == 3 and data.shape[2] >= 3:
                    rgb = cv2.cvtColor(data[:, :, :3], cv2.COLOR_BGR2RGB)
                    return PILImage.fromarray(rgb)
                return PILImage.fromarray(data)

        # Handle numpy arrays directly
        import numpy as np

        if isinstance(image, np.ndarray):
            import cv2

            if len(image.shape) == 3 and image.shape[2] >= 3:
                rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
                return PILImage.fromarray(rgb)
            return PILImage.fromarray(image)

        return None

    def supports(self, needle_type: str) -> bool:
        return needle_type == "template"

    def estimated_cost_ms(self) -> float:
        return 100.0

    @property
    def name(self) -> str:
        return "feature"
