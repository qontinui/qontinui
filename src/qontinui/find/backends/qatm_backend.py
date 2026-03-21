"""QATM (Quality-Aware Template Matching) detection backend.

Wraps QATMMatcher as a DetectionBackend for the CascadeDetector fallback
chain. Uses VGG-19 deep features with quality-aware scoring to reject
false positives. Positioned between FeatureMatchBackend (~100ms) and
OmniParserBackend (~1500ms) at ~200ms on GPU.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import DetectionBackend, DetectionResult
from .qatm_config import QATMSettings

logger = logging.getLogger(__name__)

_torch_checked: bool | None = None


def _torch_available() -> bool:
    """Check if PyTorch and torchvision are importable (cached)."""
    global _torch_checked
    if _torch_checked is None:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401

            _torch_checked = True
        except ImportError:
            _torch_checked = False
    return _torch_checked


class QATMBackend(DetectionBackend):
    """Detection backend using QATM quality-aware deep template matching.

    Uses VGG-19 feature extraction with quality-aware scoring to find
    template images in screenshots. More robust than pixel-exact template
    matching, with built-in false-positive rejection.

    Enable via ``QONTINUI_QATM_ENABLED=true``.

    Args:
        settings: QATM configuration. If None, reads from environment.
    """

    def __init__(self, settings: QATMSettings | None = None) -> None:
        self._settings = settings or QATMSettings()
        self._matcher = None

    def _get_matcher(self) -> Any:
        if self._matcher is None:
            from .qatm_matcher import QATMMatcher

            self._matcher = QATMMatcher(self._settings)
        return self._matcher

    def find(self, needle: Any, haystack: Any, config: dict[str, Any]) -> list[DetectionResult]:
        """Find needle template in haystack screenshot using QATM.

        Args:
            needle: A ``Pattern`` object with ``pixel_data``, or numpy array.
            haystack: Screenshot as numpy array (BGR).
            config: Keys used: ``min_confidence``, ``find_all``.

        Returns:
            List of DetectionResult sorted by confidence.
        """
        # Extract template image from Pattern or use directly
        template = self._to_bgr(needle)
        screenshot = self._to_bgr(haystack)

        if template is None or screenshot is None:
            logger.debug("QATMBackend: could not convert needle/haystack to BGR")
            return []

        matcher = self._get_matcher()
        min_confidence = config.get("min_confidence", self._settings.confidence_threshold)
        find_all = config.get("find_all", False)

        # Check if model should be unloaded due to inactivity
        if matcher.should_unload():
            matcher.unload()

        try:
            matches = matcher.find(
                template=template,
                screenshot=screenshot,
                min_confidence=min_confidence,
                find_all=find_all,
            )
        except Exception:
            logger.exception("QATMBackend: matching failed")
            return []

        results: list[DetectionResult] = []
        for m in matches:
            results.append(
                DetectionResult(
                    x=m.x,
                    y=m.y,
                    width=m.width,
                    height=m.height,
                    confidence=m.confidence,
                    backend_name=self.name,
                    metadata={"quality_aware": True},
                )
            )

        return results

    @staticmethod
    def _to_bgr(image: Any) -> np.ndarray | None:
        """Convert image to BGR numpy array."""
        if isinstance(image, np.ndarray):
            return image

        # Handle Pattern objects
        if hasattr(image, "pixel_data") and image.pixel_data is not None:
            data = image.pixel_data
            if isinstance(data, np.ndarray):
                return data
            return None

        # Handle PIL Images
        try:
            from PIL import Image as PILImage

            if isinstance(image, PILImage.Image):
                import cv2

                rgb = np.array(image)
                if len(rgb.shape) == 3 and rgb.shape[2] >= 3:
                    return cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
                return rgb
        except ImportError:
            pass

        return None

    def supports(self, needle_type: str) -> bool:
        return needle_type == "template"

    def estimated_cost_ms(self) -> float:
        return 200.0

    def is_available(self) -> bool:
        return self._settings.enabled and _torch_available()

    @property
    def name(self) -> str:
        return "qatm"
