"""Vision LLM detection backend.

Wraps the VisionLLMClient from the healing module as a DetectionBackend.
Most expensive backend (~2000ms) — used as last resort when cheaper
methods fail.
"""

import logging
from typing import Any

from .base import DetectionBackend, DetectionResult

logger = logging.getLogger(__name__)


class VisionLLMBackend(DetectionBackend):
    """Detection backend using a vision-capable LLM.

    Wraps ``VisionLLMClient`` from ``healing.llm_client``.
    Sends a screenshot to a vision LLM with an element description
    and parses coordinates from the response. Can find virtually
    anything but is slow and expensive.

    Args:
        llm_client: An existing ``VisionLLMClient`` instance.
    """

    def __init__(self, llm_client: Any) -> None:
        self._client = llm_client

    def find(self, needle: Any, haystack: Any, config: dict[str, Any]) -> list[DetectionResult]:
        """Find element by asking a vision LLM.

        Args:
            needle: Description string or Pattern with a name/description.
            haystack: Screenshot as PIL Image, numpy array, or bytes.
            config: Keys used: ``min_confidence``, ``action_type``.

        Returns:
            List with at most one DetectionResult.
        """
        description = self._extract_description(needle)
        if not description:
            return []

        screenshot_bytes = self._to_png_bytes(haystack)
        if screenshot_bytes is None:
            return []

        try:
            from ...healing.healing_types import HealingContext

            context = HealingContext(
                original_description=description,
                action_type=config.get("action_type"),
            )

            location = self._client.find_element(screenshot_bytes, context)
            if location is None:
                return []

            min_confidence = config.get("min_confidence", 0.8)
            if location.confidence < min_confidence:
                return []

            # VLM returns center coords; estimate a small bounding box
            region = location.region
            if region:
                x, y, w, h = region
            else:
                # Default to a small region around the point
                x, y, w, h = location.x - 15, location.y - 15, 30, 30

            return [
                DetectionResult(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    confidence=location.confidence,
                    backend_name=self.name,
                    label=location.description,
                    metadata={"vlm_description": description},
                )
            ]

        except Exception:
            logger.exception("VisionLLMBackend: LLM detection failed")
            return []

    def _extract_description(self, needle: Any) -> str | None:
        """Extract a text description from the needle."""
        if isinstance(needle, str):
            return needle

        # Pattern objects have a name
        name = getattr(needle, "name", None)
        if name:
            return str(name)

        return None

    def _to_png_bytes(self, image: Any) -> bytes | None:
        """Convert image to PNG bytes for the LLM API."""
        if isinstance(image, bytes):
            return image

        import io

        from PIL import Image as PILImage

        try:
            pil_img: PILImage.Image | None = None

            if isinstance(image, PILImage.Image):
                pil_img = image
            else:
                import numpy as np

                if isinstance(image, np.ndarray):
                    import cv2

                    if len(image.shape) == 3 and image.shape[2] >= 3:
                        rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
                        pil_img = PILImage.fromarray(rgb)
                    else:
                        pil_img = PILImage.fromarray(image)

            if pil_img is None:
                return None

            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            return buf.getvalue()

        except Exception:
            logger.exception("VisionLLMBackend: image conversion failed")
            return None

    def supports(self, needle_type: str) -> bool:
        return needle_type in ("template", "text", "description")

    def estimated_cost_ms(self) -> float:
        return 2000.0

    @property
    def name(self) -> str:
        return "vision_llm"

    def is_available(self) -> bool:
        return hasattr(self._client, "is_available") and self._client.is_available
