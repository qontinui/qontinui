"""Grounding VLM detection backend.

Sends screenshots to the fine-tuned qontinui-grounding VLM model
(LoRA-merged UI-TARS-1.5-7B) via an OpenAI-compatible API. The model
was trained on isolated component renders for precise element localization
and responds with ``<point>x y</point>`` coordinates (normalized 0-1000).

Estimated latency: 500–1500ms — more expensive than template matching
but cheaper than the generic VisionLLMBackend (~2000ms).

This backend is a thin wrapper around :class:`qontinui.vga.client.VgaClient`;
both callers share the same encode / request / parse code path. The
public API (class name, constructor args, method names) is intentionally
unchanged from prior versions so existing callers keep working.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from ...vga.client import VgaClient, VgaClientError
from .base import DetectionBackend, DetectionResult

logger = logging.getLogger(__name__)


class GroundingVLMBackend(DetectionBackend):
    """Detection backend using the fine-tuned qontinui-grounding model.

    Queries the model via the OpenAI-compatible ``/v1/chat/completions``
    endpoint (served through llama-swap). The model is expected to return
    a ``<point>x y</point>`` tag where x and y are coordinates normalized
    to a 0–1000 range.

    The predicted point is converted to pixel coordinates using the
    haystack dimensions, then wrapped in a ``box_size`` px bounding box.

    Args:
        model: Model name to request. Defaults to
            ``"qontinui-grounding-v1"`` for backwards compatibility; new
            callers should use ``"qontinui-grounding-v5"``.
        api_base: Base URL of the OpenAI-compatible API. Defaults to
            ``"http://localhost:8100/v1"`` (host-side llama-swap port).
        timeout: HTTP request timeout in seconds. Defaults to 30.
        box_size: Side length in pixels of the synthetic bounding box
            centered on the predicted point. Defaults to 40.
    """

    _SUPPORTED_NEEDLE_TYPES = frozenset({"template", "text", "description", "label"})

    def __init__(
        self,
        model: str = "qontinui-grounding-v1",
        api_base: str = "http://localhost:8100/v1",
        timeout: float = 30.0,
        box_size: int = 40,
    ) -> None:
        self._client = VgaClient(api_base=api_base, model=model, timeout=timeout)
        self._box_half = box_size // 2

    # ------------------------------------------------------------------
    # DetectionBackend interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "grounding_vlm"

    def estimated_cost_ms(self) -> float:
        # Sits between OmniParser (~1500ms) and VisionLLM (~2000ms).
        return 1000.0

    def supports(self, needle_type: str) -> bool:
        return needle_type in self._SUPPORTED_NEEDLE_TYPES

    def is_available(self) -> bool:
        """Return True — availability is checked lazily on first call."""
        return True

    def find(
        self, needle: Any, haystack: Any, config: dict[str, Any]
    ) -> list[DetectionResult]:
        """Locate an element by sending the screenshot to the grounding model.

        Args:
            needle: Element description (str) or a Pattern-like object with
                a ``name`` or ``description`` attribute.
            haystack: Screenshot as PIL Image, numpy array, or raw PNG bytes.
            config: Standard detection config dict (``min_confidence``, etc.).

        Returns:
            A single-element list on success, empty list on failure.
        """
        description = self._extract_description(needle)
        if not description:
            logger.debug("GroundingVLMBackend: no description extractable from needle")
            return []

        try:
            result = self._client.ground(haystack, description)
        except VgaClientError:
            logger.debug("GroundingVLMBackend: client error", exc_info=True)
            return []

        if result.confidence <= 0.0:
            # Response had no <point> or explicit <none/> — surface a zero-
            # confidence placeholder so callers can distinguish "tried and
            # failed" from "skipped".
            return [
                DetectionResult(
                    x=0,
                    y=0,
                    width=0,
                    height=0,
                    confidence=0.0,
                    backend_name=self.name,
                    metadata={"raw_response": result.raw_response},
                )
            ]

        px, py = result.x, result.y
        img_width, img_height = result.image_width, result.image_height

        x = max(0, px - self._box_half)
        y = max(0, py - self._box_half)
        w = self._box_half * 2
        h = self._box_half * 2

        # Clip to image bounds if known
        if img_width > 0:
            w = min(w, img_width - x)
        if img_height > 0:
            h = min(h, img_height - y)

        return [
            DetectionResult(
                x=x,
                y=y,
                width=w,
                height=h,
                confidence=result.confidence,
                backend_name=self.name,
                label=description,
                metadata={
                    "norm_x": result.norm_x,
                    "norm_y": result.norm_y,
                    "raw_response": result.raw_response,
                },
            )
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_description(needle: Any) -> str | None:
        """Pull a text description from the needle."""
        if isinstance(needle, str):
            return needle.strip() or None

        for attr in ("description", "name", "label"):
            val = getattr(needle, attr, None)
            if val and isinstance(val, str):
                return cast(str, val.strip())

        return None
