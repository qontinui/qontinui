"""Grounding VLM detection backend.

Sends screenshots to the fine-tuned qontinui-grounding-v1 model
(LoRA-merged UI-TARS-1.5-7B) via an OpenAI-compatible API. The model
was trained on isolated component renders for precise element localization
and responds with ``<point>x y</point>`` coordinates (normalized 0-1000).

Estimated latency: 500–1500ms — more expensive than template matching
but cheaper than the generic VisionLLMBackend (~2000ms).
"""

from __future__ import annotations

import base64
import io
import logging
import re
from typing import Any

from .base import DetectionBackend, DetectionResult

logger = logging.getLogger(__name__)

# Matches <point>x y</point> where x and y are integers or floats.
_POINT_RE = re.compile(r"<point>\s*([0-9.]+)\s+([0-9.]+)\s*</point>", re.IGNORECASE)

# Default bounding-box half-size (pixels) centered on the predicted point.
_DEFAULT_BOX_HALF = 20  # → 40×40 px box


class GroundingVLMBackend(DetectionBackend):
    """Detection backend using the fine-tuned qontinui-grounding-v1 model.

    Queries the model via the OpenAI-compatible ``/v1/chat/completions``
    endpoint (served through llama-swap). The model is expected to return
    a ``<point>x y</point>`` tag where x and y are coordinates normalized
    to a 0–1000 range.

    The predicted point is converted to pixel coordinates using the
    haystack dimensions, then wrapped in a 40×40 px bounding box.

    Args:
        model: Model name to request. Defaults to ``"qontinui-grounding-v1"``.
        api_base: Base URL of the OpenAI-compatible API. Defaults to
            ``"http://localhost:5800/v1"`` (llama-swap default port).
        timeout: HTTP request timeout in seconds. Defaults to 30.
        box_size: Side length in pixels of the synthetic bounding box
            centered on the predicted point. Defaults to 40.
    """

    _SUPPORTED_NEEDLE_TYPES = frozenset(
        {"template", "text", "description", "label"}
    )

    def __init__(
        self,
        model: str = "qontinui-grounding-v1",
        api_base: str = "http://localhost:5800/v1",
        timeout: float = 30.0,
        box_size: int = 40,
    ) -> None:
        self._model = model
        self._api_base = api_base.rstrip("/")
        self._timeout = timeout
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

        img_b64, img_width, img_height = self._encode_image(haystack)
        if img_b64 is None:
            logger.debug("GroundingVLMBackend: could not encode haystack image")
            return []

        raw_text = self._query_model(description, img_b64)
        if raw_text is None:
            return []

        point = self._parse_point(raw_text)
        if point is None:
            logger.debug(
                "GroundingVLMBackend: no <point> tag in response: %.120s", raw_text
            )
            return [
                DetectionResult(
                    x=0,
                    y=0,
                    width=0,
                    height=0,
                    confidence=0.0,
                    backend_name=self.name,
                    metadata={"raw_response": raw_text[:256]},
                )
            ]

        norm_x, norm_y = point  # values in [0, 1000]
        px = int((norm_x / 1000.0) * img_width) if img_width else int(norm_x)
        py = int((norm_y / 1000.0) * img_height) if img_height else int(norm_y)

        x = max(0, px - self._box_half)
        y = max(0, py - self._box_half)
        w = self._box_half * 2
        h = self._box_half * 2

        # Clip to image bounds if known
        if img_width > 0:
            w = min(w, img_width - x)
        if img_height > 0:
            h = min(h, img_height - y)

        confidence = self._estimate_confidence(raw_text)

        return [
            DetectionResult(
                x=x,
                y=y,
                width=w,
                height=h,
                confidence=confidence,
                backend_name=self.name,
                label=description,
                metadata={
                    "norm_x": norm_x,
                    "norm_y": norm_y,
                    "raw_response": raw_text[:256],
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
                return val.strip()

        return None

    def _encode_image(
        self, image: Any
    ) -> tuple[str | None, int, int]:
        """Convert image to base-64 PNG string + (width, height).

        Returns (None, 0, 0) on failure.
        """
        try:
            from PIL import Image as PILImage

            pil_img: PILImage.Image | None = None

            if isinstance(image, bytes):
                pil_img = PILImage.open(io.BytesIO(image))
            elif isinstance(image, PILImage.Image):
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
                return None, 0, 0

            width, height = pil_img.size
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return b64, width, height

        except Exception:
            logger.exception("GroundingVLMBackend: image encoding failed")
            return None, 0, 0

    def _query_model(self, description: str, img_b64: str) -> str | None:
        """POST to the grounding model and return the raw text response."""
        import json
        import urllib.error
        import urllib.request

        prompt = (
            f"Locate the following element in the screenshot and output its "
            f"position as <point>x y</point> where x and y are integers "
            f"between 0 and 1000 (normalized coordinates).\n\nElement: {description}"
        )

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 64,
            "temperature": 0.0,
        }

        url = f"{self._api_base}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
        except urllib.error.URLError:
            logger.warning(
                "GroundingVLMBackend: could not reach model at %s", url, exc_info=True
            )
            return None
        except Exception:
            logger.exception("GroundingVLMBackend: model query failed")
            return None

    @staticmethod
    def _parse_point(text: str) -> tuple[float, float] | None:
        """Extract (x, y) from a ``<point>x y</point>`` tag."""
        m = _POINT_RE.search(text)
        if m is None:
            return None
        return float(m.group(1)), float(m.group(2))

    @staticmethod
    def _estimate_confidence(text: str) -> float:
        """Heuristic confidence based on response content.

        - Valid ``<point>`` tag present → 0.75 (model produced a grounded answer)
        - No tag → 0.0 (already handled upstream; kept for completeness)
        """
        if _POINT_RE.search(text):
            return 0.75
        return 0.0
