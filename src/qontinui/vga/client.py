"""Thin OpenAI-format client to the ``qontinui-grounding-v5`` model.

Extracted from :mod:`qontinui.find.backends.grounding_vlm_backend` so
both callers share one network + parsing code path. The older backend
now wraps this client rather than duplicating the request/parse logic.

Responsibilities:

- Encode PIL / numpy / bytes images to base-64 PNG.
- POST OpenAI ``chat/completions`` payloads with an ``image_url`` + text
  message pair.
- Parse ``<point>x y</point>`` tags out of the response (normalized 0-1000)
  and convert to pixel coordinates.

Failure modes are surfaced as :class:`VgaClientError`. The caller decides
whether to retry, fall back, or log.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .prompts import GROUND_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

_POINT_RE = re.compile(r"<point>\s*([0-9.]+)\s+([0-9.]+)\s*</point>", re.IGNORECASE)
"""Matches ``<point>x y</point>`` where x and y are ints or floats."""

_NONE_RE = re.compile(r"<none\s*/?>", re.IGNORECASE)
"""Matches ``<none/>`` / ``<none>`` — used by proposal prompts when no
matching element is visible."""


class VgaClientError(RuntimeError):
    """Raised for network failures or unparseable responses from the VLM."""


@dataclass(frozen=True)
class GroundResult:
    """Single-point grounding result.

    Attributes:
        x: Pixel x coordinate (centered on the predicted point) in the
            original image's coordinate system.
        y: Pixel y coordinate.
        norm_x: Normalized x (0-1000) as returned by the model.
        norm_y: Normalized y (0-1000) as returned by the model.
        confidence: Heuristic confidence in [0, 1]. ``0.75`` if a valid
            ``<point>`` tag was parsed, ``0.0`` otherwise.
        raw_response: First 256 characters of the model's raw text output
            (useful for debugging proposal-style prompts).
        image_width: Width of the image the prediction was made against.
        image_height: Height of the image the prediction was made against.
    """

    x: int
    y: int
    norm_x: float
    norm_y: float
    confidence: float
    raw_response: str
    image_width: int
    image_height: int

    @property
    def point(self) -> tuple[int, int]:
        """Return ``(x, y)`` as a tuple."""
        return self.x, self.y


class VgaClient:
    """OpenAI-format client to llama-swap's ``qontinui-grounding-v5`` model.

    The client is deliberately small: one public ``ground`` method plus
    a small number of internal helpers. It never caches anything and is
    safe to instantiate per-call.

    Args:
        api_base: Base URL of the OpenAI-compatible API. Defaults to
            ``http://localhost:5800/v1`` (llama-swap default port).
        model: Model name to request. Defaults to
            ``qontinui-grounding-v5``.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_base: str = "http://localhost:5800/v1",
        model: str = "qontinui-grounding-v5",
        timeout: float = 30.0,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._model = model
        self._timeout = timeout

    @property
    def api_base(self) -> str:
        return self._api_base

    @property
    def model(self) -> str:
        return self._model

    @property
    def timeout(self) -> float:
        return self._timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ground(
        self,
        image: Any,
        prompt: str,
        *,
        prompt_template: str = GROUND_PROMPT_TEMPLATE,
        max_tokens: int = 64,
    ) -> GroundResult:
        """Ground a single element described by ``prompt``.

        Args:
            image: PIL Image, numpy ndarray, or raw PNG/JPEG bytes.
            prompt: Natural-language element description. Substituted into
                ``prompt_template`` as ``{description}``.
            prompt_template: Override the default
                :data:`qontinui.vga.prompts.GROUND_PROMPT_TEMPLATE`. Useful
                for proposal prompts that use a different format (e.g.
                ``PROPOSE_CATEGORY_PROMPT_TEMPLATE``).
            max_tokens: Cap on model output tokens. The default of 64 is
                plenty for a single ``<point>`` tag.

        Returns:
            :class:`GroundResult` with pixel + normalized coords.

        Raises:
            VgaClientError: If image encoding fails, the HTTP request
                fails, or the response has no ``<point>`` tag and no
                ``<none/>`` sentinel.
        """
        img_b64, img_width, img_height = self._encode_image(image)
        if img_b64 is None:
            raise VgaClientError("Could not encode haystack image")

        full_prompt = prompt_template.format(description=prompt, category=prompt)
        raw_text = self._query_model(full_prompt, img_b64, max_tokens=max_tokens)

        return self._parse_response(raw_text, img_width, img_height)

    # ------------------------------------------------------------------
    # Internal helpers (re-used by GroundingVLMBackend)
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(image: Any) -> tuple[str | None, int, int]:
        """Convert image to base-64 PNG + ``(width, height)``.

        Returns ``(None, 0, 0)`` on failure (logged at exception level).
        """
        try:
            from PIL import Image as PILImage

            pil_img: PILImage.Image | None = None

            if isinstance(image, bytes):
                pil_img = PILImage.open(io.BytesIO(image))
            elif isinstance(image, PILImage.Image):
                pil_img = image
            else:
                try:
                    import numpy as np
                except ImportError:  # pragma: no cover — numpy is a hard dep
                    return None, 0, 0

                if isinstance(image, np.ndarray):
                    try:
                        import cv2

                        if len(image.shape) == 3 and image.shape[2] >= 3:
                            rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
                            pil_img = PILImage.fromarray(rgb)
                        else:
                            pil_img = PILImage.fromarray(image)
                    except ImportError:
                        pil_img = PILImage.fromarray(image)

            if pil_img is None:
                return None, 0, 0

            width, height = pil_img.size
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return b64, width, height

        except Exception:
            logger.exception("VgaClient: image encoding failed")
            return None, 0, 0

    def _query_model(
        self, prompt: str, img_b64: str, *, max_tokens: int = 64
    ) -> str:
        """POST to ``/v1/chat/completions`` and return the raw text content.

        Raises:
            VgaClientError: On URL errors, non-200 responses, or JSON
                structure mismatches.
        """
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
            "max_tokens": max_tokens,
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
                return str(body["choices"][0]["message"]["content"])
        except urllib.error.URLError as e:
            raise VgaClientError(f"Could not reach model at {url}: {e}") from e
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise VgaClientError(f"Unexpected response shape from {url}: {e}") from e

    @staticmethod
    def _parse_response(
        raw_text: str, img_width: int, img_height: int
    ) -> GroundResult:
        """Parse a ``<point>`` tag out of ``raw_text``.

        If the response contains ``<none/>``, return a zero-confidence
        result with ``x=y=0``.

        Raises:
            VgaClientError: If neither a ``<point>`` tag nor ``<none/>``
                is present — the caller can decide whether that is a
                retry-worthy condition.
        """
        snippet = raw_text[:256]

        if _NONE_RE.search(raw_text):
            return GroundResult(
                x=0,
                y=0,
                norm_x=0.0,
                norm_y=0.0,
                confidence=0.0,
                raw_response=snippet,
                image_width=img_width,
                image_height=img_height,
            )

        m = _POINT_RE.search(raw_text)
        if m is None:
            raise VgaClientError(
                f"No <point> tag in response: {snippet!r}"
            )

        norm_x = float(m.group(1))
        norm_y = float(m.group(2))

        px = int((norm_x / 1000.0) * img_width) if img_width else int(norm_x)
        py = int((norm_y / 1000.0) * img_height) if img_height else int(norm_y)

        return GroundResult(
            x=px,
            y=py,
            norm_x=norm_x,
            norm_y=norm_y,
            confidence=0.75,
            raw_response=snippet,
            image_width=img_width,
            image_height=img_height,
        )
