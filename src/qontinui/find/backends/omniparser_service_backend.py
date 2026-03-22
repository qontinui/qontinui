"""OmniParser service backend — connects to a remote OmniParser HTTP endpoint.

For users without a local GPU. The remote endpoint can be a self-hosted
Docker container, HuggingFace Space, or any server running OmniParser's
server mode.

Expected API: POST /parse with multipart image, returns JSON with detected
elements (bounding boxes + labels).
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
from PIL import Image

from .base import DetectionBackend, DetectionResult
from .omniparser_config import OmniParserSettings

logger = logging.getLogger(__name__)


class OmniParserServiceBackend(DetectionBackend):
    """Detection backend that delegates to a remote OmniParser HTTP service."""

    def __init__(self, settings: OmniParserSettings | None = None) -> None:
        self._settings = settings or OmniParserSettings()
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(
                base_url=self._settings.service_url,
                timeout=self._settings.service_timeout,
            )
        return self._client

    def find(self, needle: Any, haystack: Any, config: dict[str, Any]) -> list[DetectionResult]:
        if not isinstance(haystack, np.ndarray):
            logger.warning("OmniParserServiceBackend expects numpy array haystack")
            return []

        # Encode screenshot as PNG bytes
        if len(haystack.shape) == 3 and haystack.shape[2] == 3:
            pil_img = Image.fromarray(haystack[..., ::-1])  # BGR -> RGB
        else:
            pil_img = Image.fromarray(haystack)

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # Call remote service
        client = self._ensure_client()
        try:
            response = client.post(
                "/parse",
                files={"image": ("screenshot.png", img_bytes, "image/png")},
                data={
                    "iou_threshold": str(config.get("iou_threshold", self._settings.iou_threshold)),
                    "confidence_threshold": str(
                        config.get(
                            "confidence_threshold",
                            self._settings.confidence_threshold,
                        )
                    ),
                },
            )
            response.raise_for_status()
            data = response.json()
        except Exception:
            logger.warning("OmniParser service request failed", exc_info=True)
            return []

        # Parse response into DetectionResults
        elements = self._parse_response(data)

        # If needle is a description, filter by semantic matching
        needle_type = config.get("needle_type", "template")
        if needle_type in ("description", "semantic", "text") and isinstance(needle, str):
            return self._filter_by_description(needle, elements, config)

        return elements

    def _parse_response(self, data: dict[str, Any]) -> list[DetectionResult]:
        """Parse the JSON response from the OmniParser service."""
        results: list[DetectionResult] = []

        # Support both flat list and nested structure
        items = data.get("elements", data.get("results", []))
        if isinstance(data, list):
            items = data

        for item in items:
            try:
                # Support multiple bbox formats
                if "bbox" in item:
                    bbox = item["bbox"]
                    if len(bbox) == 4:
                        # Could be [x1, y1, x2, y2] or [x, y, w, h]
                        x1, y1, v3, v4 = bbox
                        # Heuristic: if v3 > x1 and v4 > y1, it's xyxy format
                        if v3 > x1 and v4 > y1 and v3 > 1 and v4 > 1:
                            w, h = v3 - x1, v4 - y1
                        else:
                            w, h = v3, v4
                        x, y = int(x1), int(y1)
                        w, h = int(w), int(h)
                elif all(k in item for k in ("x", "y", "width", "height")):
                    x = int(item["x"])
                    y = int(item["y"])
                    w = int(item["width"])
                    h = int(item["height"])
                else:
                    continue

                results.append(
                    DetectionResult(
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                        confidence=float(item.get("confidence", 0.5)),
                        backend_name=self.name,
                        label=item.get("label") or item.get("caption"),
                        metadata={
                            "element_type": item.get("type", "interactive_element"),
                            "source": "omniparser_service",
                        },
                    )
                )
            except (KeyError, ValueError, TypeError):
                logger.debug("Skipping malformed element in service response")
                continue

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def _filter_by_description(
        self,
        description: str,
        elements: list[DetectionResult],
        config: dict[str, Any],
    ) -> list[DetectionResult]:
        """Filter results by matching against a text description."""
        from qontinui.find.semantic_matcher import match_element_by_description

        labels = [e.label or "" for e in elements]
        types = [e.metadata.get("element_type") for e in elements]
        min_sim = config.get("min_similarity", 0.4)

        matches = match_element_by_description(
            description, labels, element_types=types, min_similarity=min_sim
        )

        filtered: list[DetectionResult] = []
        for m in matches:
            elem = elements[m.element_index]
            filtered.append(
                DetectionResult(
                    x=elem.x,
                    y=elem.y,
                    width=elem.width,
                    height=elem.height,
                    confidence=m.score * elem.confidence,
                    backend_name=self.name,
                    label=elem.label,
                    metadata={
                        **elem.metadata,
                        "match_type": m.match_type,
                        "semantic_score": m.score,
                    },
                )
            )

        filtered.sort(key=lambda r: r.confidence, reverse=True)
        return filtered

    def supports(self, needle_type: str) -> bool:
        return needle_type in ("template", "text", "description", "semantic")

    def estimated_cost_ms(self) -> float:
        return 2000.0  # Network round trip adds overhead vs local

    @property
    def name(self) -> str:
        return "omniparser_service"

    def is_available(self) -> bool:
        if not self._settings.enabled or self._settings.provider != "service":
            return False
        # Quick health check
        try:
            client = self._ensure_client()
            resp = client.get("/health", timeout=2.0)
            return bool(resp.status_code == 200)
        except Exception:
            return False
