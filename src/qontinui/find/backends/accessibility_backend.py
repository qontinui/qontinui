"""Accessibility tree detection backend.

Wraps the HAL-level IAccessibilityCapture as a DetectionBackend.
Very fast (~5ms) since it queries structured data rather than doing
vision processing.
"""

import asyncio
import logging
from typing import Any

from .base import DetectionBackend, DetectionResult

logger = logging.getLogger(__name__)


class AccessibilityBackend(DetectionBackend):
    """Detection backend using the accessibility tree.

    Wraps ``IAccessibilityCapture`` from ``hal.interfaces.accessibility_capture``.
    Queries the accessibility tree for elements matching by ID, role, or label.
    Fastest backend since no image processing is required.

    Args:
        capture: An existing ``IAccessibilityCapture`` implementation.
    """

    def __init__(self, capture: Any) -> None:
        self._capture = capture

    def find(self, needle: Any, haystack: Any, config: dict[str, Any]) -> list[DetectionResult]:
        """Find elements in the accessibility tree.

        Args:
            needle: Search value — an accessibility ID string, role name,
                    or label text depending on ``needle_type`` in config.
            haystack: Ignored (accessibility tree is queried directly).
            config: Keys used:
                - ``needle_type``: One of ``accessibility_id``, ``role``, ``label``.
                - ``min_confidence``: Minimum confidence threshold.

        Returns:
            List of DetectionResult for matching accessibility nodes.
        """
        if not isinstance(needle, str):
            return []

        needle_type = config.get("needle_type", "accessibility_id")

        try:
            # Build selector based on needle type
            from qontinui_schemas.accessibility import AccessibilitySelector

            selector_kwargs: dict[str, Any] = {}
            if needle_type == "accessibility_id":
                selector_kwargs["automation_id"] = needle
            elif needle_type == "role":
                selector_kwargs["role"] = needle
            elif needle_type == "label":
                selector_kwargs["name"] = needle
            else:
                return []

            selector = AccessibilitySelector(**selector_kwargs)

            # Run async find_nodes synchronously
            nodes = self._run_async(self._capture.find_nodes(selector))

            results: list[DetectionResult] = []
            for node in nodes:
                bounds = getattr(node, "bounds", None)
                if bounds is None:
                    continue

                # Handle both AccessibilityBounds objects and tuples
                if hasattr(bounds, "x"):
                    x, y, w, h = int(bounds.x), int(bounds.y), int(bounds.width), int(bounds.height)
                elif isinstance(bounds, list | tuple) and len(bounds) >= 4:
                    x, y, w, h = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])
                else:
                    continue

                node_name = getattr(node, "name", "") or ""
                results.append(
                    DetectionResult(
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                        confidence=1.0,  # Accessibility matches are exact
                        backend_name=self.name,
                        label=node_name,
                        metadata={
                            "role": getattr(node, "role", None),
                            "ref": getattr(node, "ref", None),
                        },
                    )
                )

            return results

        except Exception:
            logger.exception("AccessibilityBackend: query failed")
            return []

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine from sync context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an async context — create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result(timeout=5.0)
        else:
            return asyncio.run(coro)

    def supports(self, needle_type: str) -> bool:
        return needle_type in ("accessibility_id", "role", "label")

    def estimated_cost_ms(self) -> float:
        return 5.0

    @property
    def name(self) -> str:
        return "accessibility"

    def is_available(self) -> bool:
        return self._capture is not None and self._capture.is_connected()
