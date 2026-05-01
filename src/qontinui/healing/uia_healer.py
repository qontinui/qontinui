"""UIA-specific self-healing for failed accessibility element lookups.

Provides graduated healing strategies for when a UIA element can no longer
be found by its original selector (automation_id, name, role). Inspired by
pywinassistant's self-healing selector approach.

Strategies (ordered by cost):
1. Automation ID match — most stable UIA identifier
2. Semantic match — same role + similar name
3. Structural match — same role + position zone
4. Fuzzy name match — role + fuzzy name similarity
5. LLM recovery — feed UIA tree to LLM (if enabled)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from ..hal.implementations.accessibility.uia_label_utils import infer_spatial_labels
from .healing_types import ElementLocation, HealingContext, HealingResult, HealingStrategy

if TYPE_CHECKING:
    from qontinui_schemas.accessibility import AccessibilityNode

    from ..hal.interfaces.accessibility_capture import IAccessibilityCapture

logger = logging.getLogger(__name__)


@dataclass
class UIAElementFingerprint:
    """Fingerprint of a UIA element for healing lookups.

    Captures enough identity information to re-find an element
    when its ref has become stale.
    """

    automation_id: str | None = None
    role: str | None = None
    name: str | None = None
    class_name: str | None = None
    bounds: tuple[int, int, int, int] | None = None  # x, y, w, h

    @classmethod
    def from_node(cls, node: AccessibilityNode) -> UIAElementFingerprint:
        """Create fingerprint from an AccessibilityNode."""
        bounds = None
        if node.bounds is not None:
            if hasattr(node.bounds, "x"):
                bounds = (
                    node.bounds.x,
                    node.bounds.y,
                    node.bounds.width,
                    node.bounds.height,
                )
            elif isinstance(node.bounds, list | tuple) and len(node.bounds) >= 4:
                bounds = tuple(int(v) for v in node.bounds[:4])

        return cls(
            automation_id=node.automation_id,
            role=getattr(node.role, "value", str(node.role)) if node.role else None,
            name=node.name,
            class_name=node.class_name,
            bounds=bounds,
        )

    @classmethod
    def from_context(cls, context: HealingContext) -> UIAElementFingerprint:
        """Create fingerprint from a HealingContext's additional_context."""
        ctx = context.additional_context
        return cls(
            automation_id=ctx.get("automation_id"),
            role=ctx.get("role"),
            name=ctx.get("name") or context.original_description,
            class_name=ctx.get("class_name"),
            bounds=ctx.get("bounds"),
        )


def _position_zone(x: int, y: int) -> str:
    """Classify a position into a zone (header/footer/sidebar/main)."""
    if y < 80:
        return "header"
    if y > 800:
        return "footer"
    if x < 250:
        return "sidebar-left"
    if x > 1200:
        return "sidebar-right"
    return "main"


def _flatten_interactive(node: AccessibilityNode) -> list[AccessibilityNode]:
    """Flatten tree to list of interactive nodes."""
    result: list[AccessibilityNode] = []
    if node.is_interactive:
        result.append(node)
    for child in node.children:
        result.extend(_flatten_interactive(child))
    return result


def _flatten_all(node: AccessibilityNode) -> list[AccessibilityNode]:
    """Flatten tree to list of ALL nodes (interactive and non-interactive)."""
    result: list[AccessibilityNode] = [node]
    for child in node.children:
        result.extend(_flatten_all(child))
    return result


class UIAHealer:
    """UIA-specific element healer.

    Attempts to re-find a lost UIA element using graduated strategies.
    Does not require screenshots — works purely with the accessibility tree.

    Args:
        capture: IAccessibilityCapture for querying UIA tree.
        fuzzy_threshold: Minimum SequenceMatcher ratio for fuzzy matches.
        max_attempts: Maximum strategies to try before giving up.
    """

    def __init__(
        self,
        capture: IAccessibilityCapture,
        *,
        fuzzy_threshold: float = 0.7,
        max_attempts: int = 5,
    ) -> None:
        self._capture = capture
        self._fuzzy_threshold = fuzzy_threshold
        self._max_attempts = max_attempts
        self._stats: dict[str, int] = {
            "total": 0,
            "success": 0,
            "by_automation_id": 0,
            "by_semantic": 0,
            "by_spatial_label": 0,
            "by_structural": 0,
            "by_fuzzy": 0,
        }

    def heal(self, context: HealingContext) -> HealingResult:
        """Attempt to heal a failed UIA element lookup.

        Args:
            context: Healing context. ``additional_context`` should contain
                the element's original fingerprint fields (automation_id,
                role, name, class_name, bounds).

        Returns:
            HealingResult with the re-found element location or failure.
        """
        start = time.perf_counter()
        self._stats["total"] += 1

        fingerprint = UIAElementFingerprint.from_context(context)
        attempts: list[tuple[HealingStrategy, str]] = []

        # Capture current tree
        snapshot = self._run_async(self._capture.capture_tree())
        if snapshot is None:
            return HealingResult(
                success=False,
                strategy=HealingStrategy.FAILED,
                message="Could not capture accessibility tree",
                attempts=attempts,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        nodes = _flatten_interactive(snapshot.root)
        if not nodes:
            return HealingResult(
                success=False,
                strategy=HealingStrategy.FAILED,
                message="No interactive elements in tree",
                attempts=attempts,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        # Collect all nodes (including non-interactive labels) for spatial
        # label inference. Interactive-only strategies still receive `nodes`.
        all_nodes = _flatten_all(snapshot.root)

        strategies = [
            ("by_automation_id", self._try_automation_id),
            ("by_semantic", self._try_semantic),
            ("by_spatial_label", self._try_spatial_label),
            ("by_structural", self._try_structural),
            ("by_fuzzy", self._try_fuzzy),
        ]

        for strategy_name, strategy_fn in strategies[: self._max_attempts]:
            if strategy_name == "by_spatial_label":
                node = strategy_fn(fingerprint, all_nodes)
            else:
                node = strategy_fn(fingerprint, nodes)
            if node is not None:
                location = self._node_to_location(node)
                if location is not None:
                    self._stats["success"] += 1
                    self._stats[strategy_name] += 1
                    elapsed = (time.perf_counter() - start) * 1000

                    logger.info(
                        "UIA healer: found element via %s at (%d, %d)",
                        strategy_name,
                        location.x,
                        location.y,
                    )

                    return HealingResult(
                        success=True,
                        strategy=HealingStrategy.UIA_SELECTOR,
                        location=location,
                        message=f"Re-found via {strategy_name}",
                        attempts=attempts,
                        duration_ms=elapsed,
                    )

            attempts.append(
                (HealingStrategy.UIA_SELECTOR, f"{strategy_name}: no match")
            )

        elapsed = (time.perf_counter() - start) * 1000
        return HealingResult(
            success=False,
            strategy=HealingStrategy.FAILED,
            message="All UIA healing strategies failed",
            attempts=attempts,
            duration_ms=elapsed,
        )

    def _try_automation_id(
        self,
        fp: UIAElementFingerprint,
        nodes: list[AccessibilityNode],
    ) -> AccessibilityNode | None:
        """Strategy 1: Match by automation_id (most stable)."""
        if not fp.automation_id:
            return None
        for node in nodes:
            if node.automation_id == fp.automation_id:
                return node
        return None

    def _try_semantic(
        self,
        fp: UIAElementFingerprint,
        nodes: list[AccessibilityNode],
    ) -> AccessibilityNode | None:
        """Strategy 2: Match by role + similar name."""
        if not fp.role or not fp.name:
            return None

        fp_name_lower = fp.name.lower()
        for node in nodes:
            node_role = getattr(node.role, "value", str(node.role))
            if node_role != fp.role:
                continue
            node_name = (node.name or "").lower()
            if node_name == fp_name_lower:
                return node
        return None

    def _try_spatial_label(
        self,
        fp: UIAElementFingerprint,
        nodes: list[AccessibilityNode],
    ) -> AccessibilityNode | None:
        """Strategy 2.5: Match unlabeled nodes whose inferred spatial label
        fuzzy-matches the fingerprint name.

        Applies when the fingerprint has a name but no node matched by name
        in the semantic strategy (i.e., the target control is unlabeled in the
        current tree, but a nearby text node still carries the right label).
        """
        if not fp.name:
            return None

        fp_name_lower = fp.name.lower()

        spatial_labels = infer_spatial_labels(nodes)
        if not spatial_labels:
            return None

        best_node: AccessibilityNode | None = None
        best_ratio = 0.0

        for node in nodes:
            inferred = spatial_labels.get(node.ref, "").strip()
            if not inferred:
                continue

            ratio = SequenceMatcher(None, fp_name_lower, inferred.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_node = node

        if best_ratio >= self._fuzzy_threshold:
            return best_node

        return None

    def _try_structural(
        self,
        fp: UIAElementFingerprint,
        nodes: list[AccessibilityNode],
    ) -> AccessibilityNode | None:
        """Strategy 3: Match by role + position zone."""
        if not fp.role or not fp.bounds:
            return None

        orig_zone = _position_zone(fp.bounds[0], fp.bounds[1])

        for node in nodes:
            node_role = getattr(node.role, "value", str(node.role))
            if node_role != fp.role:
                continue

            bounds = getattr(node, "bounds", None)
            if bounds is None:
                continue

            if hasattr(bounds, "x"):
                x, y = bounds.x, bounds.y
            elif isinstance(bounds, list | tuple) and len(bounds) >= 2:
                x, y = int(bounds[0]), int(bounds[1])
            else:
                continue

            if _position_zone(x, y) == orig_zone:
                return node

        return None

    def _try_fuzzy(
        self,
        fp: UIAElementFingerprint,
        nodes: list[AccessibilityNode],
    ) -> AccessibilityNode | None:
        """Strategy 4: Match by role + fuzzy name similarity."""
        if not fp.name:
            return None

        fp_name_lower = fp.name.lower()
        best_node: AccessibilityNode | None = None
        best_ratio = 0.0

        for node in nodes:
            # Optionally filter by role
            if fp.role:
                node_role = getattr(node.role, "value", str(node.role))
                if node_role != fp.role:
                    continue

            node_name = (node.name or "").lower()
            if not node_name:
                continue

            ratio = SequenceMatcher(None, fp_name_lower, node_name).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_node = node

        if best_ratio >= self._fuzzy_threshold:
            return best_node

        return None

    def _node_to_location(self, node: AccessibilityNode) -> ElementLocation | None:
        """Convert an AccessibilityNode to an ElementLocation."""
        bounds = getattr(node, "bounds", None)
        if bounds is None:
            return None

        if hasattr(bounds, "x"):
            x, y, w, h = bounds.x, bounds.y, bounds.width, bounds.height
        elif isinstance(bounds, list | tuple) and len(bounds) >= 4:
            x, y, w, h = int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])
        else:
            return None

        return ElementLocation(
            x=x + w // 2,
            y=y + h // 2,
            confidence=0.9,
            region=(x, y, w, h),
            description=f"UIA: {node.name or node.automation_id or node.ref}",
        )

    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine from sync context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result(timeout=10.0)
        else:
            return asyncio.run(coro)

    def get_stats(self) -> dict[str, int]:
        """Get healing statistics by strategy."""
        return dict(self._stats)
