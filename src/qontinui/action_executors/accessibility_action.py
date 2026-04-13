"""Accessibility-aware action execution.

When the last FIND result came from an accessibility backend (contains a 'ref'
in metadata), this module can execute actions via UIA patterns instead of
generic mouse/keyboard, providing more reliable and control-type-aware
interactions.

The entry points are :func:`try_accessibility_click` and
:func:`try_accessibility_type`.  Both return an
:class:`AccessibilityActionResult` whose ``handled`` flag tells the caller
whether to skip the normal mouse/keyboard path.

Typical usage inside a mouse executor::

    from qontinui.action_executors.accessibility_action import try_accessibility_click

    result = await try_accessibility_click(
        self.context.last_action_result,
        self.context.hal_container,
    )
    if result.handled:
        if result.wait_after_s > 0:
            await asyncio.sleep(result.wait_after_s)
        return result.success
    # ... fall through to generic click
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AccessibilityActionResult:
    """Result of an accessibility-aware action attempt.

    Attributes:
        handled:      True if the action was dispatched via accessibility.
                      False means the caller should fall back to generic input.
        success:      True if the accessibility action completed successfully.
                      Only meaningful when ``handled`` is True.
        wait_after_s: Recommended post-action stabilisation wait in seconds.
                      Already scaled by the speed-profile multiplier.
    """

    handled: bool
    success: bool
    wait_after_s: float = 0.0


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


async def try_accessibility_click(
    last_action_result: Any,
    hal_container: Any,
) -> AccessibilityActionResult:
    """Try to click via UIA patterns instead of generic mouse.

    Inspects ``last_action_result`` for a match whose backend_metadata
    contains an accessibility ``ref``.  If found, dispatches the click
    through the :class:`ActionDispatchRegistry` (or falls back to
    ``click_by_ref`` directly).

    Args:
        last_action_result: The :class:`ActionResult` stored on the execution
                            context, or *None*.
        hal_container:      The :class:`HALContainer` holding
                            ``accessibility_capture`` and ``action_dispatch``.

    Returns:
        :class:`AccessibilityActionResult` — check ``handled`` before
        deciding whether to run the normal mouse path.
    """
    ref, role_str, node = _extract_accessibility_info(last_action_result)
    if ref is None:
        return AccessibilityActionResult(handled=False, success=False)

    capture = _get_capture(hal_container)
    if capture is None:
        return AccessibilityActionResult(handled=False, success=False)

    registry = getattr(hal_container, "action_dispatch", None)

    if registry is not None and node is not None:
        try:
            dispatch_result = await registry.dispatch(node, "click", capture)
            return AccessibilityActionResult(
                handled=True,
                success=dispatch_result.success,
                wait_after_s=dispatch_result.wait_after_s,
            )
        except Exception as exc:
            logger.warning(
                "Accessibility dispatch (click) failed for %s: %s — falling back to click_by_ref",
                ref,
                exc,
            )

    # Fallback: direct click_by_ref without pattern strategy
    try:
        ok = await capture.click_by_ref(ref)
        return AccessibilityActionResult(handled=True, success=bool(ok), wait_after_s=0.05)
    except Exception as exc:
        logger.warning("click_by_ref failed for %s: %s", ref, exc)
        return AccessibilityActionResult(handled=True, success=False)


async def try_accessibility_type(
    last_action_result: Any,
    hal_container: Any,
    text: str,
    clear_first: bool = False,
) -> AccessibilityActionResult:
    """Try to type via UIA Value/type_by_ref instead of generic keyboard.

    Args:
        last_action_result: The :class:`ActionResult` stored on the execution
                            context, or *None*.
        hal_container:      The :class:`HALContainer`.
        text:               Text to type.
        clear_first:        Whether to clear existing content before typing.

    Returns:
        :class:`AccessibilityActionResult` — check ``handled`` before
        deciding whether to run the normal keyboard path.
    """
    ref, role_str, node = _extract_accessibility_info(last_action_result)
    if ref is None:
        return AccessibilityActionResult(handled=False, success=False)

    capture = _get_capture(hal_container)
    if capture is None:
        return AccessibilityActionResult(handled=False, success=False)

    registry = getattr(hal_container, "action_dispatch", None)

    if registry is not None and node is not None:
        try:
            dispatch_result = await registry.dispatch(
                node, "type", capture, text=text, clear_first=clear_first
            )
            return AccessibilityActionResult(
                handled=True,
                success=dispatch_result.success,
                wait_after_s=dispatch_result.wait_after_s,
            )
        except Exception as exc:
            logger.warning(
                "Accessibility dispatch (type) failed for %s: %s — falling back to type_by_ref",
                ref,
                exc,
            )

    # Fallback: direct type_by_ref without pattern strategy
    try:
        ok = await capture.type_by_ref(ref, text, clear_first=clear_first)
        return AccessibilityActionResult(handled=True, success=bool(ok), wait_after_s=0.03)
    except Exception as exc:
        logger.warning("type_by_ref failed for %s: %s", ref, exc)
        return AccessibilityActionResult(handled=True, success=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_capture(hal_container: Any) -> Any | None:
    """Extract accessibility_capture from hal_container, or None."""
    if hal_container is None:
        return None
    capture = getattr(hal_container, "accessibility_capture", None)
    if capture is None:
        return None
    # Guard: only use if the capture is connected
    is_connected = getattr(capture, "is_connected", None)
    if callable(is_connected) and not is_connected():
        logger.debug("accessibility_capture not connected — skipping accessibility path")
        return None
    return capture


def _extract_accessibility_info(
    last_action_result: Any,
) -> tuple[str | None, str | None, Any]:
    """Extract (ref, role_str, AccessibilityNode) from the best match.

    Looks at ``last_action_result.matches[0].match_object.metadata.backend_metadata``
    for ``ref`` and ``role`` keys placed there by the accessibility detection backend.

    Returns:
        Tuple of (ref, role_str, node).  All three are *None* when no
        accessibility metadata is present.  ``node`` is *None* if we cannot
        import/construct an ``AccessibilityNode`` (e.g. schema package absent).
    """
    if last_action_result is None:
        return None, None, None

    matches = getattr(last_action_result, "matches", None)
    if not matches:
        return None, None, None

    # Take the best (first) match
    best_match = matches[0]

    # Navigate to the backend_metadata dict stored by find_executor
    match_obj = getattr(best_match, "match_object", None)
    if match_obj is None:
        return None, None, None

    meta = getattr(match_obj, "metadata", None)
    if meta is None:
        return None, None, None

    backend_meta: dict = getattr(meta, "backend_metadata", {}) or {}
    ref: str | None = backend_meta.get("ref")
    role_str: str | None = backend_meta.get("role")

    if not ref:
        return None, None, None

    # Try to reconstruct a minimal AccessibilityNode for the dispatch registry
    node = _build_node(ref, role_str, best_match)
    return ref, role_str, node


def _build_node(ref: str, role_str: str | None, match: Any) -> Any | None:
    """Construct a minimal AccessibilityNode from a ref and optional role string.

    Returns None if the schema package is unavailable or role is unknown.
    """
    try:
        from qontinui_schemas.accessibility import AccessibilityNode, AccessibilityRole

        # Resolve role — default to UNKNOWN if the string doesn't map to a value
        role = AccessibilityRole.UNKNOWN
        if role_str:
            try:
                role = AccessibilityRole(role_str)
            except ValueError:
                pass  # Keep UNKNOWN default

        # Optionally populate bounds from the match region
        bounds = None
        try:
            from qontinui_schemas.accessibility import AccessibilityBounds

            region = getattr(match, "region", None)
            if region is None:
                # Try via match_object
                mo = getattr(match, "match_object", None)
                if mo is not None:
                    region = mo.get_region() if callable(getattr(mo, "get_region", None)) else None
            if region is not None:
                bounds = AccessibilityBounds(
                    x=int(getattr(region, "x", 0)),
                    y=int(getattr(region, "y", 0)),
                    width=int(getattr(region, "width", 0)),
                    height=int(getattr(region, "height", 0)),
                )
        except Exception:
            pass  # Bounds are optional

        return AccessibilityNode(ref=ref, role=role, bounds=bounds)

    except ImportError:
        logger.debug("qontinui_schemas not available — skipping AccessibilityNode construction")
        return None
    except Exception as exc:
        logger.debug("Failed to construct AccessibilityNode for %s: %s", ref, exc)
        return None
