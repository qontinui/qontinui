"""Shared spatial label inference utilities for UIA accessibility trees.

Provides :func:`infer_spatial_labels`, a preprocessing step that assigns
inferred labels to unlabeled interactive controls by inspecting adjacent
Static/Text/Label nodes — mirroring pywinauto's
``get_non_text_control_name()`` algorithm.
"""

from __future__ import annotations

import logging

from qontinui_schemas.accessibility import AccessibilityNode, AccessibilityRole

logger = logging.getLogger(__name__)

# Roles that act as visible text labels for adjacent controls.
_LABEL_ROLES: frozenset[AccessibilityRole] = frozenset(
    {
        AccessibilityRole.STATIC_TEXT,
        AccessibilityRole.HEADING,
        AccessibilityRole.PARAGRAPH,
        AccessibilityRole.DEFINITION,
        AccessibilityRole.TERM,
        AccessibilityRole.TOOLTIP,
        AccessibilityRole.STATUS,
        # Some apps use GENERIC or NONE nodes as plain text labels.
        AccessibilityRole.GENERIC,
    }
)

# Maximum pixel gap between a label node and an unlabeled control.
_MAX_LABEL_GAP_PX: int = 300

# Hard cap on total (text_node × unlabeled_node) comparisons per call.
_MAX_COMPARISONS: int = 500


def _node_center(node: AccessibilityNode) -> tuple[int, int] | None:
    """Return (cx, cy) for a node, or None if bounds are missing."""
    b = node.bounds
    if b is None:
        return None
    if hasattr(b, "x"):
        return (b.x + b.width // 2, b.y + b.height // 2)
    if isinstance(b, (list, tuple)) and len(b) >= 4:
        x, y, w, h = int(b[0]), int(b[1]), int(b[2]), int(b[3])
        return (x + w // 2, y + h // 2)
    return None


def _node_top_left(node: AccessibilityNode) -> tuple[int, int] | None:
    """Return (x, y) top-left for a node, or None if bounds are missing."""
    b = node.bounds
    if b is None:
        return None
    if hasattr(b, "x"):
        return (b.x, b.y)
    if isinstance(b, (list, tuple)) and len(b) >= 2:
        return (int(b[0]), int(b[1]))
    return None


def infer_spatial_labels(
    nodes: list[AccessibilityNode],
) -> dict[str, str]:
    """Map ``node.ref`` -> inferred label for unlabeled interactive nodes.

    Only considers Static/Text/Label nodes above or to the left of the
    target, per pywinauto's ``get_non_text_control_name()`` algorithm.
    Caps at :data:`_MAX_COMPARISONS` total comparisons for pathological trees.

    Args:
        nodes: Flat list of accessibility nodes (any mix of roles).

    Returns:
        Mapping from ``node.ref`` to the inferred label string for every
        unlabeled interactive node that has a sufficiently close label
        neighbour.
    """
    text_nodes: list[AccessibilityNode] = []
    unlabeled: list[AccessibilityNode] = []

    for node in nodes:
        name = (node.name or "").strip()
        if node.role in _LABEL_ROLES and name:
            text_nodes.append(node)
        elif node.is_interactive and not name and node.bounds is not None:
            unlabeled.append(node)

    if not text_nodes or not unlabeled:
        return {}

    result: dict[str, str] = {}
    comparisons = 0

    for unode in unlabeled:
        u_pos = _node_top_left(unode)
        if u_pos is None:
            continue
        u_x, u_y = u_pos

        best_label: str | None = None
        best_dist: float = float("inf")

        for tnode in text_nodes:
            if comparisons >= _MAX_COMPARISONS:
                break

            comparisons += 1

            t_pos = _node_top_left(tnode)
            if t_pos is None:
                continue
            t_x, t_y = t_pos

            # Only consider labels that are above or to the left of the control.
            # "Above": t_y < u_y  (label top edge is above control top edge)
            # "Left" : t_x < u_x  (label left edge is left of control left edge)
            if t_y >= u_y and t_x >= u_x:
                continue

            # Manhattan distance between the two top-left corners.
            dist = abs(t_x - u_x) + abs(t_y - u_y)
            if dist > _MAX_LABEL_GAP_PX:
                continue

            if dist < best_dist:
                best_dist = dist
                best_label = (tnode.name or "").strip()

        if best_label:
            result[unode.ref] = best_label

        if comparisons >= _MAX_COMPARISONS:
            logger.debug(
                "infer_spatial_labels: comparison cap (%d) reached, "
                "%d/%d unlabeled nodes processed",
                _MAX_COMPARISONS,
                len(result),
                len(unlabeled),
            )
            break

    return result


__all__ = ["infer_spatial_labels"]
