"""Reference manager for accessibility nodes.

This module provides a ref assignment and lookup system for accessibility
nodes, enabling stable identifiers (@e1, @e2, etc.) that can be used for
AI-driven automation.

Supports optional persistence: ``save()``/``load()`` serialize the ref→element
fingerprint mapping to JSON so learned element locations survive reconnections.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from qontinui_schemas.accessibility import AccessibilityNode

logger = logging.getLogger(__name__)


class RefManager:
    """Manages ref assignment and lookup for accessibility nodes.

    Refs are assigned sequentially (@e1, @e2, etc.) during tree traversal.
    The manager maintains a mapping from refs to nodes for quick lookup.

    Example:
        >>> manager = RefManager()
        >>> manager.assign_refs(root_node)
        >>> node = manager.get_node_by_ref("@e5")
        >>> if node:
        ...     print(f"Found: {node.role} '{node.name}'")
    """

    def __init__(self) -> None:
        """Initialize the ref manager."""
        self._counter: int = 0
        self._nodes_by_ref: dict[str, AccessibilityNode] = {}

    def reset(self) -> None:
        """Reset the ref manager state.

        Clears all ref assignments. Call this before processing a new tree.
        """
        self._counter = 0
        self._nodes_by_ref.clear()

    def clear(self) -> None:
        """Clear the ref manager state.

        Alias for reset(). Clears all ref assignments.
        """
        self.reset()

    def assign_ref(self, *, is_interactive: bool = True) -> str:
        """Assign a single ref for manual tree building.

        This method is used when building trees node-by-node (e.g., UIA capture)
        rather than assigning refs to a complete tree.

        Args:
            is_interactive: Whether this node is interactive (for tracking)

        Returns:
            The assigned ref (e.g., "@e1", "@e2")
        """
        self._counter += 1
        return f"@e{self._counter}"

    def assign_refs(
        self,
        node: AccessibilityNode,
        *,
        interactive_only: bool = False,
    ) -> int:
        """Assign refs to all nodes in the tree.

        Traverses the tree depth-first and assigns sequential refs
        (@e1, @e2, etc.) to each node.

        Args:
            node: Root node of the tree
            interactive_only: If True, only assign refs to interactive nodes

        Returns:
            Total number of refs assigned
        """
        self.reset()
        self._assign_refs_recursive(node, interactive_only)
        return self._counter

    def _assign_refs_recursive(
        self,
        node: AccessibilityNode,
        interactive_only: bool,
    ) -> None:
        """Recursively assign refs to nodes.

        Args:
            node: Current node
            interactive_only: If True, only assign refs to interactive nodes
        """
        # Assign ref to this node
        if not interactive_only or node.is_interactive:
            self._counter += 1
            ref = f"@e{self._counter}"
            node.ref = ref
            self._nodes_by_ref[ref] = node
        else:
            # Still need a ref, but it won't be indexed
            node.ref = ""

        # Process children
        for child in node.children:
            self._assign_refs_recursive(child, interactive_only)

    def get_node_by_ref(self, ref: str) -> AccessibilityNode | None:
        """Get a node by its ref.

        Args:
            ref: Reference ID (e.g., "@e1", "@e5")

        Returns:
            The node if found, None otherwise
        """
        return self._nodes_by_ref.get(ref)

    def get_all_refs(self) -> list[str]:
        """Get all assigned refs.

        Returns:
            List of all ref IDs
        """
        return list(self._nodes_by_ref.keys())

    def get_interactive_nodes(self) -> list[AccessibilityNode]:
        """Get all nodes that have refs assigned.

        Returns:
            List of nodes with refs
        """
        return list(self._nodes_by_ref.values())

    @property
    def count(self) -> int:
        """Get the number of refs assigned.

        Returns:
            Number of refs
        """
        return len(self._nodes_by_ref)

    # =================================================================
    # Persistence
    # =================================================================

    def save(self, path: str | Path) -> int:
        """Save ref→element fingerprints to a JSON file.

        Persists enough identity information (automation_id, role, name,
        bounds) to attempt re-resolution on a future tree capture.

        Args:
            path: File path to write JSON to.

        Returns:
            Number of refs saved.
        """
        entries: dict[str, dict[str, Any]] = {}
        for ref, node in self._nodes_by_ref.items():
            bounds = None
            if node.bounds is not None:
                if hasattr(node.bounds, "x"):
                    bounds = [
                        node.bounds.x,
                        node.bounds.y,
                        node.bounds.width,
                        node.bounds.height,
                    ]
                elif isinstance(node.bounds, (list, tuple)):
                    bounds = list(node.bounds[:4])

            entries[ref] = {
                "automation_id": node.automation_id,
                "role": getattr(node.role, "value", str(node.role)) if node.role else None,
                "name": node.name,
                "class_name": node.class_name,
                "bounds": bounds,
            }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        logger.debug("Saved %d ref fingerprints to %s", len(entries), path)
        return len(entries)

    def load(self, path: str | Path) -> dict[str, dict[str, Any]]:
        """Load persisted ref fingerprints from a JSON file.

        Does NOT restore refs into the manager — caller must re-resolve
        fingerprints against a fresh tree capture and call ``assign_refs``
        or ``register_node`` as appropriate.

        Args:
            path: File path to read JSON from.

        Returns:
            Dict of ref → fingerprint dicts. Empty dict if file missing.
        """
        path = Path(path)
        if not path.exists():
            return {}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            logger.debug("Loaded %d ref fingerprints from %s", len(data), path)
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load ref fingerprints from %s: %s", path, e)
            return {}

    def register_node(self, ref: str, node: AccessibilityNode) -> None:
        """Manually register a node under a specific ref.

        Used during re-resolution of persisted fingerprints against a
        new tree capture.

        Args:
            ref: The ref to assign (e.g., "@e5").
            node: The node to register.
        """
        node.ref = ref
        self._nodes_by_ref[ref] = node
