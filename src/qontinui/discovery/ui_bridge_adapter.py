"""Adapter to discover states from UI Bridge render snapshots.

This module bridges UI Bridge's semantic element data with qontinui's
co-occurrence analysis algorithm to discover application states.

UI Bridge provides:
- Registered elements (via useUIElement hook with data-ui-id attribute)
- data-testid attributes from the component tree

The co-occurrence algorithm groups elements that appear together
across multiple renders into states.

Supports two render log formats:
1. DomSnapshotRenderLogEntry (from qontinui-web) - Full DOM snapshots
2. Simple format (for testing) - {"elements": [...], "componentTree": {...}}

Example:
    renders = [
        {"id": "r1", "elements": [{"id": "nav"}], "componentTree": {...}},
        {"id": "r2", "elements": [{"id": "nav"}], "componentTree": {...}},
    ]
    states = discover_states_from_renders(renders)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from .models import DiscoveredState, StateImage
from .pixel_analysis.analyzers.cooccurrence_analyzer import CooccurrenceAnalyzer


@dataclass
class UIBridgeElement:
    """An element extracted from a UI Bridge render snapshot."""

    id: str  # Element ID (prefixed with type, e.g., "ui:nav-menu" or "testid:sidebar")
    name: str  # Human-readable name (without prefix)
    type: str  # 'ui-id' | 'testid' | 'html-id'
    render_ids: list[str] = field(default_factory=list)

    # Optional metadata from UI Bridge
    tag_name: str | None = None
    text_content: str | None = None
    component_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "renderIds": self.render_ids,
            "tagName": self.tag_name,
            "textContent": self.text_content,
            "componentName": self.component_name,
        }


@dataclass
class UIBridgeRender:
    """A render snapshot from UI Bridge."""

    id: str  # Unique render ID
    route: str  # Current route/URL
    timestamp: datetime
    element_ids: list[str]  # Elements present in this render

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "route": self.route,
            "timestamp": self.timestamp.isoformat(),
            "elementIds": self.element_ids,
        }


@dataclass
class UIBridgeStateDiscoveryResult:
    """Result of state discovery from UI Bridge renders."""

    states: list[DiscoveredState]
    elements: list[UIBridgeElement]
    element_to_renders: dict[str, list[str]]
    render_count: int
    unique_element_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "states": [s.to_dict() for s in self.states],
            "elements": [e.to_dict() for e in self.elements],
            "elementToRenders": self.element_to_renders,
            "renderCount": self.render_count,
            "uniqueElementCount": self.unique_element_count,
        }


def _extract_from_dom_element(
    node: dict[str, Any],
    element_ids: set[str],
    include_html_ids: bool = False,
) -> None:
    """
    Recursively extract element IDs from a DomElementSnapshot node.

    Extracts:
    - data-ui-id (UI Bridge registered elements)
    - data-testid (testing convention)
    - Optionally: HTML id attributes

    Args:
        node: A DomElementSnapshot dict
        element_ids: Set to add found element IDs to
        include_html_ids: Whether to include HTML id attributes
    """
    if not isinstance(node, dict):
        return

    # Get attributes dict (DomElementSnapshot format)
    attrs = node.get("attributes", {})
    if isinstance(attrs, dict):
        # Priority 1: data-ui-id (UI Bridge registered elements)
        ui_id = attrs.get("data-ui-id")
        if ui_id:
            element_ids.add(f"ui:{ui_id}")

        # Priority 2: data-testid (testing convention)
        testid = attrs.get("data-testid")
        if testid:
            element_ids.add(f"testid:{testid}")

        # Optional: HTML id attribute
        if include_html_ids:
            html_id = node.get("id") or attrs.get("id")
            if html_id:
                element_ids.add(f"html:{html_id}")

    # Also check props (for React component tree format)
    props = node.get("props", {})
    if isinstance(props, dict):
        ui_id = props.get("data-ui-id")
        if ui_id:
            element_ids.add(f"ui:{ui_id}")

        testid = props.get("data-testid")
        if testid:
            element_ids.add(f"testid:{testid}")

    # Recurse into children
    children = node.get("children", [])
    if isinstance(children, list):
        for child in children:
            _extract_from_dom_element(child, element_ids, include_html_ids)


def extract_elements_from_render(
    render_log_entry: dict[str, Any],
    include_html_ids: bool = False,
) -> list[str]:
    """
    Extract element IDs from a UI Bridge render log entry.

    Supports two formats:
    1. DomSnapshotRenderLogEntry: {"type": "dom_snapshot", "snapshot": {"root": ...}}
    2. Simple format: {"elements": [...], "componentTree": {...}}

    Uses registered elements (data-ui-id) + data-testid attributes.

    Args:
        render_log_entry: Single entry from UI Bridge render log
        include_html_ids: Whether to include HTML id attributes (default False)

    Returns:
        List of element IDs present in this render (prefixed with type)
    """
    element_ids: set[str] = set()

    # Format 1: DomSnapshotRenderLogEntry (from qontinui-web)
    if render_log_entry.get("type") == "dom_snapshot":
        snapshot = render_log_entry.get("snapshot", {})
        root = snapshot.get("root")
        if root:
            _extract_from_dom_element(root, element_ids, include_html_ids)
        return sorted(element_ids)

    # Format 2: Simple format (for testing and backward compatibility)

    # Extract registered elements (from useUIElement hook - simple format)
    if "elements" in render_log_entry:
        for elem in render_log_entry["elements"]:
            elem_id = elem.get("id")
            if elem_id:
                # Use "reg:" prefix for backward compatibility with tests
                element_ids.add(f"reg:{elem_id}")

    # Extract from componentTree (simple format)
    if "componentTree" in render_log_entry:
        _extract_from_dom_element(render_log_entry["componentTree"], element_ids, include_html_ids)

    # Also check "tree" key (alternative simple format)
    if "tree" in render_log_entry:
        _extract_from_dom_element(render_log_entry["tree"], element_ids, include_html_ids)

    return sorted(element_ids)


def build_element_render_mapping(
    renders: list[dict[str, Any]],
    include_html_ids: bool = False,
) -> tuple[dict[str, set[str]], list[tuple[str, list[str]]]]:
    """
    Build mapping of elements to the renders they appear in.

    Args:
        renders: List of render log entries from UI Bridge

    Returns:
        Tuple of (element_to_renders mapping, list of (render_id, elements) tuples)
    """
    element_to_renders: dict[str, set[str]] = {}
    render_elements: list[tuple[str, list[str]]] = []

    for i, render in enumerate(renders):
        render_id = render.get("id", f"render_{i}")
        elements = extract_elements_from_render(render, include_html_ids)
        render_elements.append((render_id, elements))

        for elem_id in elements:
            if elem_id not in element_to_renders:
                element_to_renders[elem_id] = set()
            element_to_renders[elem_id].add(render_id)

    return element_to_renders, render_elements


def discover_states_from_renders(
    renders: list[dict[str, Any]],
    include_html_ids: bool = False,
) -> UIBridgeStateDiscoveryResult:
    """
    Discover states from UI Bridge render snapshots using co-occurrence analysis.

    Elements that appear in exactly the same set of renders are grouped into states.
    This identifies stable UI patterns across different application views.

    Args:
        renders: List of render log entries from UI Bridge. Each entry should have:
            - id: Unique render identifier
            - elements: List of registered UI Bridge elements
            - componentTree or tree: Component tree with data-testid attributes

    Returns:
        UIBridgeStateDiscoveryResult containing discovered states and element info
    """
    if not renders:
        return UIBridgeStateDiscoveryResult(
            states=[],
            elements=[],
            element_to_renders={},
            render_count=0,
            unique_element_count=0,
        )

    # Step 1: Build element -> render_ids mapping
    element_to_renders, render_elements = build_element_render_mapping(renders, include_html_ids)

    if not element_to_renders:
        return UIBridgeStateDiscoveryResult(
            states=[],
            elements=[],
            element_to_renders={},
            render_count=len(renders),
            unique_element_count=0,
        )

    # Step 2: Convert to StateImage format (minimal fields needed for algorithm)
    state_images: list[StateImage] = []
    for elem_id, render_ids in element_to_renders.items():
        state_images.append(
            StateImage(
                id=elem_id,
                name=elem_id,
                x=0,
                y=0,
                x2=1,
                y2=1,  # Placeholder coordinates (not used)
                pixel_hash="",  # Not used for UI Bridge
                frequency=len(render_ids) / len(renders),
                screenshot_ids=sorted(render_ids),
            )
        )

    # Step 3: Run co-occurrence analysis
    # Create dummy screenshot array (algorithm normalizes by length)
    dummy_screenshots = [np.zeros((1, 1, 3), dtype=np.uint8) for _ in renders]

    analyzer = CooccurrenceAnalyzer()
    discovered_states = analyzer.analyze(state_images, dummy_screenshots)

    # Step 4: Build UIBridgeElement list with details
    elements = []
    for elem_id, render_ids in element_to_renders.items():
        # Determine element type from prefix
        if elem_id.startswith("ui:"):
            elem_type = "ui-id"
        elif elem_id.startswith("testid:"):
            elem_type = "testid"
        elif elem_id.startswith("html:"):
            elem_type = "html-id"
        elif elem_id.startswith("reg:"):
            elem_type = "registered"  # Backward compatibility
        else:
            elem_type = "unknown"

        clean_name = elem_id.split(":", 1)[1] if ":" in elem_id else elem_id

        elements.append(
            UIBridgeElement(
                id=elem_id,
                name=clean_name,
                type=elem_type,
                render_ids=sorted(render_ids),
            )
        )

    # Convert set values to lists for JSON serialization
    element_to_renders_serializable = {k: sorted(v) for k, v in element_to_renders.items()}

    return UIBridgeStateDiscoveryResult(
        states=discovered_states,
        elements=elements,
        element_to_renders=element_to_renders_serializable,
        render_count=len(renders),
        unique_element_count=len(element_to_renders),
    )


def get_state_elements(
    state: DiscoveredState,
    all_elements: list[UIBridgeElement],
) -> list[UIBridgeElement]:
    """
    Get detailed element info for elements in a discovered state.

    Args:
        state: A discovered state
        all_elements: List of all UIBridgeElements from discovery result

    Returns:
        List of UIBridgeElements that belong to this state
    """
    element_ids = set(state.state_image_ids)
    return [elem for elem in all_elements if elem.id in element_ids]


def get_elements_by_render(
    render_id: str,
    element_to_renders: dict[str, list[str]],
    all_elements: list[UIBridgeElement],
) -> list[UIBridgeElement]:
    """
    Get all elements present in a specific render.

    Args:
        render_id: The render ID to query
        element_to_renders: Mapping from element ID to render IDs
        all_elements: List of all UIBridgeElements

    Returns:
        List of UIBridgeElements present in the specified render
    """
    element_ids = {
        elem_id for elem_id, render_ids in element_to_renders.items() if render_id in render_ids
    }
    return [elem for elem in all_elements if elem.id in element_ids]


def get_active_states_for_render(
    render_id: str,
    states: list[DiscoveredState],
) -> list[DiscoveredState]:
    """
    Get all states that are active in a specific render.

    Args:
        render_id: The render ID to query
        states: List of discovered states

    Returns:
        List of states that are active in the specified render
    """
    return [state for state in states if render_id in state.screenshot_ids]
