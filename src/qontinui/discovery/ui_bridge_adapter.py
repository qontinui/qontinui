"""Adapter to discover states from UI Bridge render snapshots.

This module bridges UI Bridge's semantic element data with qontinui's
co-occurrence analysis algorithm to discover application states.

**NOTE: This module now delegates to the unified StateDiscoveryService.**
The functions here are maintained for backward compatibility. For new code,
use the StateDiscoveryService directly:

    from qontinui.discovery.state_discovery import StateDiscoveryService

    service = StateDiscoveryService()
    result = service.discover_from_renders(renders)

UI Bridge provides:
- Registered elements (via useUIElement hook with data-ui-id attribute)
- data-testid attributes from the component tree

The co-occurrence algorithm groups elements that appear together
across multiple renders into states.

Supports two render log formats:
1. DomSnapshotRenderLogEntry (from qontinui-web) - Full DOM snapshots
2. Simple format (for testing) - {"elements": [...], "componentTree": {...}}

For fingerprint-enhanced discovery:
    from qontinui.discovery.state_discovery import StateDiscoveryService

    service = StateDiscoveryService()
    result = service.discover_from_export(cooccurrence_export)

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

from .models import DiscoveredState
from .state_discovery import DiscoveryStrategyType, StateDiscoveryService


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
    cooccurrence_export: dict[str, Any] | None = None,
) -> UIBridgeStateDiscoveryResult:
    """
    Discover states from UI Bridge render snapshots using co-occurrence analysis.

    Elements that appear in exactly the same set of renders are grouped into states.
    This identifies stable UI patterns across different application views.

    **NEW:** If `cooccurrence_export` is provided with fingerprint data,
    the enhanced fingerprint strategy will be used automatically.

    Args:
        renders: List of render log entries from UI Bridge. Each entry should have:
            - id: Unique render identifier
            - elements: List of registered UI Bridge elements
            - componentTree or tree: Component tree with data-testid attributes
        include_html_ids: Whether to include HTML id attributes
        cooccurrence_export: Optional co-occurrence export with fingerprint data.
            If provided and contains fingerprint data, uses enhanced discovery.

    Returns:
        UIBridgeStateDiscoveryResult containing discovered states and element info
    """
    if not renders and not cooccurrence_export:
        return UIBridgeStateDiscoveryResult(
            states=[],
            elements=[],
            element_to_renders={},
            render_count=0,
            unique_element_count=0,
        )

    # Use the unified service
    service = StateDiscoveryService()

    # If cooccurrence_export with fingerprints is provided, use AUTO to prefer fingerprints
    if cooccurrence_export:
        unified_result = service.discover_from_export(
            cooccurrence_export,
            strategy=DiscoveryStrategyType.AUTO,
        )
    else:
        unified_result = service.discover_from_renders(
            renders,
            include_html_ids=include_html_ids,
            strategy=DiscoveryStrategyType.LEGACY,
        )

    # Convert unified result back to legacy format for backward compatibility
    legacy_states = []
    for state in unified_result.states:
        legacy_states.append(
            DiscoveredState(
                id=state.id,
                name=state.name,
                state_image_ids=state.element_ids,
                screenshot_ids=state.render_ids,
                confidence=state.confidence,
                metadata=state.metadata,
            )
        )

    legacy_elements = []
    for elem in unified_result.elements:
        legacy_elements.append(
            UIBridgeElement(
                id=elem.id,
                name=elem.name,
                type=elem.element_type,
                render_ids=elem.render_ids,
                tag_name=elem.tag_name,
                text_content=elem.text_content,
                component_name=elem.component_name,
            )
        )

    return UIBridgeStateDiscoveryResult(
        states=legacy_states,
        elements=legacy_elements,
        element_to_renders=unified_result.element_to_renders,
        render_count=unified_result.render_count,
        unique_element_count=unified_result.unique_element_count,
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


def convert_to_ui_bridge_states(
    result: UIBridgeStateDiscoveryResult,
) -> list[Any]:
    """
    Convert discovery results to UIBridgeState objects for state_machine module.

    This bridges the discovery adapter with the state_machine runtime.

    Args:
        result: State discovery result from discover_states_from_renders

    Returns:
        List of UIBridgeState objects ready for runtime registration

    Example:
        renders = [...] # Render log entries
        result = discover_states_from_renders(renders)
        ui_states = convert_to_ui_bridge_states(result)

        from qontinui.state_machine import UIBridgeRuntime
        runtime = UIBridgeRuntime(client)
        runtime.register_states(ui_states)
    """
    # Import here to avoid circular dependency
    from qontinui.state_machine import UIBridgeState

    ui_states: list[UIBridgeState] = []

    for state in result.states:
        # Determine if state is blocking (modal dialogs, etc.)
        is_blocking = False
        modal_indicators = ["modal", "dialog", "popup", "alert", "overlay"]
        state_name_lower = state.name.lower()

        for indicator in modal_indicators:
            if indicator in state_name_lower:
                is_blocking = True
                break

        # Determine group from element patterns
        group = None
        element_str = " ".join(state.state_image_ids).lower()
        if "nav" in element_str or "header" in element_str:
            group = "navigation"
        elif "sidebar" in element_str:
            group = "sidebar"
        elif "modal" in element_str or "dialog" in element_str:
            group = "modals"
        elif "form" in element_str:
            group = "forms"

        ui_state = UIBridgeState(
            id=state.id,
            name=state.name,
            element_ids=state.state_image_ids,
            blocking=is_blocking,
            blocks=[],  # Will be computed by runtime
            group=group,
            path_cost=1.0,
            metadata={
                "confidence": state.confidence,
                "render_ids": state.screenshot_ids,
                "discovery_metadata": state.metadata,
            },
        )

        ui_states.append(ui_state)

    return ui_states
