"""State Discovery from UI Bridge Data.

This module discovers application states from UI Bridge render snapshots
using element co-occurrence analysis. Supports both batch and incremental
learning for real-time state discovery.

The discovery process:
1. Collect render snapshots from UI Bridge
2. Extract element IDs from each render
3. Group elements that appear/disappear together
4. Generate states from element groups
5. Detect modal/blocking states

Features:
- Batch discovery from render history
- Incremental learning from real-time renders
- Automatic state merging based on similarity
- Confidence tracking based on observation count
- State graph export for visualization

Example:
    from qontinui.state_machine import UIBridgeStateDiscovery

    discovery = UIBridgeStateDiscovery()

    # Batch mode: Process renders from UI Bridge
    for render in renders:
        discovery.process_render(render)

    # Get discovered states
    states = discovery.get_discovered_states()

    # Incremental mode: Enable auto-discovery
    discovery.enable_incremental_mode()
    discovery.process_render_incremental(new_render)

    # Export state graph
    graph = discovery.get_state_graph()
    dot_output = discovery.export_state_graph("dot")
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class StateGraphFormat(Enum):
    """Export formats for state graph."""

    JSON = "json"
    DOT = "dot"
    MERMAID = "mermaid"


@dataclass
class StateGraphNode:
    """Node in the state graph for visualization."""

    id: str
    label: str
    node_type: str  # "discovered" or "merged"
    observation_count: int
    confidence: float
    element_count: int
    is_blocking: bool
    position: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "label": self.label,
            "type": self.node_type,
            "observationCount": self.observation_count,
            "confidence": self.confidence,
            "elementCount": self.element_count,
            "isBlocking": self.is_blocking,
            "position": self.position,
        }


@dataclass
class StateGraphEdge:
    """Edge in the state graph for visualization."""

    id: str
    source: str
    target: str
    label: str
    reliability: float
    observation_count: int
    bidirectional: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "label": self.label,
            "reliability": self.reliability,
            "observationCount": self.observation_count,
            "bidirectional": self.bidirectional,
        }


@dataclass
class StateGraph:
    """State graph for visualization."""

    nodes: list[StateGraphNode]
    edges: list[StateGraphEdge]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Export to JSON format."""
        return json.dumps(self.to_dict(), indent=2)

    def to_dot(self) -> str:
        """Export to DOT (Graphviz) format."""
        lines = ["digraph StateGraph {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box];")
        lines.append("")

        for node in self.nodes:
            attrs = [f'label="{node.label}"']
            if node.is_blocking:
                attrs.append("shape=diamond")
            if node.node_type == "discovered":
                attrs.append("style=dashed")
            lines.append(f'  "{node.id}" [{", ".join(attrs)}];')

        lines.append("")

        for edge in self.edges:
            attrs = [f'label="{edge.label}"']
            if edge.reliability < 0.9:
                attrs.append("style=dashed")
            if edge.bidirectional:
                attrs.append("dir=both")
            lines.append(f'  "{edge.source}" -> "{edge.target}" [{", ".join(attrs)}];')

        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Export to Mermaid format."""
        lines = ["stateDiagram-v2"]

        for node in self.nodes:
            if node.is_blocking:
                lines.append(f'  state "{node.label}" as {node.id}')
                lines.append(f"  note right of {node.id}: Blocking state")

        for edge in self.edges:
            arrow = "<-->" if edge.bidirectional else "-->"
            lines.append(f"  {edge.source} {arrow} {edge.target}: {edge.label}")

        return "\n".join(lines)


@dataclass
class StateDiscoveryConfig:
    """Configuration for state discovery."""

    # Minimum renders for element to be considered
    min_element_occurrences: int = 2

    # Co-occurrence threshold (0.0-1.0) for grouping elements
    cooccurrence_threshold: float = 0.9

    # Minimum elements per state
    min_state_elements: int = 1

    # Maximum elements per state
    max_state_elements: int = 50

    # Keywords indicating modal/blocking states
    modal_indicators: list[str] = field(
        default_factory=lambda: [
            "modal",
            "dialog",
            "popup",
            "alert",
            "confirm",
            "overlay",
            "dropdown",
            "menu",
        ]
    )

    # Element type prefixes for identification
    element_type_prefixes: dict[str, str] = field(
        default_factory=lambda: {
            "ui:": "ui-id",
            "testid:": "testid",
            "html:": "html-id",
            "reg:": "registered",
        }
    )

    # Incremental learning settings
    incremental_confidence_threshold: float = 0.7
    auto_merge_similar_states: bool = True
    merge_similarity_threshold: float = 0.85
    max_discovered_states: int = 1000

    # Event callbacks
    on_state_discovered: Callable[[DiscoveredUIState], None] | None = None
    on_state_merged: Callable[[str, str], None] | None = None
    on_confidence_updated: Callable[[str, float], None] | None = None


@dataclass
class DiscoveredUIState:
    """A state discovered from UI Bridge data."""

    id: str
    name: str
    element_ids: list[str]
    render_ids: list[str]
    confidence: float
    is_blocking: bool = False
    blocks: list[str] = field(default_factory=list)
    group: str | None = None
    frequency: float = 0.0
    observation_count: int = 1
    first_observed: datetime = field(default_factory=datetime.utcnow)
    last_observed: datetime = field(default_factory=datetime.utcnow)
    merged_from: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "elementIds": self.element_ids,
            "renderIds": self.render_ids,
            "confidence": self.confidence,
            "isBlocking": self.is_blocking,
            "blocks": self.blocks,
            "group": self.group,
            "frequency": self.frequency,
            "observationCount": self.observation_count,
            "firstObserved": self.first_observed.isoformat(),
            "lastObserved": self.last_observed.isoformat(),
            "mergedFrom": self.merged_from,
            "metadata": self.metadata,
            "createdAt": self.created_at.isoformat(),
        }

    def increment_observation(self, render_id: str | None = None) -> None:
        """Increment observation count and update confidence."""
        self.observation_count += 1
        self.last_observed = datetime.utcnow()
        if render_id:
            self.render_ids.append(render_id)

    def merge_with(self, other: DiscoveredUIState) -> None:
        """Merge another state into this one."""
        # Merge element IDs
        merged_elements = set(self.element_ids) | set(other.element_ids)
        self.element_ids = sorted(merged_elements)

        # Merge render IDs
        merged_renders = set(self.render_ids) | set(other.render_ids)
        self.render_ids = sorted(merged_renders)

        # Update counts
        self.observation_count += other.observation_count
        self.last_observed = max(self.last_observed, other.last_observed)
        self.first_observed = min(self.first_observed, other.first_observed)

        # Track merge history
        self.merged_from.append(other.id)
        self.merged_from.extend(other.merged_from)


@dataclass
class ElementInfo:
    """Information about an element across renders."""

    id: str
    type: str
    name: str
    render_ids: set[str] = field(default_factory=set)
    frequency: float = 0.0
    tag_name: str | None = None
    text_content: str | None = None

    def add_render(self, render_id: str) -> None:
        """Record that this element appeared in a render."""
        self.render_ids.add(render_id)


class UIBridgeStateDiscovery:
    """Discovers states from UI Bridge render data.

    This class uses co-occurrence analysis to identify elements that
    appear together, forming application states.

    States are groups of elements with high co-occurrence scores,
    meaning they appear and disappear together across renders.
    """

    def __init__(self, config: StateDiscoveryConfig | None = None) -> None:
        """Initialize state discovery.

        Args:
            config: Discovery configuration
        """
        self.config = config or StateDiscoveryConfig()

        # Element tracking
        self._elements: dict[str, ElementInfo] = {}
        self._render_elements: dict[str, set[str]] = {}
        self._render_count: int = 0

        # Discovery results
        self._discovered_states: list[DiscoveredUIState] = []
        self._element_groups: list[set[str]] = []

    def process_render(self, render: dict[str, Any]) -> list[str]:
        """Process a render snapshot from UI Bridge.

        Args:
            render: Render log entry from UI Bridge

        Returns:
            List of element IDs found in this render
        """
        render_id = render.get("id", f"render_{self._render_count}")
        self._render_count += 1

        # Extract element IDs from render
        element_ids = self._extract_elements(render)

        # Store render -> elements mapping
        self._render_elements[render_id] = set(element_ids)

        # Update element info
        for elem_id in element_ids:
            if elem_id not in self._elements:
                elem_type, elem_name = self._parse_element_id(elem_id)
                self._elements[elem_id] = ElementInfo(
                    id=elem_id,
                    type=elem_type,
                    name=elem_name,
                )
            self._elements[elem_id].add_render(render_id)

        logger.debug(f"Processed render {render_id}: {len(element_ids)} elements")
        return element_ids

    def process_renders(self, renders: list[dict[str, Any]]) -> None:
        """Process multiple renders.

        Args:
            renders: List of render log entries
        """
        for render in renders:
            self.process_render(render)

    def discover_states(self) -> list[DiscoveredUIState]:
        """Run state discovery on processed renders.

        Returns:
            List of discovered states
        """
        if self._render_count == 0:
            logger.warning("No renders processed for state discovery")
            return []

        # Update element frequencies
        self._update_frequencies()

        # Find element co-occurrence groups
        self._find_cooccurrence_groups()

        # Generate states from groups
        self._generate_states()

        # Detect blocking states
        self._detect_blocking_states()

        logger.info(f"Discovered {len(self._discovered_states)} states from {self._render_count} renders")
        return self._discovered_states

    def _extract_elements(self, render: dict[str, Any]) -> list[str]:
        """Extract element IDs from a render.

        Supports multiple render formats:
        1. DomSnapshotRenderLogEntry with nested DOM tree
        2. Simple format with elements/tree arrays
        3. UI Bridge control snapshot format

        Args:
            render: Render log entry

        Returns:
            List of element IDs
        """
        element_ids: set[str] = set()

        # Format 1: DOM snapshot with root element
        if render.get("type") == "dom_snapshot":
            snapshot = render.get("snapshot", {})
            root = snapshot.get("root")
            if root:
                self._extract_from_dom_node(root, element_ids)
            return sorted(element_ids)

        # Format 2: Simple elements array
        if "elements" in render:
            for elem in render["elements"]:
                elem_id = elem.get("id")
                if elem_id:
                    element_ids.add(f"ui:{elem_id}")

        # Format 3: Component tree
        for key in ["componentTree", "tree", "root"]:
            if key in render:
                self._extract_from_dom_node(render[key], element_ids)

        # Format 4: UI Bridge snapshot
        if "activeStates" in render:
            # This is a state snapshot, not element snapshot
            pass

        return sorted(element_ids)

    def _extract_from_dom_node(self, node: dict[str, Any], element_ids: set[str]) -> None:
        """Recursively extract element IDs from DOM node.

        Args:
            node: DOM node dictionary
            element_ids: Set to add IDs to
        """
        if not isinstance(node, dict):
            return

        # Check attributes for IDs
        attrs = node.get("attributes", {})
        if isinstance(attrs, dict):
            # data-ui-id (UI Bridge registered elements)
            ui_id = attrs.get("data-ui-id")
            if ui_id:
                element_ids.add(f"ui:{ui_id}")

            # data-testid
            testid = attrs.get("data-testid")
            if testid:
                element_ids.add(f"testid:{testid}")

            # id attribute
            html_id = attrs.get("id") or node.get("id")
            if html_id:
                element_ids.add(f"html:{html_id}")

        # Check props for React components
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
                self._extract_from_dom_node(child, element_ids)

    def _parse_element_id(self, elem_id: str) -> tuple[str, str]:
        """Parse element ID into type and name.

        Args:
            elem_id: Full element ID (e.g., "ui:nav-menu")

        Returns:
            Tuple of (type, name)
        """
        for prefix, elem_type in self.config.element_type_prefixes.items():
            if elem_id.startswith(prefix):
                name = elem_id[len(prefix) :]
                return elem_type, name

        return "unknown", elem_id

    def _update_frequencies(self) -> None:
        """Update element frequencies based on render appearances."""
        if self._render_count == 0:
            return

        for elem in self._elements.values():
            elem.frequency = len(elem.render_ids) / self._render_count

    def _find_cooccurrence_groups(self) -> None:
        """Find groups of elements that co-occur across renders."""
        # Filter elements with sufficient occurrences
        frequent_elements = [
            elem_id
            for elem_id, elem in self._elements.items()
            if len(elem.render_ids) >= self.config.min_element_occurrences
        ]

        if not frequent_elements:
            self._element_groups = []
            return

        # Build co-occurrence matrix signature for each element
        # (which renders does each element appear in)
        element_signatures: dict[str, frozenset[str]] = {}
        for elem_id in frequent_elements:
            element_signatures[elem_id] = frozenset(self._elements[elem_id].render_ids)

        # Group elements by exact same render signature (100% co-occurrence)
        signature_groups: dict[frozenset[str], set[str]] = {}
        for elem_id, signature in element_signatures.items():
            if signature not in signature_groups:
                signature_groups[signature] = set()
            signature_groups[signature].add(elem_id)

        # Merge groups with high overlap
        self._element_groups = list(signature_groups.values())

        # Further merge groups with high co-occurrence
        self._merge_similar_groups()

    def _merge_similar_groups(self) -> None:
        """Merge element groups with high co-occurrence."""
        if len(self._element_groups) <= 1:
            return

        threshold = self.config.cooccurrence_threshold
        merged = True

        while merged:
            merged = False
            new_groups: list[set[str]] = []

            for _i, group in enumerate(self._element_groups):
                merged_with_existing = False

                for new_group in new_groups:
                    # Calculate co-occurrence between groups
                    cooc = self._calculate_group_cooccurrence(group, new_group)
                    if cooc >= threshold:
                        new_group.update(group)
                        merged_with_existing = True
                        merged = True
                        break

                if not merged_with_existing:
                    new_groups.append(group.copy())

            self._element_groups = new_groups

    def _calculate_group_cooccurrence(self, group1: set[str], group2: set[str]) -> float:
        """Calculate co-occurrence score between two element groups.

        Args:
            group1: First element group
            group2: Second element group

        Returns:
            Co-occurrence score (0.0-1.0)
        """
        if not group1 or not group2:
            return 0.0

        # Get renders where each group's elements appear
        renders1: set[str] = set()
        for elem_id in group1:
            if elem_id in self._elements:
                renders1.update(self._elements[elem_id].render_ids)

        renders2: set[str] = set()
        for elem_id in group2:
            if elem_id in self._elements:
                renders2.update(self._elements[elem_id].render_ids)

        if not renders1 or not renders2:
            return 0.0

        # Jaccard similarity
        intersection = len(renders1 & renders2)
        union = len(renders1 | renders2)

        return intersection / union if union > 0 else 0.0

    def _generate_states(self) -> None:
        """Generate states from element groups."""
        self._discovered_states = []

        for _i, group in enumerate(self._element_groups):
            # Skip groups that are too small or too large
            if len(group) < self.config.min_state_elements:
                continue
            if len(group) > self.config.max_state_elements:
                continue

            # Generate state ID from elements
            state_id = self._generate_state_id(group)

            # Generate name from first element
            name = self._generate_state_name(group)

            # Get renders where this state appears
            render_ids = self._get_group_renders(group)

            # Calculate confidence based on co-occurrence consistency
            confidence = self._calculate_state_confidence(group)

            # Calculate frequency
            frequency = len(render_ids) / self._render_count if self._render_count > 0 else 0.0

            state = DiscoveredUIState(
                id=state_id,
                name=name,
                element_ids=sorted(group),
                render_ids=sorted(render_ids),
                confidence=confidence,
                frequency=frequency,
            )

            self._discovered_states.append(state)

        logger.debug(f"Generated {len(self._discovered_states)} states from {len(self._element_groups)} groups")

    def _generate_state_id(self, elements: set[str] | frozenset[str]) -> str:
        """Generate deterministic state ID from elements.

        Args:
            elements: Element IDs in the state

        Returns:
            State ID
        """
        sorted_ids = sorted(elements)
        hash_input = "|".join(sorted_ids)
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        return f"state_{hash_value}"

    def _generate_state_name(self, elements: set[str]) -> str:
        """Generate human-readable state name.

        Args:
            elements: Element IDs in the state

        Returns:
            State name
        """
        # Use the first element's name as base
        sorted_elements = sorted(elements)
        if not sorted_elements:
            return "Unknown State"

        first_elem = sorted_elements[0]
        _, name = self._parse_element_id(first_elem)

        # Clean up the name
        name = name.replace("-", " ").replace("_", " ").title()

        if len(elements) > 1:
            name = f"{name} Group ({len(elements)} elements)"

        return name

    def _get_group_renders(self, elements: set[str]) -> set[str]:
        """Get renders where all elements in group appear.

        Args:
            elements: Element group

        Returns:
            Set of render IDs
        """
        if not elements:
            return set()

        # Start with first element's renders
        first_elem = next(iter(elements))
        if first_elem not in self._elements:
            return set()

        common_renders = self._elements[first_elem].render_ids.copy()

        # Intersect with other elements
        for elem_id in elements:
            if elem_id in self._elements:
                common_renders &= self._elements[elem_id].render_ids

        return common_renders

    def _calculate_state_confidence(self, elements: set[str]) -> float:
        """Calculate confidence score for a state.

        Based on how consistently elements co-occur.

        Args:
            elements: Element group

        Returns:
            Confidence score (0.0-1.0)
        """
        if len(elements) <= 1:
            return 1.0

        # Calculate pairwise co-occurrence
        elem_list = list(elements)
        total_cooc = 0.0
        pair_count = 0

        for i, elem1 in enumerate(elem_list):
            for elem2 in elem_list[i + 1 :]:
                cooc = self._calculate_element_cooccurrence(elem1, elem2)
                total_cooc += cooc
                pair_count += 1

        return total_cooc / pair_count if pair_count > 0 else 1.0

    def _calculate_element_cooccurrence(self, elem1: str, elem2: str) -> float:
        """Calculate co-occurrence between two elements.

        Args:
            elem1: First element ID
            elem2: Second element ID

        Returns:
            Co-occurrence score (0.0-1.0)
        """
        if elem1 not in self._elements or elem2 not in self._elements:
            return 0.0

        renders1 = self._elements[elem1].render_ids
        renders2 = self._elements[elem2].render_ids

        if not renders1 or not renders2:
            return 0.0

        intersection = len(renders1 & renders2)
        union = len(renders1 | renders2)

        return intersection / union if union > 0 else 0.0

    def _detect_blocking_states(self) -> None:
        """Detect which states are blocking (modal dialogs, etc.)."""
        for state in self._discovered_states:
            # Check element names for modal indicators
            state_name_lower = state.name.lower()
            element_names_lower = " ".join(state.element_ids).lower()

            for indicator in self.config.modal_indicators:
                if indicator in state_name_lower or indicator in element_names_lower:
                    state.is_blocking = True
                    state.group = "modals"
                    break

            # Detect which states this blocks
            if state.is_blocking:
                state.blocks = self._find_blocked_states(state)

    def _find_blocked_states(self, blocking_state: DiscoveredUIState) -> list[str]:
        """Find states that are blocked when a blocking state is active.

        Args:
            blocking_state: The blocking state

        Returns:
            List of blocked state IDs
        """
        blocked: list[str] = []

        blocking_renders = set(blocking_state.render_ids)

        for other_state in self._discovered_states:
            if other_state.id == blocking_state.id:
                continue

            other_renders = set(other_state.render_ids)

            # Check if other state appears when blocking state appears
            # but disappears or is hidden
            overlap = blocking_renders & other_renders

            # If blocking state always appears with other state, it might block it
            if overlap and len(overlap) < len(other_renders):
                blocked.append(other_state.id)

        return blocked

    def get_discovered_states(self) -> list[DiscoveredUIState]:
        """Get discovered states (runs discovery if needed).

        Returns:
            List of discovered states
        """
        if not self._discovered_states and self._render_count > 0:
            return self.discover_states()
        return self._discovered_states

    def get_element_info(self, element_id: str) -> ElementInfo | None:
        """Get information about an element.

        Args:
            element_id: Element ID

        Returns:
            ElementInfo or None
        """
        return self._elements.get(element_id)

    def get_statistics(self) -> dict[str, Any]:
        """Get discovery statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "renders_processed": self._render_count,
            "unique_elements": len(self._elements),
            "element_groups": len(self._element_groups),
            "discovered_states": len(self._discovered_states),
            "blocking_states": sum(1 for s in self._discovered_states if s.is_blocking),
            "average_elements_per_state": (
                sum(len(s.element_ids) for s in self._discovered_states) / len(self._discovered_states)
                if self._discovered_states
                else 0
            ),
        }

    def reset(self) -> None:
        """Reset discovery state."""
        self._elements.clear()
        self._render_elements.clear()
        self._render_count = 0
        self._discovered_states.clear()
        self._element_groups.clear()
        self._incremental_mode = False

    # =========================================================================
    # Incremental Learning
    # =========================================================================

    def enable_incremental_mode(self) -> None:
        """Enable incremental learning mode.

        In incremental mode, states are learned and updated in real-time
        as renders are processed, rather than in batch.
        """
        self._incremental_mode = True
        logger.info("Incremental learning mode enabled")

    def disable_incremental_mode(self) -> None:
        """Disable incremental learning mode."""
        self._incremental_mode = False
        logger.info("Incremental learning mode disabled")

    def process_render_incremental(
        self, render: dict[str, Any]
    ) -> list[DiscoveredUIState]:
        """Process a single render incrementally.

        This method updates states in real-time without requiring
        full batch discovery.

        Args:
            render: Render log entry from UI Bridge

        Returns:
            List of states that were updated or discovered
        """
        # Extract elements
        element_ids = self.process_render(render)
        element_set = frozenset(element_ids)

        if not element_ids:
            return []

        updated_states: list[DiscoveredUIState] = []
        render_id = render.get("id", f"render_{self._render_count}")

        # Check if this matches any existing state
        best_match: DiscoveredUIState | None = None
        best_similarity = 0.0

        for state in self._discovered_states:
            state_elements = frozenset(state.element_ids)
            similarity = self._calculate_set_similarity(element_set, state_elements)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = state

        # If we have a high similarity match, update it
        if best_match and best_similarity >= self.config.cooccurrence_threshold:
            best_match.increment_observation(render_id)
            best_match.confidence = self._calculate_incremental_confidence(best_match)
            updated_states.append(best_match)

            # Notify callback
            if self.config.on_confidence_updated:
                self.config.on_confidence_updated(best_match.id, best_match.confidence)

        # Check for partial match that should be merged
        elif (
            best_match
            and best_similarity >= self.config.merge_similarity_threshold
            and self.config.auto_merge_similar_states
        ):
            # Merge the new elements into existing state
            best_match.merge_with(
                DiscoveredUIState(
                    id=self._generate_state_id(element_set),
                    name="",
                    element_ids=element_ids,
                    render_ids=[render_id],
                    confidence=0.1,
                )
            )
            best_match.confidence = self._calculate_incremental_confidence(best_match)
            updated_states.append(best_match)

            if self.config.on_state_merged:
                self.config.on_state_merged(
                    best_match.id, self._generate_state_id(element_set)
                )

        # Create new state if no good match
        else:
            new_state = self._create_incremental_state(element_ids, render_id)
            self._discovered_states.append(new_state)
            updated_states.append(new_state)

            if self.config.on_state_discovered:
                self.config.on_state_discovered(new_state)

            # Auto-merge similar states if enabled
            if self.config.auto_merge_similar_states:
                self._auto_merge_similar_states(new_state)

        # Prune if too many states
        if len(self._discovered_states) > self.config.max_discovered_states:
            self._prune_low_confidence_states()

        return updated_states

    def _create_incremental_state(
        self, element_ids: list[str], render_id: str
    ) -> DiscoveredUIState:
        """Create a new state for incremental learning.

        Args:
            element_ids: Element IDs in the state
            render_id: ID of the render that triggered creation

        Returns:
            New DiscoveredUIState
        """
        state_id = self._generate_state_id(frozenset(element_ids))
        name = self._generate_state_name(set(element_ids))
        is_blocking = self._detect_blocking(element_ids)

        return DiscoveredUIState(
            id=state_id,
            name=name,
            element_ids=sorted(element_ids),
            render_ids=[render_id],
            confidence=0.1,  # Start with low confidence
            is_blocking=is_blocking,
            observation_count=1,
            first_observed=datetime.utcnow(),
            last_observed=datetime.utcnow(),
        )

    def _calculate_incremental_confidence(self, state: DiscoveredUIState) -> float:
        """Calculate confidence for incremental learning.

        Uses observation count with logarithmic scaling and recency bonus.

        Args:
            state: State to calculate confidence for

        Returns:
            Confidence score (0.0-1.0)
        """
        import math

        # Observation score (logarithmic scale)
        observation_score = min(1.0, math.log10(state.observation_count + 1) / 2)

        # Recency bonus (decays over 24 hours)
        hours_since_observed = (
            datetime.utcnow() - state.last_observed
        ).total_seconds() / 3600
        recency_score = max(0.0, 1.0 - hours_since_observed / 24)

        # Element consistency (more elements = higher confidence)
        element_score = min(1.0, len(state.element_ids) / 10)

        # Weighted combination
        return (
            observation_score * 0.5
            + recency_score * 0.2
            + element_score * 0.3
        )

    def _calculate_set_similarity(
        self, set1: frozenset[str], set2: frozenset[str]
    ) -> float:
        """Calculate Jaccard similarity between two sets.

        Args:
            set1: First set
            set2: Second set

        Returns:
            Similarity score (0.0-1.0)
        """
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _detect_blocking(self, element_ids: list[str]) -> bool:
        """Detect if element IDs indicate a blocking/modal state.

        Args:
            element_ids: Element IDs to check

        Returns:
            True if blocking state detected
        """
        joined = " ".join(element_ids).lower()
        return any(indicator in joined for indicator in self.config.modal_indicators)

    def _auto_merge_similar_states(self, new_state: DiscoveredUIState) -> None:
        """Automatically merge states that are very similar.

        Args:
            new_state: Newly created state
        """
        threshold = self.config.merge_similarity_threshold
        new_elements = frozenset(new_state.element_ids)

        states_to_remove: list[DiscoveredUIState] = []

        for existing_state in self._discovered_states:
            if existing_state.id == new_state.id:
                continue

            existing_elements = frozenset(existing_state.element_ids)
            similarity = self._calculate_set_similarity(new_elements, existing_elements)

            if similarity >= threshold:
                # Merge into the older state (more observations)
                if existing_state.observation_count >= new_state.observation_count:
                    existing_state.merge_with(new_state)
                    existing_state.confidence = self._calculate_incremental_confidence(
                        existing_state
                    )
                    states_to_remove.append(new_state)

                    if self.config.on_state_merged:
                        self.config.on_state_merged(existing_state.id, new_state.id)
                    break
                else:
                    new_state.merge_with(existing_state)
                    new_state.confidence = self._calculate_incremental_confidence(
                        new_state
                    )
                    states_to_remove.append(existing_state)

                    if self.config.on_state_merged:
                        self.config.on_state_merged(new_state.id, existing_state.id)
                    break

        # Remove merged states
        for state in states_to_remove:
            if state in self._discovered_states:
                self._discovered_states.remove(state)

    def _prune_low_confidence_states(self) -> None:
        """Prune states with low confidence to stay under limit."""
        if len(self._discovered_states) <= self.config.max_discovered_states:
            return

        # Sort by confidence (descending)
        self._discovered_states.sort(key=lambda s: s.confidence, reverse=True)

        # Keep top N
        self._discovered_states = self._discovered_states[
            : self.config.max_discovered_states
        ]

        logger.debug(
            f"Pruned states to {len(self._discovered_states)} "
            f"(max: {self.config.max_discovered_states})"
        )

    def merge_states(self, state_ids: list[str]) -> DiscoveredUIState | None:
        """Manually merge multiple states into one.

        Args:
            state_ids: IDs of states to merge

        Returns:
            Merged state or None if states not found
        """
        states_to_merge = [
            s for s in self._discovered_states if s.id in state_ids
        ]

        if len(states_to_merge) < 2:
            logger.warning(f"Need at least 2 states to merge, found {len(states_to_merge)}")
            return None

        # Merge into first state
        target = states_to_merge[0]
        for source in states_to_merge[1:]:
            target.merge_with(source)
            self._discovered_states.remove(source)

        target.confidence = self._calculate_incremental_confidence(target)
        logger.info(f"Merged {len(states_to_merge)} states into {target.id}")

        return target

    # =========================================================================
    # State Graph
    # =========================================================================

    def get_state_graph(
        self,
        transitions: list[dict[str, Any]] | None = None,
        min_confidence: float | None = None,
    ) -> StateGraph:
        """Get state graph for visualization.

        Args:
            transitions: Optional list of transitions to include
            min_confidence: Minimum confidence for states (uses config default if None)

        Returns:
            StateGraph for visualization
        """
        threshold = min_confidence or self.config.incremental_confidence_threshold
        nodes: list[StateGraphNode] = []
        edges: list[StateGraphEdge] = []

        # Add states as nodes
        for state in self._discovered_states:
            if state.confidence < threshold:
                continue

            nodes.append(
                StateGraphNode(
                    id=state.id,
                    label=state.name,
                    node_type="merged" if state.merged_from else "discovered",
                    observation_count=state.observation_count,
                    confidence=state.confidence,
                    element_count=len(state.element_ids),
                    is_blocking=state.is_blocking,
                )
            )

        # Add transitions as edges
        if transitions:
            for trans in transitions:
                trans_id = trans.get("id", "unknown")
                for from_state in trans.get("fromStates", []):
                    for to_state in trans.get("activateStates", []):
                        edges.append(
                            StateGraphEdge(
                                id=trans_id,
                                source=from_state,
                                target=to_state,
                                label=trans.get("name", trans_id),
                                reliability=trans.get("reliability", 1.0),
                                observation_count=trans.get("observationCount", 0),
                                bidirectional=trans.get("bidirectional", False),
                            )
                        )

        return StateGraph(
            nodes=nodes,
            edges=edges,
            metadata={
                "generatedAt": datetime.utcnow().isoformat(),
                "totalStates": len(self._discovered_states),
                "filteredStates": len(nodes),
                "totalRenders": self._render_count,
                "confidenceThreshold": threshold,
            },
        )

    def export_state_graph(
        self,
        format: str | StateGraphFormat,
        transitions: list[dict[str, Any]] | None = None,
        min_confidence: float | None = None,
    ) -> str:
        """Export state graph in specified format.

        Args:
            format: Export format ("json", "dot", "mermaid" or StateGraphFormat)
            transitions: Optional list of transitions
            min_confidence: Minimum confidence threshold

        Returns:
            Exported graph as string
        """
        graph = self.get_state_graph(transitions, min_confidence)

        if isinstance(format, str):
            format = StateGraphFormat(format.lower())

        if format == StateGraphFormat.JSON:
            return graph.to_json()
        elif format == StateGraphFormat.DOT:
            return graph.to_dot()
        elif format == StateGraphFormat.MERMAID:
            return graph.to_mermaid()
        else:
            raise ValueError(f"Unknown format: {format}")
