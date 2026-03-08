"""Fingerprint-enhanced state discovery strategy.

This strategy uses element fingerprints from the UI Bridge for enhanced
state discovery with cross-page element matching, position-aware grouping,
and semantic identification.

When fingerprint data is not available, it falls back to synthesizing
fingerprints from element IDs (bridge registry IDs, data-testid, HTML id)
extracted from render log entries, then feeds them through the same
co-occurrence analysis pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from qontinui.state_machine.fingerprint_state_discovery import (
    FingerprintStateDiscovery,
    FingerprintStateDiscoveryConfig,
)

from ..base import (
    DiscoveredElement,
    DiscoveredState,
    DiscoveredTransition,
    DiscoveryStrategyType,
    StateDiscoveryInput,
    StateDiscoveryResult,
    StateDiscoveryStrategy,
)

logger = logging.getLogger(__name__)


class FingerprintStrategy(StateDiscoveryStrategy):
    """Fingerprint-enhanced state discovery.

    This strategy uses element fingerprints computed by the UI Bridge
    for enhanced state discovery with:
    - Cross-page element matching via structural fingerprints
    - Position zone classification (header/footer = global, modal = blocking)
    - Repeat pattern deduplication (list items grouped)
    - Size-weighted co-occurrence scoring
    - Semantic name matching

    When fingerprint data is not available but render logs with element IDs
    are provided, it synthesizes fingerprints from element IDs and feeds
    them into the same co-occurrence pipeline.
    """

    def __init__(self, config: FingerprintStateDiscoveryConfig | None = None) -> None:
        """Initialize with optional configuration.

        Args:
            config: Configuration for fingerprint discovery
        """
        self._config = config
        self._discovery: FingerprintStateDiscovery | None = None

    @property
    def strategy_type(self) -> DiscoveryStrategyType:
        return DiscoveryStrategyType.FINGERPRINT

    def can_process(self, input_data: StateDiscoveryInput) -> bool:
        """Check if this strategy can process the input.

        Handles both fingerprint data and render data with element IDs.
        """
        return input_data.has_fingerprint_data() or input_data.has_render_data()

    def discover(self, input_data: StateDiscoveryInput) -> StateDiscoveryResult:
        """Discover states using fingerprint-enhanced analysis.

        If fingerprint data (cooccurrence_export) is available, uses it directly.
        Otherwise, falls back to synthesizing fingerprints from element IDs
        in render log entries.

        Args:
            input_data: Input containing cooccurrence_export with fingerprints,
                        or render log entries with element IDs

        Returns:
            Discovery result with enhanced state information
        """
        if input_data.has_fingerprint_data() and input_data.cooccurrence_export:
            return self._discover_from_fingerprints(input_data.cooccurrence_export)

        if input_data.has_render_data():
            return self._discover_from_element_ids(input_data)

        logger.warning("No fingerprint or render data provided for fingerprint strategy")
        return StateDiscoveryResult(
            states=[],
            elements=[],
            element_to_renders={},
            render_count=0,
            unique_element_count=0,
            strategy_used=self.strategy_type,
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get discovery statistics."""
        if self._discovery:
            return self._discovery.get_statistics()
        return {}

    # =========================================================================
    # Fingerprint-based discovery (primary path)
    # =========================================================================

    def _discover_from_fingerprints(
        self, cooccurrence_export: dict[str, Any]
    ) -> StateDiscoveryResult:
        """Discover states from fingerprint co-occurrence export.

        Args:
            cooccurrence_export: Export data from UI Bridge with fingerprints

        Returns:
            Discovery result with enhanced state information
        """
        # Initialize discovery with config
        self._discovery = FingerprintStateDiscovery(self._config)

        # Load the export data
        self._discovery.load_cooccurrence_export(cooccurrence_export)

        # Run discovery
        fp_states = self._discovery.discover_states()

        # Convert to unified types
        elements = self._build_element_list(cooccurrence_export)
        states = self._convert_states(fp_states)
        transitions = self._build_transitions()

        # Build element to renders mapping
        element_to_renders = self._build_element_render_mapping(cooccurrence_export)

        # Get statistics
        stats = self._discovery.get_statistics()

        logger.info(
            f"Fingerprint strategy discovered {len(states)} states "
            f"from {stats.get('total_captures', 0)} captures "
            f"with {stats.get('total_fingerprints', 0)} fingerprints"
        )

        return StateDiscoveryResult(
            states=states,
            elements=elements,
            transitions=transitions,
            element_to_renders=element_to_renders,
            render_count=stats.get("total_captures", 0),
            unique_element_count=stats.get("total_fingerprints", 0),
            strategy_used=self.strategy_type,
            strategy_metadata={
                "algorithm": "fingerprint_cooccurrence",
                "global_fingerprints": stats.get("global_fingerprints", 0),
                "modal_states": stats.get("modal_states", 0),
            },
        )

    # =========================================================================
    # Element ID fallback (when no fingerprint data available)
    # =========================================================================

    def _discover_from_element_ids(self, input_data: StateDiscoveryInput) -> StateDiscoveryResult:
        """Discover states by synthesizing fingerprints from element IDs.

        Extracts element IDs from render log entries, creates synthetic
        ElementFingerprint objects, builds a presence matrix, and feeds
        everything into the standard fingerprint co-occurrence pipeline.

        Args:
            input_data: Input containing render log entries

        Returns:
            Discovery result
        """
        renders = input_data.renders
        include_html_ids = input_data.include_html_ids

        if not renders:
            return StateDiscoveryResult(
                states=[],
                elements=[],
                element_to_renders={},
                render_count=0,
                unique_element_count=0,
                strategy_used=self.strategy_type,
            )

        # Step 1: Extract element IDs from each render
        render_element_ids: list[tuple[str, list[str]]] = []
        all_element_ids: set[str] = set()

        for i, render in enumerate(renders):
            render_id = render.get("id", f"render_{i}")
            elem_ids = self._extract_elements_from_render(render, include_html_ids)
            render_element_ids.append((render_id, elem_ids))
            all_element_ids.update(elem_ids)

        if not all_element_ids:
            return StateDiscoveryResult(
                states=[],
                elements=[],
                element_to_renders={},
                render_count=len(renders),
                unique_element_count=0,
                strategy_used=self.strategy_type,
                strategy_metadata={"algorithm": "id_fallback"},
            )

        # Step 2: Synthesize a cooccurrence export from element IDs
        synthetic_export = self._synthesize_cooccurrence_export(render_element_ids, all_element_ids)

        # Step 3: Feed through the standard fingerprint pipeline
        self._discovery = FingerprintStateDiscovery(self._config)
        self._discovery.load_cooccurrence_export(synthetic_export)
        fp_states = self._discovery.discover_states()

        # Step 4: Convert to unified types
        elements = self._build_id_element_list(all_element_ids, render_element_ids)
        states = self._convert_states(fp_states)

        # Build element to renders mapping
        element_to_renders: dict[str, list[str]] = {}
        for elem_id in all_element_ids:
            render_ids = sorted(rid for rid, eids in render_element_ids if elem_id in eids)
            if render_ids:
                element_to_renders[f"fp:{elem_id[:12]}"] = render_ids

        stats = self._discovery.get_statistics()

        logger.info(
            f"Fingerprint strategy (ID fallback) discovered {len(states)} states "
            f"from {len(renders)} renders with {len(all_element_ids)} elements"
        )

        return StateDiscoveryResult(
            states=states,
            elements=elements,
            element_to_renders=element_to_renders,
            render_count=len(renders),
            unique_element_count=len(all_element_ids),
            strategy_used=self.strategy_type,
            strategy_metadata={
                "algorithm": "id_fallback",
                "global_fingerprints": stats.get("global_fingerprints", 0),
                "modal_states": stats.get("modal_states", 0),
            },
        )

    def _extract_elements_from_render(
        self,
        render_log_entry: dict[str, Any],
        include_html_ids: bool = False,
    ) -> list[str]:
        """Extract element IDs from a UI Bridge render log entry.

        Supports multiple render formats:
        - DomSnapshotRenderLogEntry (type: dom_snapshot)
        - Simple format with elements list
        - Component tree format (componentTree or tree key)

        Args:
            render_log_entry: A render log entry dict
            include_html_ids: Whether to include HTML id attributes

        Returns:
            Sorted list of element ID strings
        """
        element_ids: set[str] = set()

        # Format 1: DomSnapshotRenderLogEntry (from qontinui-web)
        if render_log_entry.get("type") == "dom_snapshot":
            snapshot = render_log_entry.get("snapshot", {})
            root = snapshot.get("root")
            if root:
                self._extract_from_dom_node(root, element_ids, include_html_ids)
            return sorted(element_ids)

        # Format 2: Simple format (for testing and backward compatibility)
        if "elements" in render_log_entry:
            for elem in render_log_entry["elements"]:
                elem_id = elem.get("id")
                if elem_id:
                    element_ids.add(f"reg:{elem_id}")

        # Extract from componentTree (simple format)
        if "componentTree" in render_log_entry:
            self._extract_from_dom_node(
                render_log_entry["componentTree"], element_ids, include_html_ids
            )

        # Also check "tree" key (alternative simple format)
        if "tree" in render_log_entry:
            self._extract_from_dom_node(render_log_entry["tree"], element_ids, include_html_ids)

        return sorted(element_ids)

    def _extract_from_dom_node(
        self,
        node: dict[str, Any],
        element_ids: set[str],
        include_html_ids: bool = False,
    ) -> None:
        """Recursively extract element IDs from a DOM node.

        Args:
            node: DOM node dict
            element_ids: Set to collect IDs into
            include_html_ids: Whether to include HTML id attributes
        """
        if not isinstance(node, dict):
            return

        # Priority 1: Element registry ID (from bridge registry 'id' field)
        reg_id = node.get("id")
        if reg_id and isinstance(reg_id, str):
            element_ids.add(f"ui:{reg_id}")

        # Get attributes dict
        attrs = node.get("attributes", {})
        if isinstance(attrs, dict):
            # Priority 2: data-testid (testing convention)
            testid = attrs.get("data-testid")
            if testid:
                element_ids.add(f"testid:{testid}")

            # Optional: HTML id attribute
            if include_html_ids:
                html_id = attrs.get("id")
                if html_id:
                    element_ids.add(f"html:{html_id}")

        # Also check props (for React component tree format)
        props = node.get("props", {})
        if isinstance(props, dict):
            testid = props.get("data-testid")
            if testid:
                element_ids.add(f"testid:{testid}")

        # Recurse into children
        children = node.get("children", [])
        if isinstance(children, list):
            for child in children:
                self._extract_from_dom_node(child, element_ids, include_html_ids)

    def _synthesize_cooccurrence_export(
        self,
        render_element_ids: list[tuple[str, list[str]]],
        all_element_ids: set[str],
    ) -> dict[str, Any]:
        """Synthesize a cooccurrence export dict from element IDs.

        Creates synthetic ElementFingerprint objects and a presence matrix
        that can be loaded by FingerprintStateDiscovery.

        Args:
            render_element_ids: List of (render_id, element_ids) tuples
            all_element_ids: Set of all unique element IDs

        Returns:
            Dict matching CooccurrenceExport.from_dict() format
        """
        # Build fingerprint details: use the element ID as the hash
        fingerprint_details: dict[str, dict[str, Any]] = {}
        for elem_id in all_element_ids:
            # Clean up the name (remove prefix like "ui:", "testid:", etc.)
            clean_name = elem_id.split(":", 1)[1] if ":" in elem_id else elem_id
            fingerprint_details[elem_id] = {
                "hash": elem_id,
                "structuralPath": "",
                "positionZone": "main",
                "landmarkContext": "",
                "role": "",
                "tagName": "",
                "accessibleName": clean_name,
                "sizeCategory": "medium",
                "relativePosition": {},
                "isRepeating": False,
            }

        # Build presence matrix
        presence_matrix: list[dict[str, Any]] = []
        for render_id, elem_ids in render_element_ids:
            presence_matrix.append(
                {
                    "captureId": render_id,
                    "url": "",
                    "fingerprints": elem_ids,
                }
            )

        # Build fingerprint stats
        fingerprint_stats: dict[str, dict[str, Any]] = {}
        for elem_id in all_element_ids:
            capture_ids = [rid for rid, eids in render_element_ids if elem_id in eids]
            fingerprint_stats[elem_id] = {
                "totalAppearances": len(capture_ids),
                "captureIds": capture_ids,
                "firstSeen": 0,
                "lastSeen": 0,
            }

        # Build cooccurrence counts
        cooccurrence_counts: dict[str, dict[str, int]] = {}
        for _render_id, elem_ids in render_element_ids:
            for i, id1 in enumerate(elem_ids):
                if id1 not in cooccurrence_counts:
                    cooccurrence_counts[id1] = {}
                for id2 in elem_ids[i + 1 :]:
                    if id2 not in cooccurrence_counts:
                        cooccurrence_counts[id2] = {}
                    cooccurrence_counts[id1][id2] = cooccurrence_counts[id1].get(id2, 0) + 1
                    cooccurrence_counts[id2][id1] = cooccurrence_counts[id2].get(id1, 0) + 1

        return {
            "sessionId": "synthetic-from-element-ids",
            "exportedAt": 0,
            "allFingerprints": sorted(all_element_ids),
            "fingerprintDetails": fingerprint_details,
            "presenceMatrix": presence_matrix,
            "cooccurrenceCounts": cooccurrence_counts,
            "fingerprintStats": fingerprint_stats,
            "transitions": [],
            "stateCandidates": [],  # Let the pipeline compute them
        }

    def _build_id_element_list(
        self,
        all_element_ids: set[str],
        render_element_ids: list[tuple[str, list[str]]],
    ) -> list[DiscoveredElement]:
        """Build element list from synthesized element IDs.

        Args:
            all_element_ids: All unique element IDs
            render_element_ids: Mapping of renders to their element IDs

        Returns:
            List of DiscoveredElement objects
        """
        elements: list[DiscoveredElement] = []

        for elem_id in sorted(all_element_ids):
            # Determine element type from prefix
            if elem_id.startswith("ui:"):
                elem_type = "ui-id"
            elif elem_id.startswith("testid:"):
                elem_type = "testid"
            elif elem_id.startswith("html:"):
                elem_type = "html-id"
            elif elem_id.startswith("reg:"):
                elem_type = "registered"
            else:
                elem_type = "unknown"

            clean_name = elem_id.split(":", 1)[1] if ":" in elem_id else elem_id

            render_ids = sorted(rid for rid, eids in render_element_ids if elem_id in eids)

            elements.append(
                DiscoveredElement(
                    id=f"fp:{elem_id[:12]}",
                    name=clean_name,
                    element_type=elem_type,
                    render_ids=render_ids,
                    fingerprint_hash=elem_id,
                    position_zone="main",
                    size_category="medium",
                )
            )

        return elements

    # =========================================================================
    # Shared conversion helpers
    # =========================================================================

    def _build_element_list(self, export_data: dict[str, Any]) -> list[DiscoveredElement]:
        """Build element list from fingerprint details."""
        elements: list[DiscoveredElement] = []

        fingerprint_details = export_data.get("fingerprintDetails", {})
        fingerprint_stats = export_data.get("fingerprintStats", {})

        for fp_hash, fp_data in fingerprint_details.items():
            # Get render IDs from stats
            stats = fingerprint_stats.get(fp_hash, {})
            render_ids = stats.get("captureIds", [])

            elements.append(
                DiscoveredElement(
                    id=f"fp:{fp_hash[:12]}",  # Shortened fingerprint hash as ID
                    name=fp_data.get("accessibleName") or fp_data.get("role") or fp_hash[:12],
                    element_type="fingerprint",
                    render_ids=render_ids,
                    fingerprint_hash=fp_hash,
                    position_zone=fp_data.get("positionZone"),
                    landmark_context=fp_data.get("landmarkContext"),
                    role=fp_data.get("role"),
                    accessible_name=fp_data.get("accessibleName"),
                    size_category=fp_data.get("sizeCategory"),
                    is_repeating=fp_data.get("isRepeating", False),
                    tag_name=fp_data.get("tagName"),
                )
            )

        return elements

    def _convert_states(self, fp_states: list[Any]) -> list[DiscoveredState]:
        """Convert fingerprint states to unified format."""
        unified_states: list[DiscoveredState] = []

        for fp_state in fp_states:
            # Convert fingerprint hashes to element IDs
            element_ids = [f"fp:{h[:12]}" for h in fp_state.fingerprint_hashes]

            unified_states.append(
                DiscoveredState(
                    id=fp_state.state_id,
                    name=fp_state.name,
                    element_ids=element_ids,
                    render_ids=[],  # Not directly available from fingerprint states
                    confidence=fp_state.confidence,
                    position_zone=fp_state.position_zone,
                    landmark_context=fp_state.landmark_context,
                    is_global=fp_state.is_global,
                    is_modal=fp_state.is_modal,
                    observation_count=fp_state.observation_count,
                    first_observed=fp_state.first_observed,
                    last_observed=fp_state.last_observed,
                    metadata=fp_state.metadata,
                )
            )

        return unified_states

    def _build_transitions(self) -> list[DiscoveredTransition]:
        """Build transitions from recorded transition data."""
        if not self._discovery:
            return []

        transitions: list[DiscoveredTransition] = []
        transition_dicts = self._discovery.build_transitions_from_records()

        for i, trans in enumerate(transition_dicts):
            transitions.append(
                DiscoveredTransition(
                    id=trans.get("id", f"trans_{i}"),
                    name=trans.get("name", ""),
                    action_type=trans.get("actionType", ""),
                    from_state_ids=trans.get("fromStates", []),
                    to_state_ids=trans.get("activateStates", []),
                    trigger_element_id=trans.get("triggerElement"),
                    timestamp=trans.get("timestamp", 0),
                )
            )

        return transitions

    def _build_element_render_mapping(self, export_data: dict[str, Any]) -> dict[str, list[str]]:
        """Build element to renders mapping from presence matrix."""
        element_to_renders: dict[str, list[str]] = {}

        fingerprint_stats = export_data.get("fingerprintStats", {})

        for fp_hash, stats in fingerprint_stats.items():
            element_id = f"fp:{fp_hash[:12]}"
            element_to_renders[element_id] = stats.get("captureIds", [])

        return element_to_renders
