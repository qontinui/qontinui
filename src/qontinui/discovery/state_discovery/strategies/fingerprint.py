"""Fingerprint-enhanced state discovery strategy.

This strategy uses element fingerprints from the UI Bridge for enhanced
state discovery with cross-page element matching, position-aware grouping,
and semantic identification.
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
        """Check if fingerprint data is available."""
        return input_data.has_fingerprint_data()

    def discover(self, input_data: StateDiscoveryInput) -> StateDiscoveryResult:
        """Discover states using fingerprint-enhanced analysis.

        Args:
            input_data: Input containing cooccurrence_export with fingerprints

        Returns:
            Discovery result with enhanced state information
        """
        if not input_data.cooccurrence_export:
            logger.warning("No cooccurrence export provided for fingerprint strategy")
            return StateDiscoveryResult(
                states=[],
                elements=[],
                element_to_renders={},
                render_count=0,
                unique_element_count=0,
                strategy_used=self.strategy_type,
            )

        # Initialize discovery with config
        self._discovery = FingerprintStateDiscovery(self._config)

        # Load the export data
        self._discovery.load_cooccurrence_export(input_data.cooccurrence_export)

        # Run discovery
        fp_states = self._discovery.discover_states()

        # Convert to unified types
        elements = self._build_element_list(input_data.cooccurrence_export)
        states = self._convert_states(fp_states)
        transitions = self._build_transitions()

        # Build element to renders mapping
        element_to_renders = self._build_element_render_mapping(input_data.cooccurrence_export)

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

    def get_statistics(self) -> dict[str, Any]:
        """Get discovery statistics."""
        if self._discovery:
            return self._discovery.get_statistics()
        return {}

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
