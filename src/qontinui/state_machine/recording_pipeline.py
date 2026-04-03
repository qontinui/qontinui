"""Recording-to-State-Machine Pipeline.

Orchestrates the conversion of a recording session's CooccurrenceExport
into a fully populated state machine with discovered states, transitions,
and confidence scores.

Pipeline:
1. Load CooccurrenceExport into FingerprintStateDiscovery
2. Discover states via co-occurrence analysis
3. Build transitions from recorded transition records
4. Feed TransitionDetector for confidence scoring
5. Convert to UIBridgeState/UIBridgeTransition
6. Persist to SQLite

Example:
    from qontinui.state_machine import RecordingPipeline, StatePersistence

    persistence = StatePersistence(db_path)
    pipeline = RecordingPipeline(persistence)

    # export_data is the JSON from the SDK's RecordingSessionManager.stop()
    result = pipeline.process_recording(export_data)
    print(f"Discovered {result.state_count} states, {result.transition_count} transitions")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .fingerprint_state_discovery import (
    DiscoveredFingerprintState,
    FingerprintStateDiscovery,
    FingerprintStateDiscoveryConfig,
)
from .fingerprint_types import CooccurrenceExport
from .persistence import StatePersistence
from .transition_detector import DetectedTransition, TransitionDetector
from .ui_bridge_runtime import UIBridgeState, UIBridgeTransition

logger = logging.getLogger(__name__)


@dataclass
class RecordingPipelineConfig:
    """Configuration for the recording pipeline."""

    # State discovery
    min_confidence: float = 0.3
    """Minimum confidence for a transition to be included."""

    treat_header_footer_as_global: bool = True
    """Whether to treat header/footer states as global (always-active)."""

    dedupe_repeating_elements: bool = True
    """Whether to deduplicate repeating list/grid/table items."""

    use_size_weighting: bool = True
    """Whether to weight larger elements more in co-occurrence."""

    auto_detect_modal_states: bool = True
    """Whether to auto-detect modal/dialog states."""

    # Persistence (used when StatePersistence is provided for local/SQLite use)
    persist: bool = False
    """Whether to persist via the local StatePersistence (PostgreSQL persistence is handled at the API layer)."""


@dataclass
class RecordingPipelineResult:
    """Result of processing a recording session."""

    states: list[UIBridgeState]
    transitions: list[UIBridgeTransition]
    detected_transitions: list[DetectedTransition]
    state_count: int
    transition_count: int
    session_id: str = ""
    global_state_count: int = 0
    modal_state_count: int = 0


class RecordingPipeline:
    """Orchestrates recording → state machine conversion.

    Takes a CooccurrenceExport from the UI Bridge SDK's RecordingSessionManager
    and produces a fully populated state machine with discovered states,
    transitions, and confidence scores.
    """

    def __init__(
        self,
        persistence: StatePersistence | None = None,
        config: RecordingPipelineConfig | None = None,
    ) -> None:
        self.persistence = persistence
        self.config = config or RecordingPipelineConfig()

    def process_recording(self, export_data: dict[str, Any]) -> RecordingPipelineResult:
        """Process a recording session export into a state machine.

        Args:
            export_data: JSON dict matching CooccurrenceExport.to_dict() format,
                         as produced by the SDK's RecordingSessionManager.stop().

        Returns:
            RecordingPipelineResult with discovered states and transitions.
        """
        # 1. Parse export
        export = CooccurrenceExport.from_dict(export_data)
        logger.info(
            "Processing recording %s: %d fingerprints, %d captures, %d transitions",
            export.session_id,
            len(export.all_fingerprints),
            len(export.presence_matrix),
            len(export.transitions),
        )

        # 2. Load into FingerprintStateDiscovery
        discovery_config = FingerprintStateDiscoveryConfig(
            treat_header_footer_as_global=self.config.treat_header_footer_as_global,
            dedupe_repeating_elements=self.config.dedupe_repeating_elements,
            use_size_weighting=self.config.use_size_weighting,
            auto_detect_modal_states=self.config.auto_detect_modal_states,
        )
        discovery = FingerprintStateDiscovery(discovery_config)
        discovery.load_cooccurrence_export(export_data)

        # 3. Discover states
        discovered_states = discovery.discover_states()
        logger.info("Discovered %d states", len(discovered_states))

        # 4. Build transitions from fingerprint-level records
        fp_transitions = discovery.build_transitions_from_records()
        logger.info("Built %d fingerprint-level transitions", len(fp_transitions))

        # 5. Feed TransitionDetector for confidence scoring
        detector = TransitionDetector()
        state_index = _build_state_index(discovered_states)

        for record in export.transitions:
            # Map fingerprints to state IDs
            _before_state_ids = _map_fingerprints_to_states(
                record.disappeared_fingerprints, state_index
            )
            _after_state_ids = _map_fingerprints_to_states(record.appeared_fingerprints, state_index)

            # Also include states that didn't change (stable states)
            before_capture = _find_capture(export.presence_matrix, record.before_capture_id)
            after_capture = _find_capture(export.presence_matrix, record.after_capture_id)

            all_before = set()
            all_after = set()
            if before_capture:
                all_before = _map_fingerprints_to_states(before_capture.fingerprints, state_index)
            if after_capture:
                all_after = _map_fingerprints_to_states(after_capture.fingerprints, state_index)

            if all_before or all_after:
                detector.record_action(
                    action={
                        "type": record.action_type,
                        "targetFingerprint": record.target_fingerprint,
                        "actionId": record.action_id,
                    },
                    before_states=all_before,
                    after_states=all_after,
                    success=True,
                )

        # 6. Get confidence-scored transitions
        detected = detector.get_detected_transitions(min_confidence=self.config.min_confidence)
        logger.info(
            "Detected %d transitions with confidence >= %.2f",
            len(detected),
            self.config.min_confidence,
        )

        # 7. Convert to UIBridgeState / UIBridgeTransition
        ui_states = _convert_states(discovered_states)
        ui_transitions = _convert_transitions(detected, discovered_states)

        # 8. Persist
        if self.config.persist and self.persistence:
            for state in ui_states:
                self.persistence.save_state(state)
            for transition in ui_transitions:
                self.persistence.save_transition(transition)
            logger.info(
                "Persisted %d states and %d transitions",
                len(ui_states),
                len(ui_transitions),
            )

        return RecordingPipelineResult(
            states=ui_states,
            transitions=ui_transitions,
            detected_transitions=detected,
            state_count=len(ui_states),
            transition_count=len(ui_transitions),
            session_id=export.session_id,
            global_state_count=sum(1 for s in discovered_states if s.is_global),
            modal_state_count=sum(1 for s in discovered_states if s.is_modal),
        )

    def merge_recording(
        self,
        export_data: dict[str, Any],
        existing_states: list[UIBridgeState],
        existing_transitions: list[UIBridgeTransition],
    ) -> RecordingPipelineResult:
        """Merge a new recording into an existing state machine.

        Matches new states to existing ones by fingerprint overlap,
        updates confidence scores, adds new transitions, and increases
        observation counts for existing transitions.

        Args:
            export_data: New recording's CooccurrenceExport JSON.
            existing_states: Current states from the state machine.
            existing_transitions: Current transitions from the state machine.

        Returns:
            RecordingPipelineResult with merged states and transitions.
        """
        # Run discovery on the new recording
        new_result = self.process_recording(export_data)

        # Merge states
        merged_states: list[UIBridgeState] = list(existing_states)
        state_id_map: dict[str, str] = {}  # new_id -> merged_id

        for new_state in new_result.states:
            # Find best matching existing state by fingerprint overlap
            best_match: UIBridgeState | None = None
            best_overlap = 0.0

            new_fps = set(new_state.element_ids)
            for existing in existing_states:
                existing_fps = set(existing.element_ids)
                union = new_fps | existing_fps
                if not union:
                    continue
                overlap = len(new_fps & existing_fps) / len(union)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = existing

            if best_match and best_overlap >= 0.5:
                # Merge into existing state — union fingerprints, update confidence
                merged_fps = list(set(best_match.element_ids) | new_fps)
                best_match.element_ids = merged_fps

                # Weighted confidence update
                old_conf = best_match.metadata.get("confidence", 0.5)
                new_conf = new_state.metadata.get("confidence", 0.5)
                best_match.metadata["confidence"] = (old_conf + new_conf) / 2
                best_match.metadata["observation_count"] = (
                    best_match.metadata.get("observation_count", 1) + 1
                )

                state_id_map[new_state.id] = best_match.id
                logger.debug(
                    "Merged state %s into %s (overlap=%.2f)",
                    new_state.id,
                    best_match.id,
                    best_overlap,
                )
            else:
                # New state — add to merged set
                merged_states.append(new_state)
                state_id_map[new_state.id] = new_state.id

        # Merge transitions
        existing_transition_sigs: dict[str, UIBridgeTransition] = {}
        for t in existing_transitions:
            sig = _transition_signature(t.from_states, t.activate_states)
            existing_transition_sigs[sig] = t

        merged_transitions: list[UIBridgeTransition] = list(existing_transitions)

        for new_t in new_result.transitions:
            # Remap state IDs through the merge map
            remapped_from = [state_id_map.get(s, s) for s in new_t.from_states]
            remapped_activate = [state_id_map.get(s, s) for s in new_t.activate_states]

            sig = _transition_signature(remapped_from, remapped_activate)

            if sig in existing_transition_sigs:
                # Existing transition — increase confidence and observation count
                existing_t = existing_transition_sigs[sig]
                old_conf = existing_t.metadata.get("confidence", 0.5)
                new_conf = new_t.metadata.get("confidence", 0.5)
                existing_t.metadata["confidence"] = min(1.0, (old_conf + new_conf) / 2 + 0.05)
                existing_t.metadata["observation_count"] = (
                    existing_t.metadata.get("observation_count", 1) + 1
                )
                # Lower cost with higher confidence
                existing_t.path_cost = 1.0 / max(existing_t.metadata["confidence"], 0.1)
            else:
                # New transition — remap and add
                new_t.from_states = remapped_from
                new_t.activate_states = remapped_activate
                new_t.exit_states = [state_id_map.get(s, s) for s in new_t.exit_states]
                merged_transitions.append(new_t)
                existing_transition_sigs[sig] = new_t

        logger.info(
            "Merged recording: %d states (%d new), %d transitions (%d new)",
            len(merged_states),
            len(merged_states) - len(existing_states),
            len(merged_transitions),
            len(merged_transitions) - len(existing_transitions),
        )

        return RecordingPipelineResult(
            states=merged_states,
            transitions=merged_transitions,
            detected_transitions=new_result.detected_transitions,
            state_count=len(merged_states),
            transition_count=len(merged_transitions),
            session_id=new_result.session_id,
            global_state_count=sum(1 for s in merged_states if s.metadata.get("is_global", False)),
            modal_state_count=sum(1 for s in merged_states if s.blocking),
        )


# =============================================================================
# Helpers
# =============================================================================


def _transition_signature(from_states: list[str], activate_states: list[str]) -> str:
    """Create a deterministic signature for a transition for dedup."""
    return f"{','.join(sorted(from_states))}→{','.join(sorted(activate_states))}"


def _build_state_index(
    states: list[DiscoveredFingerprintState],
) -> dict[str, list[str]]:
    """Build a reverse index: fingerprint hash → list of state IDs that contain it."""
    index: dict[str, list[str]] = {}
    for state in states:
        for fp_hash in state.fingerprint_hashes:
            if fp_hash not in index:
                index[fp_hash] = []
            index[fp_hash].append(state.state_id)
    return index


def _map_fingerprints_to_states(
    fingerprints: list[str],
    state_index: dict[str, list[str]],
) -> set[str]:
    """Map a list of fingerprint hashes to the set of state IDs that contain them."""
    state_ids: set[str] = set()
    for fp_hash in fingerprints:
        if fp_hash in state_index:
            state_ids.update(state_index[fp_hash])
    return state_ids


def _find_capture(
    presence_matrix: list[Any],
    capture_id: str,
) -> Any | None:
    """Find a capture record by ID in the presence matrix."""
    for entry in presence_matrix:
        entry_id = entry.capture_id if hasattr(entry, "capture_id") else entry.get("captureId", "")
        if entry_id == capture_id:
            return entry
    return None


def _convert_states(
    discovered: list[DiscoveredFingerprintState],
) -> list[UIBridgeState]:
    """Convert discovered fingerprint states to UIBridgeState objects."""
    result: list[UIBridgeState] = []

    for state in discovered:
        ui_state = UIBridgeState(
            id=state.state_id,
            name=state.name,
            element_ids=list(state.fingerprint_hashes),
            blocking=state.is_modal,
            group=state.position_zone if state.is_global else None,
            metadata={
                "confidence": state.confidence,
                "is_global": state.is_global,
                "position_zone": state.position_zone,
                "source": "recording",
            },
        )
        result.append(ui_state)

    return result


def _convert_transitions(
    detected: list[DetectedTransition],
    discovered_states: list[DiscoveredFingerprintState],
) -> list[UIBridgeTransition]:
    """Convert detected transitions to UIBridgeTransition objects."""
    # Build a set of valid state IDs for filtering
    valid_state_ids = {s.state_id for s in discovered_states}
    result: list[UIBridgeTransition] = []

    for dt in detected:
        # Filter to valid states
        from_states = [s for s in dt.from_states if s in valid_state_ids]
        to_states = [s for s in dt.to_states if s in valid_state_ids]
        activate = [s for s in dt.activate_states if s in valid_state_ids]
        exit_states = [s for s in dt.exit_states if s in valid_state_ids]

        if not from_states and not to_states:
            continue

        # Build action from detected transition data
        actions: list[dict[str, Any]] = []
        if dt.actions:
            actions = list(dt.actions)

        transition = UIBridgeTransition(
            id=dt.id,
            name=dt.name,
            from_states=from_states,
            activate_states=activate if activate else to_states,
            exit_states=exit_states,
            actions=actions,
            path_cost=1.0 / max(dt.confidence, 0.1),  # Higher confidence = lower cost
            metadata={
                "confidence": dt.confidence,
                "observation_count": dt.observation_count,
                "is_bidirectional": dt.is_bidirectional,
                "source": "recording",
            },
        )
        result.append(transition)

    return result
