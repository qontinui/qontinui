"""Occlusion detector for GUI states in Qontinui.

Detects and manages state occlusions:
- Modal dialogs covering main windows
- Popups blocking underlying UI
- Layered UI components
- Dynamic reveal transitions
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from qontinui.model.state.state import State
from qontinui.model.transition.enhanced_state_transition import StateTransition
from qontinui.state_management.state_memory import StateMemory

from .multistate_adapter import MultiStateAdapter

logger = logging.getLogger(__name__)


class OcclusionType(Enum):
    """Types of state occlusion."""

    MODAL = "modal"  # Modal dialog blocking everything
    SPATIAL = "spatial"  # Spatially overlapping UI
    LOGICAL = "logical"  # Logically exclusive states
    HIERARCHICAL = "hierarchical"  # Parent-child relationships


@dataclass
class OcclusionRelation:
    """Represents an occlusion relationship between states."""

    covering_state_id: int
    covered_state_ids: set[int]
    occlusion_type: OcclusionType
    confidence: float = 1.0
    reversible: bool = True  # Can covered states be revealed?

    def covers(self, state_id: int) -> bool:
        """Check if this relation covers a specific state."""
        return state_id in self.covered_state_ids


@dataclass
class RevealEvent:
    """Event when covered states are revealed."""

    covering_state_id: int
    revealed_state_ids: set[int]
    reveal_transition: StateTransition | None
    timestamp: float = 0.0


class OcclusionDetector:
    """Detects and manages GUI state occlusions in Qontinui.

    This detector identifies when states cover or hide other states,
    and generates appropriate reveal transitions when covering states
    are closed.

    Key features:
    1. Automatic occlusion detection
    2. Multiple occlusion types (modal, spatial, logical)
    3. Dynamic reveal transition generation
    4. Occlusion confidence scoring
    5. Hierarchical state management
    """

    def __init__(
        self,
        state_memory: StateMemory,
        multistate_adapter: MultiStateAdapter | None = None,
    ) -> None:
        """Initialize occlusion detector.

        Args:
            state_memory: Qontinui's state memory
            multistate_adapter: MultiState adapter for occlusion logic
        """
        self.state_memory = state_memory
        self.multistate_adapter = multistate_adapter or MultiStateAdapter(state_memory)

        # Track occlusion relationships
        self.occlusions: dict[int, OcclusionRelation] = {}

        # Track reveal events
        self.reveal_history: list[RevealEvent] = []

        # Configuration
        self.modal_keywords = ["modal", "dialog", "popup", "alert", "confirm", "prompt"]
        self.overlay_keywords = ["overlay", "dropdown", "tooltip", "menu", "submenu"]
        self.confidence_threshold = 0.7

    def detect_occlusions(
        self, active_state_ids: set[int] | None = None
    ) -> list[OcclusionRelation]:
        """Detect occlusion relationships in current active states.

        Args:
            active_state_ids: States to analyze (uses state_memory if None)

        Returns:
            List of detected occlusion relationships
        """
        # Get states to analyze
        state_ids = active_state_ids or self.state_memory.active_states
        if not state_ids:
            return []

        # Clear previous occlusions
        self.occlusions.clear()

        # Detect different types of occlusions
        self._detect_modal_occlusions(state_ids)
        self._detect_spatial_occlusions(state_ids)
        self._detect_logical_occlusions(state_ids)
        self._detect_hierarchical_occlusions(state_ids)

        return list(self.occlusions.values())

    def _detect_modal_occlusions(self, state_ids: set[int]) -> None:
        """Detect modal dialog occlusions.

        Modal dialogs typically block all other states.

        Args:
            state_ids: Active state IDs
        """
        if not self.state_memory.state_service:
            return

        for state_id in state_ids:
            state = self.state_memory.state_service.get_state(state_id)
            if not state:
                continue

            # Check if state is modal
            if self._is_modal_state(state):
                # Modal covers all other active states
                covered = state_ids - {state_id}
                if covered:
                    self.occlusions[state_id] = OcclusionRelation(
                        covering_state_id=state_id,
                        covered_state_ids=covered,
                        occlusion_type=OcclusionType.MODAL,
                        confidence=0.95,
                    )
                    logger.debug(
                        f"Modal state {state.name} covers {len(covered)} states"
                    )

    def _detect_spatial_occlusions(self, state_ids: set[int]) -> None:
        """Detect spatial UI occlusions.

        UI elements that spatially overlap, like dropdowns and tooltips.

        Args:
            state_ids: Active state IDs
        """
        if not self.state_memory.state_service:
            return

        # Check for overlay states
        overlay_states = []
        regular_states = []

        for state_id in state_ids:
            state = self.state_memory.state_service.get_state(state_id)
            if not state:
                continue

            if self._is_overlay_state(state):
                overlay_states.append(state_id)
            else:
                regular_states.append(state_id)

        # Overlays cover regular states (but not other overlays)
        for overlay_id in overlay_states:
            if overlay_id not in self.occlusions:  # Don't override modal
                covered = set(regular_states)
                if covered:
                    self.occlusions[overlay_id] = OcclusionRelation(
                        covering_state_id=overlay_id,
                        covered_state_ids=covered,
                        occlusion_type=OcclusionType.SPATIAL,
                        confidence=0.8,
                    )

    def _detect_logical_occlusions(self, state_ids: set[int]) -> None:
        """Detect logical state exclusions.

        States that are logically exclusive, like different tabs.

        Args:
            state_ids: Active state IDs
        """
        # Use MultiState's occlusion detection
        occlusion_pairs = self.multistate_adapter.detect_occlusions(state_ids)

        for covering_id, covered_id in occlusion_pairs:
            if covering_id in self.occlusions:
                # Add to existing occlusion
                self.occlusions[covering_id].covered_state_ids.add(covered_id)
            else:
                # Create new logical occlusion
                self.occlusions[covering_id] = OcclusionRelation(
                    covering_state_id=covering_id,
                    covered_state_ids={covered_id},
                    occlusion_type=OcclusionType.LOGICAL,
                    confidence=0.9,
                )

    def _detect_hierarchical_occlusions(self, state_ids: set[int]) -> None:
        """Detect hierarchical parent-child occlusions.

        Parent states that contain and potentially hide child states.

        Args:
            state_ids: Active state IDs
        """
        # This would analyze parent-child relationships
        # For now, we use group information as a proxy

        if not hasattr(self.state_memory, "state_groups"):
            return

        # States in same group don't occlude each other
        # But states from different groups might
        pass  # Simplified for now

    def _is_modal_state(self, state: State) -> bool:
        """Check if a state is modal (blocks everything).

        Args:
            state: State to check

        Returns:
            True if state is modal
        """
        state_name_lower = state.name.lower()

        # Check keywords
        for keyword in self.modal_keywords:
            if keyword in state_name_lower:
                return True

        # Check state properties
        if hasattr(state, "blocking") and state.blocking:
            return True

        return False

    def _is_overlay_state(self, state: State) -> bool:
        """Check if a state is an overlay (partial blocking).

        Args:
            state: State to check

        Returns:
            True if state is an overlay
        """
        state_name_lower = state.name.lower()

        for keyword in self.overlay_keywords:
            if keyword in state_name_lower:
                return True

        return False

    def get_covered_states(self, covering_state_id: int) -> set[int]:
        """Get states covered by a specific state.

        Args:
            covering_state_id: State that might be covering others

        Returns:
            Set of covered state IDs
        """
        if covering_state_id in self.occlusions:
            return self.occlusions[covering_state_id].covered_state_ids
        return set()

    def get_covering_state(self, covered_state_id: int) -> int | None:
        """Get state covering a specific state.

        Args:
            covered_state_id: State that might be covered

        Returns:
            ID of covering state, or None
        """
        for covering_id, relation in self.occlusions.items():
            if relation.covers(covered_state_id):
                return covering_id
        return None

    def is_occluded(self, state_id: int) -> bool:
        """Check if a state is currently occluded.

        Args:
            state_id: State to check

        Returns:
            True if state is occluded
        """
        return self.get_covering_state(state_id) is not None

    def generate_reveal_transition(
        self, covering_state_id: int
    ) -> StateTransition | None:
        """Generate transition to reveal states when covering state closes.

        Args:
            covering_state_id: State that is covering others

        Returns:
            Reveal transition or None
        """
        if covering_state_id not in self.occlusions:
            return None

        relation = self.occlusions[covering_state_id]
        if not relation.reversible:
            return None

        # Use MultiState to generate reveal transition
        reveal_trans = self.multistate_adapter.generate_reveal_transition(
            covering_state_id=covering_state_id,
            hidden_state_ids=relation.covered_state_ids,
        )

        if reveal_trans:
            # Record reveal event
            event = RevealEvent(
                covering_state_id=covering_state_id,
                revealed_state_ids=relation.covered_state_ids,
                reveal_transition=reveal_trans,
            )
            self.reveal_history.append(event)
            logger.info(
                f"Generated reveal transition for {len(relation.covered_state_ids)} states"
            )

        return reveal_trans

    def handle_state_closure(self, closing_state_id: int) -> StateTransition | None:
        """Handle when a state is closing, potentially revealing others.

        Args:
            closing_state_id: State that is closing

        Returns:
            Reveal transition if states were revealed, None otherwise
        """
        # Check if closing state was covering others
        if closing_state_id not in self.occlusions:
            return None

        # Generate and return reveal transition
        return self.generate_reveal_transition(closing_state_id)

    def get_visible_states(self) -> set[int]:
        """Get currently visible (non-occluded) states.

        Returns:
            Set of visible state IDs
        """
        active = self.state_memory.active_states
        occluded = self.get_all_occluded_states()
        return cast(set[int], active - occluded)

    def get_all_occluded_states(self) -> set[int]:
        """Get all currently occluded states.

        Returns:
            Set of all occluded state IDs
        """
        occluded = set()
        for relation in self.occlusions.values():
            occluded.update(relation.covered_state_ids)
        return occluded

    def calculate_occlusion_score(
        self, state_id: int, active_states: set[int] | None = None
    ) -> float:
        """Calculate occlusion score for a state.

        Score indicates how much the state occludes others.

        Args:
            state_id: State to score
            active_states: Context states

        Returns:
            Occlusion score (0.0 = no occlusion, 1.0 = complete occlusion)
        """
        if state_id not in self.occlusions:
            return 0.0

        relation = self.occlusions[state_id]
        active = active_states or self.state_memory.active_states

        if not active:
            return 0.0

        # Calculate coverage ratio
        covered_count = len(relation.covered_state_ids & active)
        total_count = len(active) - 1  # Exclude the covering state

        if total_count <= 0:
            return 0.0

        coverage_ratio = covered_count / total_count

        # Weight by occlusion type
        type_weights = {
            OcclusionType.MODAL: 1.0,
            OcclusionType.LOGICAL: 0.9,
            OcclusionType.SPATIAL: 0.7,
            OcclusionType.HIERARCHICAL: 0.5,
        }

        weight = type_weights.get(relation.occlusion_type, 0.5)

        return coverage_ratio * weight * relation.confidence

    def get_occlusion_chain(self, state_id: int) -> list[int]:
        """Get chain of states occluding each other.

        Args:
            state_id: Starting state

        Returns:
            List of state IDs in occlusion chain
        """
        chain = [state_id]
        current = state_id

        # Follow covering states
        while True:
            covering = self.get_covering_state(current)
            if covering and covering not in chain:
                chain.append(covering)
                current = covering
            else:
                break

        return chain

    def get_statistics(self) -> dict[str, Any]:
        """Get occlusion detection statistics.

        Returns:
            Dictionary with statistics
        """
        active = self.state_memory.active_states
        occluded = self.get_all_occluded_states()
        visible = self.get_visible_states()

        occlusion_type_counts: dict[str, int] = {}
        for relation in self.occlusions.values():
            type_name = relation.occlusion_type.value
            occlusion_type_counts[type_name] = (
                occlusion_type_counts.get(type_name, 0) + 1
            )

        return {
            "active_states": len(active),
            "visible_states": len(visible),
            "occluded_states": len(occluded),
            "occlusion_relations": len(self.occlusions),
            "occlusion_types": occlusion_type_counts,
            "reveal_events": len(self.reveal_history),
            "occlusion_rate": len(occluded) / len(active) if active else 0,
        }

    def explain_occlusions(self) -> str:
        """Generate human-readable explanation of current occlusions.

        Returns:
            Explanation string
        """
        if not self.occlusions:
            return "No occlusions detected"

        lines = []
        lines.append(f"Detected {len(self.occlusions)} occlusion relationships:")

        for covering_id, relation in self.occlusions.items():
            if self.state_memory.state_service:
                covering_state = self.state_memory.state_service.get_state(covering_id)
                covering_name = (
                    covering_state.name if covering_state else f"State {covering_id}"
                )
            else:
                covering_name = f"State {covering_id}"

            lines.append(f"\n• {covering_name} ({relation.occlusion_type.value})")
            lines.append(f"  Covers {len(relation.covered_state_ids)} states")
            lines.append(f"  Confidence: {relation.confidence:.1%}")
            lines.append(f"  Reversible: {'Yes' if relation.reversible else 'No'}")

        return "\n".join(lines)
