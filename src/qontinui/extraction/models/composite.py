"""
State structure models for GUI environments.

A StateStructure represents a collection of states and transitions that may form
one or more disjoint state trees. Whether these trees come from separate applications
or from non-overlapping functionalities within a single application is irrelevant
to the model - it's simply a state structure with potentially disjoint subgraphs.

Each state tracks its origin (source_id) so that when a source is re-extracted,
only the states from that source are replaced while states from other sources
remain unchanged.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .base import Screenshot, Viewport
from .correlated import CorrelatedState, ExtractionResult, InferredTransition, VerifiedTransition
from .runtime import ExtractedElement


@dataclass
class StateStructure:
    """
    A collection of states and transitions forming one or more state graphs.

    A StateStructure may contain:
    - A single connected state graph (typical single-app extraction)
    - Multiple disjoint state graphs (multi-app environment, or app with separate features)
    - Any combination of connected and disjoint states

    The structure tracks:
    - All states (which may form disjoint trees)
    - All transitions (which only connect states within the same tree)
    - Currently active states (can span multiple trees simultaneously)
    - Origin of each state (for replacement during re-extraction)

    Example: A desktop environment with a browser and terminal would have two
    disjoint state trees. The browser's "login-page" and terminal's "command-prompt"
    could both be active simultaneously, but there are no transitions between them.
    When the browser app is re-extracted, only browser states are replaced.
    """

    id: str
    """Unique identifier for this state structure."""

    name: str
    """Human-readable name for this structure."""

    # Core state data
    states: list[CorrelatedState] = field(default_factory=list)
    """All states in this structure (may form disjoint trees)."""

    transitions: list[InferredTransition | VerifiedTransition] = field(default_factory=list)
    """All transitions (connect states within the same tree)."""

    elements: list[ExtractedElement] = field(default_factory=list)
    """All extracted elements."""

    screenshots: dict[str, Screenshot] = field(default_factory=dict)
    """Screenshots by ID."""

    # Active state tracking
    _active_state_ids: set[str] = field(default_factory=set)
    """Set of currently active state IDs."""

    # Origin tracking for replacement operations
    state_origins: dict[str, str] = field(default_factory=dict)
    """Maps state_id -> source_id (origin identifier for replacement)."""

    transition_origins: dict[str, str] = field(default_factory=dict)
    """Maps transition_id -> source_id (origin identifier for replacement)."""

    element_origins: dict[str, str] = field(default_factory=dict)
    """Maps element_id -> source_id (origin identifier for replacement)."""

    # Environment info
    viewport: Viewport = field(default_factory=lambda: Viewport(width=1920, height=1080))
    """Environment viewport/screen size."""

    created_at: datetime = field(default_factory=datetime.now)
    """When this structure was created."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    # =========================================================================
    # State Management
    # =========================================================================

    def add_state(self, state: CorrelatedState, source_id: str | None = None) -> None:
        """
        Add a state to the structure.

        Args:
            state: The state to add
            source_id: Origin identifier (e.g., app ID) for replacement operations
        """
        # Check for duplicate
        if any(s.id == state.id for s in self.states):
            raise ValueError(f"State with ID '{state.id}' already exists")

        self.states.append(state)

        if source_id:
            self.state_origins[state.id] = source_id

    def remove_state(self, state_id: str) -> None:
        """Remove a state and its associated transitions."""
        self.states = [s for s in self.states if s.id != state_id]
        self.transitions = [
            t
            for t in self.transitions
            if not (
                (hasattr(t, "state_before") and t.state_before == state_id)
                or (hasattr(t, "state_after") and t.state_after == state_id)
            )
        ]
        self.state_origins.pop(state_id, None)
        self._active_state_ids.discard(state_id)

    def get_state(self, state_id: str) -> CorrelatedState | None:
        """Get a state by ID."""
        for state in self.states:
            if state.id == state_id:
                return state
        return None

    def get_states_by_source(self, source_id: str) -> list[CorrelatedState]:
        """Get all states from a specific source."""
        state_ids = {sid for sid, src in self.state_origins.items() if src == source_id}
        return [s for s in self.states if s.id in state_ids]

    def get_sources(self) -> set[str]:
        """Get all unique source IDs in this structure."""
        return set(self.state_origins.values())

    # =========================================================================
    # Transition Management
    # =========================================================================

    def add_transition(
        self,
        transition: InferredTransition | VerifiedTransition,
        source_id: str | None = None,
    ) -> None:
        """Add a transition to the structure."""
        if any(t.id == transition.id for t in self.transitions):
            raise ValueError(f"Transition with ID '{transition.id}' already exists")
        self.transitions.append(transition)

        if source_id:
            self.transition_origins[transition.id] = source_id

    def remove_transition(self, transition_id: str) -> None:
        """Remove a transition."""
        self.transitions = [t for t in self.transitions if t.id != transition_id]
        self.transition_origins.pop(transition_id, None)

    def get_transitions_from_state(
        self, state_id: str
    ) -> list[InferredTransition | VerifiedTransition]:
        """Get all transitions originating from a state."""
        return [
            t for t in self.transitions if hasattr(t, "state_before") and t.state_before == state_id
        ]

    def get_transitions_to_state(
        self, state_id: str
    ) -> list[InferredTransition | VerifiedTransition]:
        """Get all transitions leading to a state."""
        return [
            t for t in self.transitions if hasattr(t, "state_after") and t.state_after == state_id
        ]

    def get_transitions_by_source(
        self, source_id: str
    ) -> list[InferredTransition | VerifiedTransition]:
        """Get all transitions from a specific source."""
        transition_ids = {tid for tid, src in self.transition_origins.items() if src == source_id}
        return [t for t in self.transitions if t.id in transition_ids]

    # =========================================================================
    # Element Management
    # =========================================================================

    def add_element(self, element: ExtractedElement, source_id: str | None = None) -> None:
        """Add an element to the structure."""
        if any(e.id == element.id for e in self.elements):
            raise ValueError(f"Element with ID '{element.id}' already exists")
        self.elements.append(element)

        if source_id:
            self.element_origins[element.id] = source_id

    def remove_element(self, element_id: str) -> None:
        """Remove an element."""
        self.elements = [e for e in self.elements if e.id != element_id]
        self.element_origins.pop(element_id, None)

    def get_elements_by_source(self, source_id: str) -> list[ExtractedElement]:
        """Get all elements from a specific source."""
        element_ids = {eid for eid, src in self.element_origins.items() if src == source_id}
        return [e for e in self.elements if e.id in element_ids]

    # =========================================================================
    # Source Replacement (for re-extraction)
    # =========================================================================

    def replace_source(self, source_id: str, new_result: "ExtractionResult") -> None:
        """
        Replace all states, transitions, and elements from a source with new extraction results.

        This is the key operation for updating part of a state structure when an
        application or feature is re-extracted.

        Args:
            source_id: The source identifier to replace
            new_result: The new extraction result to insert
        """
        # Remove all existing data from this source
        self.remove_source(source_id)

        # Add new states
        all_states = new_result.correlated_states.copy()
        if new_result.inferred_states:
            all_states.extend(new_result.inferred_states)

        for state in all_states:
            self.states.append(state)
            self.state_origins[state.id] = source_id
            if state.screenshot:
                self.screenshots[state.screenshot.id] = state.screenshot

        # Add new transitions
        for verified in new_result.verified_transitions:
            self.transitions.append(verified)
            self.transition_origins[verified.id] = source_id

        for inferred in new_result.inferred_transitions:
            self.transitions.append(inferred)
            self.transition_origins[inferred.id] = source_id

        # Add elements from runtime extraction if available
        if new_result.runtime_extraction and hasattr(new_result.runtime_extraction, "states"):
            for runtime_state in new_result.runtime_extraction.states:
                if hasattr(runtime_state, "elements"):
                    for element in runtime_state.elements:
                        self.elements.append(element)
                        self.element_origins[element.id] = source_id

    def remove_source(self, source_id: str) -> None:
        """
        Remove all states, transitions, and elements from a specific source.

        Args:
            source_id: The source identifier to remove
        """
        # Get IDs to remove
        state_ids_to_remove = {sid for sid, src in self.state_origins.items() if src == source_id}
        transition_ids_to_remove = {
            tid for tid, src in self.transition_origins.items() if src == source_id
        }
        element_ids_to_remove = {
            eid for eid, src in self.element_origins.items() if src == source_id
        }

        # Remove states and their screenshots
        for state in self.states:
            if state.id in state_ids_to_remove and state.screenshot:
                self.screenshots.pop(state.screenshot.id, None)

        self.states = [s for s in self.states if s.id not in state_ids_to_remove]
        self.transitions = [t for t in self.transitions if t.id not in transition_ids_to_remove]
        self.elements = [e for e in self.elements if e.id not in element_ids_to_remove]

        # Clean up origin maps
        for sid in state_ids_to_remove:
            self.state_origins.pop(sid, None)
            self._active_state_ids.discard(sid)

        for tid in transition_ids_to_remove:
            self.transition_origins.pop(tid, None)

        for eid in element_ids_to_remove:
            self.element_origins.pop(eid, None)

    # =========================================================================
    # Active State Management
    # =========================================================================

    def get_active_states(self) -> list[CorrelatedState]:
        """Get all currently active states."""
        return [s for s in self.states if s.id in self._active_state_ids]

    def get_active_state_ids(self) -> set[str]:
        """Get the set of active state IDs."""
        return self._active_state_ids.copy()

    def set_active_states(self, state_ids: set[str]) -> None:
        """Set the active states (replaces current)."""
        # Validate all state IDs exist
        existing_ids = {s.id for s in self.states}
        invalid = state_ids - existing_ids
        if invalid:
            raise ValueError(f"States not found: {invalid}")
        self._active_state_ids = state_ids.copy()

    def activate_state(self, state_id: str) -> None:
        """Activate a state."""
        if not any(s.id == state_id for s in self.states):
            raise ValueError(f"State '{state_id}' does not exist")
        self._active_state_ids.add(state_id)

    def deactivate_state(self, state_id: str) -> None:
        """Deactivate a state."""
        self._active_state_ids.discard(state_id)

    def is_state_active(self, state_id: str) -> bool:
        """Check if a state is active."""
        return state_id in self._active_state_ids

    def clear_active_states(self) -> None:
        """Clear all active states."""
        self._active_state_ids.clear()

    def get_active_states_by_source(self, source_id: str) -> list[CorrelatedState]:
        """Get active states from a specific source."""
        source_state_ids = {sid for sid, src in self.state_origins.items() if src == source_id}
        active_in_source = self._active_state_ids & source_state_ids
        return [s for s in self.states if s.id in active_in_source]

    # =========================================================================
    # Graph Analysis
    # =========================================================================

    def get_connected_components(self) -> list[set[str]]:
        """
        Find disjoint state trees/graphs in the structure.

        Returns a list of sets, where each set contains state IDs that are
        connected (directly or indirectly) through transitions.
        """
        if not self.states:
            return []

        # Build adjacency list (undirected for connectivity)
        adjacency: dict[str, set[str]] = {s.id: set() for s in self.states}

        for t in self.transitions:
            if hasattr(t, "state_before") and hasattr(t, "state_after"):
                if t.state_before and t.state_after:
                    if t.state_before in adjacency and t.state_after in adjacency:
                        adjacency[t.state_before].add(t.state_after)
                        adjacency[t.state_after].add(t.state_before)

        # Find connected components using BFS
        visited: set[str] = set()
        components: list[set[str]] = []

        for state_id in adjacency:
            if state_id in visited:
                continue

            # BFS from this state
            component: set[str] = set()
            queue = [state_id]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                component.add(current)

                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            components.append(component)

        return components

    def is_connected(self) -> bool:
        """Check if all states form a single connected graph."""
        components = self.get_connected_components()
        return len(components) <= 1

    def get_root_states(self) -> list[CorrelatedState]:
        """
        Get states that have no incoming transitions (potential root states).

        In a multi-tree structure, this returns the roots of all trees.
        """
        states_with_incoming = set()
        for t in self.transitions:
            if hasattr(t, "state_after") and t.state_after:
                states_with_incoming.add(t.state_after)

        return [s for s in self.states if s.id not in states_with_incoming]

    # =========================================================================
    # Serialization
    # =========================================================================

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "states": [
                {
                    "id": s.id,
                    "name": s.name,
                    "confidence": s.confidence,
                    "state_type": (
                        s.state_type.value if hasattr(s.state_type, "value") else str(s.state_type)
                    ),
                    "source_component": s.source_component,
                    "route": s.route,
                    "screenshot_id": s.screenshot.id if s.screenshot else None,
                }
                for s in self.states
            ],
            "transitions": [
                {
                    "id": t.id,
                    "type": ("verified" if isinstance(t, VerifiedTransition) else "inferred"),
                    "state_before": getattr(t, "state_before", None),
                    "state_after": getattr(t, "state_after", None),
                    "confidence": t.confidence,
                }
                for t in self.transitions
            ],
            "elements_count": len(self.elements),
            "screenshots_count": len(self.screenshots),
            "active_state_ids": list(self._active_state_ids),
            "state_origins": self.state_origins,
            "transition_origins": self.transition_origins,
            "element_origins": self.element_origins,
            "viewport": {
                "width": self.viewport.width,
                "height": self.viewport.height,
                "scale_factor": self.viewport.scale_factor,
            },
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StateStructure":
        """Deserialize from dictionary."""
        viewport_data = data.get("viewport", {})
        viewport = Viewport(
            width=viewport_data.get("width", 1920),
            height=viewport_data.get("height", 1080),
            scale_factor=viewport_data.get("scale_factor", 1.0),
        )

        structure = cls(
            id=data["id"],
            name=data["name"],
            viewport=viewport,
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.now()
            ),
            metadata=data.get("metadata", {}),
            state_origins=data.get("state_origins", {}),
            transition_origins=data.get("transition_origins", {}),
            element_origins=data.get("element_origins", {}),
        )

        # Restore active states
        structure._active_state_ids = set(data.get("active_state_ids", []))

        return structure

    @classmethod
    def from_extraction_result(
        cls,
        result: ExtractionResult,
        structure_id: str,
        structure_name: str,
        source_id: str | None = None,
    ) -> "StateStructure":
        """
        Create a StateStructure from an extraction result.

        Args:
            result: The extraction result
            structure_id: ID for the new structure
            structure_name: Name for the new structure
            source_id: Optional source identifier for replacement operations
        """
        # Collect all states
        all_states = result.correlated_states.copy()
        if result.inferred_states:
            all_states.extend(result.inferred_states)

        # Collect all transitions
        all_transitions: list[InferredTransition | VerifiedTransition] = []
        all_transitions.extend(result.verified_transitions)
        all_transitions.extend(result.inferred_transitions)

        # Extract screenshots from states
        screenshots_dict: dict[str, Screenshot] = {}
        for state in all_states:
            if state.screenshot:
                screenshots_dict[state.screenshot.id] = state.screenshot

        # Extract elements from runtime extraction if available
        elements: list[ExtractedElement] = []
        if result.runtime_extraction and hasattr(result.runtime_extraction, "states"):
            for runtime_state in result.runtime_extraction.states:
                if hasattr(runtime_state, "elements"):
                    elements.extend(runtime_state.elements)

        # Build origin maps if source_id provided
        state_origins = {}
        transition_origins = {}
        element_origins = {}

        if source_id:
            for state in all_states:
                state_origins[state.id] = source_id
            for transition in all_transitions:
                transition_origins[transition.id] = source_id
            for element in elements:
                element_origins[element.id] = source_id

        structure = cls(
            id=structure_id,
            name=structure_name,
            states=all_states,
            transitions=all_transitions,
            elements=elements,
            screenshots=screenshots_dict,
            state_origins=state_origins,
            transition_origins=transition_origins,
            element_origins=element_origins,
            metadata={
                "extraction_id": result.extraction_id,
                "mode": result.mode,
                "framework": result.framework,
                "started_at": result.started_at.isoformat(),
                "completed_at": (result.completed_at.isoformat() if result.completed_at else None),
            },
        )

        return structure

    def merge(self, other: "StateStructure") -> "StateStructure":
        """
        Merge another state structure into a new combined structure.

        States and transitions are combined. If there are ID conflicts,
        states/transitions from 'other' take precedence.
        """
        merged = StateStructure(
            id=f"{self.id}+{other.id}",
            name=f"{self.name} + {other.name}",
            viewport=self.viewport,
            created_at=datetime.now(),
            metadata={
                "merged_from": [self.id, other.id],
                "merged_at": datetime.now().isoformat(),
            },
        )

        # Merge states (other takes precedence on conflicts)
        state_map: dict[str, CorrelatedState] = {}
        for s in self.states:
            state_map[s.id] = s
        for s in other.states:
            state_map[s.id] = s
        merged.states = list(state_map.values())

        # Merge transitions
        transition_map: dict[str, InferredTransition | VerifiedTransition] = {}
        for t in self.transitions:
            transition_map[t.id] = t
        for t in other.transitions:
            transition_map[t.id] = t
        merged.transitions = list(transition_map.values())

        # Merge elements
        element_map: dict[str, ExtractedElement] = {}
        for e in self.elements:
            element_map[e.id] = e
        for e in other.elements:
            element_map[e.id] = e
        merged.elements = list(element_map.values())

        # Merge screenshots
        merged.screenshots = {**self.screenshots, **other.screenshots}

        # Merge origin tracking (other takes precedence on conflicts)
        merged.state_origins = {**self.state_origins, **other.state_origins}
        merged.transition_origins = {
            **self.transition_origins,
            **other.transition_origins,
        }
        merged.element_origins = {**self.element_origins, **other.element_origins}

        # Merge active states
        merged._active_state_ids = self._active_state_ids | other._active_state_ids

        return merged


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# These aliases maintain compatibility with code using the old names
ApplicationStateStructure = StateStructure
CompositeStateStructure = StateStructure
