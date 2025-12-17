"""
Correlated/output models.

These models represent the final output of the extraction process - states
and transitions that have been correlated between static analysis and runtime
extraction, with confidence scores and evidence.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .base import BoundingBox, Screenshot
from .runtime import StateType
from .static import StaticAnalysisResult


class EvidenceType(Enum):
    """Type of matching evidence."""

    # Element-based matching
    ELEMENT_MATCH = "element_match"  # Element text/attributes match
    TEST_ID_MATCH = "test_id_match"  # data-testid attribute matches
    CLASS_NAME_MATCH = "class_name_match"  # CSS class name matches component
    TEXT_CONTENT_MATCH = "text_content_match"  # Text content matches JSX
    ARIA_MATCH = "aria_match"  # ARIA label/role matches
    NAME_MATCH = "name_match"  # Component name appears in selector/ID

    # Structural matching
    SELECTOR_MATCH = "selector_match"  # CSS selector matches component
    STRUCTURAL_MATCH = "structural_match"  # Component hierarchy matches DOM

    # State-based matching
    STATE_VARIABLE_MATCH = "state_variable_match"  # State variable controls visibility
    CONDITIONAL_MATCH = "conditional_match"  # Conditional logic matches
    RUNTIME_VERIFIED = "runtime_verified"  # Verified by runtime execution

    # Event-based matching
    EVENT_HANDLER_MATCH = "event_handler_match"  # Event handler matches action

    # Route-based matching
    ROUTE_MATCH = "route_match"  # URL matches route definition

    # Timing-based matching
    TIMING_MATCH = "timing_match"  # Timing of appearance/disappearance


@dataclass
class MatchingEvidence:
    """Evidence for a correlation between static and runtime data."""

    evidence_type: EvidenceType
    description: str

    # Strength/confidence (use either field, they're synchronized)
    strength: float = 0.0  # Primary field: How strong this evidence is (0-1)
    confidence_contribution: float | None = None  # Legacy field, auto-synced with strength

    # Source references
    source_file: Path | None = None
    source_line: int | None = None
    runtime_state_id: str | None = None
    runtime_element_id: str | None = None

    # Additional references for matcher compatibility
    static_reference: str | None = None  # Reference to static analysis item
    runtime_reference: str | None = None  # Reference to runtime item

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure confidence_contribution is synchronized with strength."""
        if self.confidence_contribution is None:
            self.confidence_contribution = self.strength


@dataclass
class CorrelatedState:
    """A UI state correlated between static analysis and runtime extraction."""

    id: str
    name: str

    # Static analysis source
    source_component: str | None = None  # Component ID from static analysis
    controlling_variables: list[str] = field(default_factory=list)  # State variable IDs
    conditions: list[str] = field(default_factory=list)  # Conditional render IDs
    source_file: Path | None = None
    source_line: int | None = None

    # Runtime extraction
    screenshot: Screenshot | None = None
    bounding_box: BoundingBox | None = None
    elements: list[str] = field(default_factory=list)  # Element IDs from runtime
    route: str | None = None  # URL or route path

    # Classification
    state_type: StateType = StateType.UNKNOWN

    # Confidence and evidence
    confidence: float = 0.0  # Overall confidence (0-1)
    match_evidence: list[MatchingEvidence] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferredTransition:
    """A transition inferred from static analysis."""

    id: str
    trigger_handler: str  # Event handler ID from static analysis

    # States
    state_before: str | None = None  # Correlated state ID
    state_after: str | None = None  # Correlated state ID

    # Effects (from static analysis)
    causes_appear: list[str] = field(default_factory=list)  # Component/conditional IDs
    causes_disappear: list[str] = field(default_factory=list)  # Component/conditional IDs

    # Confidence
    confidence: float = 0.0

    # Source
    source_file: Path | None = None
    source_line: int | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationDiscrepancy:
    """A discrepancy between inferred and observed transitions."""

    discrepancy_type: str  # missing_element, extra_element, wrong_state, etc.
    description: str
    expected: Any = None
    actual: Any = None


@dataclass
class VerifiedTransition:
    """A transition verified by correlating static and runtime data."""

    id: str

    # Source transitions
    inferred: str | None = None  # InferredTransition ID
    observed: str | None = None  # ObservedTransition ID (from runtime)

    # Verification
    verified: bool = False
    verification_method: str = ""  # How it was verified

    # Action
    action_type: str = ""  # click, hover, type, etc.
    trigger_element: str | None = None  # Element ID from runtime
    trigger_selector: str | None = None  # CSS selector

    # Effects
    causes_appear: list[str] = field(default_factory=list)  # Correlated state IDs
    causes_disappear: list[str] = field(default_factory=list)  # Correlated state IDs

    # Confidence
    confidence: float = 0.0

    # Discrepancies
    discrepancies: list[VerificationDiscrepancy] = field(default_factory=list)

    # Source
    source_file: Path | None = None
    source_line: int | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Complete extraction result with correlated data."""

    extraction_id: str
    mode: str  # ExtractionMode value
    framework: str  # FrameworkType value

    # Source results
    static_analysis: StaticAnalysisResult | None = None
    runtime_extraction: Any = None  # RuntimeExtractionResult (avoid circular import)

    # Inferred from static
    inferred_states: list[CorrelatedState] = field(default_factory=list)
    inferred_transitions: list[InferredTransition] = field(default_factory=list)

    # Correlated results
    correlated_states: list[CorrelatedState] = field(default_factory=list)
    verified_transitions: list[VerifiedTransition] = field(default_factory=list)

    # Timestamps
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_correlated_state(self, state_id: str) -> CorrelatedState | None:
        """Get correlated state by ID."""
        for state in self.correlated_states:
            if state.id == state_id:
                return state
        return None

    def get_verified_transition(self, transition_id: str) -> VerifiedTransition | None:
        """Get verified transition by ID."""
        for transition in self.verified_transitions:
            if transition.id == transition_id:
                return transition
        return None

    def get_inferred_transition(self, transition_id: str) -> InferredTransition | None:
        """Get inferred transition by ID."""
        for transition in self.inferred_transitions:
            if transition.id == transition_id:
                return transition
        return None

    def get_high_confidence_states(self, min_confidence: float = 0.7) -> list[CorrelatedState]:
        """Get states with confidence above threshold."""
        return [s for s in self.correlated_states if s.confidence >= min_confidence]

    def get_verified_transitions_only(self) -> list[VerifiedTransition]:
        """Get only verified transitions."""
        return [t for t in self.verified_transitions if t.verified]

    def get_unverified_transitions(self) -> list[VerifiedTransition]:
        """Get transitions that couldn't be verified."""
        return [t for t in self.verified_transitions if not t.verified]
