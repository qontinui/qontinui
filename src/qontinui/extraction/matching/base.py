"""
Abstract base class for correlating static and runtime data.

This module defines the interface for state matchers that combine information
from static analysis and runtime extraction to build a complete state model.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..runtime.base import RuntimeExtractionResult, RuntimeExtractor
from ..static.models import StaticAnalysisResult

if TYPE_CHECKING:
    pass


@dataclass
class MatchingEvidence:
    """Evidence supporting a correlation between static and runtime data."""

    evidence_type: str  # name_match, structural_match, behavioral_match, visual_match
    source_static: str | None = None  # ID or reference to static analysis item
    source_runtime: str | None = None  # ID or reference to runtime item
    confidence: float = 0.5
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrelatedState:
    """A UI state with correlated static and runtime data."""

    id: str
    name: str
    static_component_id: str | None = None
    runtime_state_id: str | None = None
    evidence: list[MatchingEvidence] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InferredTransition:
    """A transition inferred from static analysis that needs verification."""

    id: str
    from_state_id: str | None
    to_state_id: str | None
    trigger_type: str  # click, hover, focus, etc.
    trigger_element_id: str | None
    predicted_changes: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifiedTransition:
    """A transition that has been verified at runtime."""

    inferred: InferredTransition
    verified: bool = False
    actual_changes: dict[str, Any] = field(default_factory=dict)
    matches_prediction: bool = False
    confidence: float = 0.0
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class StateMatcher(ABC):
    """
    Abstract base class for correlating static and runtime data.

    State matchers bridge the gap between:
    - Static analysis: What the code says the UI should do
    - Runtime extraction: What the UI actually does

    By correlating these two sources of information, we can:
    - Validate that static analysis correctly predicted runtime behavior
    - Discover dynamic behaviors not evident from static analysis
    - Build high-confidence state models with evidence from both sources
    - Identify discrepancies between code and runtime behavior
    """

    @abstractmethod
    def correlate(
        self,
        static: StaticAnalysisResult,
        runtime: RuntimeExtractionResult,
    ) -> list[CorrelatedState]:
        """
        Match static analysis results with runtime observations.

        This is the core correlation algorithm that:
        1. Matches components from static analysis to UI elements from runtime
        2. Correlates state variables with observed UI states
        3. Maps conditional rendering logic to actual state transitions
        4. Identifies which event handlers trigger which transitions
        5. Validates route definitions against actual navigation

        The correlation uses multiple matching strategies:
        - Name/label matching (component names, IDs, selectors)
        - Structural matching (DOM hierarchy, component tree)
        - Behavioral matching (state changes, event handlers)
        - Visual matching (screenshots, bounding boxes)

        Args:
            static: Results from static code analysis containing component
                   definitions, state variables, event handlers, etc.
            runtime: Results from runtime extraction containing observed UI
                    elements, states, and transitions.

        Returns:
            List of CorrelatedState objects, each representing a UI state
            with evidence from both static and runtime sources.

        Example:
            A correlated state might represent a "Login Modal" that was:
            - Found in static analysis as a component with state variable
              "isLoginModalOpen"
            - Observed at runtime as a dialog region with login form elements
            - Matched based on the component name, element IDs, and visual
              structure
        """
        pass

    @abstractmethod
    async def verify_transitions(
        self,
        transitions: list[InferredTransition],
        runtime_extractor: RuntimeExtractor,
    ) -> list[VerifiedTransition]:
        """
        Verify inferred transitions by executing them at runtime.

        Static analysis can infer potential transitions (e.g., "clicking this
        button should open this modal"), but we need runtime verification to
        confirm:
        1. The transition actually works as expected
        2. The predicted state changes actually occur
        3. There are no unexpected side effects
        4. The transition is reliably reproducible

        This method:
        1. Takes a list of transitions inferred from static analysis
        2. Uses the runtime extractor to execute each transition
        3. Observes the actual state changes
        4. Compares predicted vs. actual outcomes
        5. Returns verified transitions with confidence scores

        Args:
            transitions: List of transitions inferred from static analysis
                        that need runtime verification.
            runtime_extractor: Connected runtime extractor to use for executing
                             transitions and observing state changes.

        Returns:
            List of VerifiedTransition objects containing:
            - The original inferred transition
            - Whether it was successfully executed
            - Actual vs. predicted state changes
            - Confidence score based on verification results
            - Any unexpected behaviors observed

        Example:
            An inferred transition might predict:
            - Clicking "Login" button opens login modal
            - Verification executes the click
            - Observes that login modal does appear
            - Returns VerifiedTransition with high confidence (0.95)

        Raises:
            VerificationError: If unable to execute transitions for verification.
        """
        pass

    @abstractmethod
    def compute_confidence(
        self,
        state: CorrelatedState,
        evidence: list[MatchingEvidence],
    ) -> float:
        """
        Compute confidence score for a state based on matching evidence.

        Not all state correlations are equally certain. This method computes
        a confidence score (0.0 to 1.0) based on the quality and quantity of
        evidence supporting the correlation.

        Evidence types considered:
        - Exact name/ID matches (high confidence)
        - Structural matches (medium confidence)
        - Visual/layout matches (medium confidence)
        - Behavioral matches (high confidence)
        - Multiple independent sources of evidence (higher confidence)

        The confidence score helps prioritize which states to use for
        automation:
        - 0.9-1.0: Very high confidence, safe to use
        - 0.7-0.9: High confidence, generally reliable
        - 0.5-0.7: Medium confidence, may need manual review
        - <0.5: Low confidence, likely needs manual intervention

        Args:
            state: The correlated state to compute confidence for.
            evidence: List of evidence items supporting this correlation,
                     each with its own type and strength.

        Returns:
            Confidence score between 0.0 (no confidence) and 1.0 (certain).

        Example:
            A state with evidence:
            - Component name matches element ID (weight: 0.8)
            - DOM structure matches component hierarchy (weight: 0.6)
            - Visual layout matches predicted layout (weight: 0.5)
            - Event handler matches observed transition (weight: 0.9)
            Might compute to confidence score: 0.87
        """
        pass
