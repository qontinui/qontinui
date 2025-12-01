"""
Default implementation of StateMatcher.

Correlates static analysis results with runtime extraction to build
a complete state model with confidence scores.
"""

import logging
import uuid
from typing import TYPE_CHECKING

from ..models.correlated import CorrelatedState, InferredTransition, VerifiedTransition
from ..models.static import StaticAnalysisResult
from .component_matcher import ComponentMatcher
from .confidence import (
    compute_state_confidence,
    compute_transition_confidence,
    filter_weak_evidence,
)
from .state_matcher import StateVariableMatcher
from .transition_verifier import TransitionVerifier

if TYPE_CHECKING:
    from ..runtime.base import RuntimeExtractor
    from ..runtime.types import RuntimeExtractionSession


logger = logging.getLogger(__name__)


class DefaultStateMatcher:
    """Default implementation of state matching."""

    def __init__(self):
        """Initialize the state matcher."""
        self.component_matcher = ComponentMatcher()
        self.state_matcher = StateVariableMatcher()

    def correlate(
        self, static: StaticAnalysisResult, runtime: "RuntimeExtractionSession"
    ) -> list[CorrelatedState]:
        """Match static analysis results with runtime observations.

        Process:
        1. Match components to DOM elements
        2. Match state variables to UI visibility
        3. Correlate conditional renders to observed states
        4. Build correlated state objects with confidence scores

        Args:
            static: Results from static code analysis.
            runtime: Results from runtime extraction.

        Returns:
            List of correlated states with evidence and confidence scores.
        """
        logger.info("Starting state correlation...")
        correlated_states: list[CorrelatedState] = []

        # Get all runtime captures
        if not runtime.captures:
            logger.warning("No runtime captures available for correlation")
            return correlated_states

        # Extract regions from first capture for matching
        first_capture = runtime.captures[0]
        all_elements = first_capture.elements
        all_states = first_capture.states

        # Match each component to runtime elements
        for component in static.components:
            logger.debug(f"Correlating component: {component.name}")

            # Find matching elements using component matcher
            # Note: regions aren't directly available, so we use states as a proxy
            evidence = self.component_matcher.match_component_to_elements(
                component, all_elements, []  # Empty regions for now
            )

            if not evidence:
                logger.debug(f"No evidence found for component: {component.name}")
                continue

            # Filter out weak evidence
            strong_evidence = filter_weak_evidence(evidence, threshold=0.4)

            if not strong_evidence:
                continue

            # Find the most likely runtime state for this component
            runtime_state = self._find_best_runtime_state(
                component, all_states, evidence
            )

            # Get state variables for this component
            component_state_vars = [
                sv for sv in static.state_variables if sv.component == component.name
            ]

            # Create correlated state
            state_id = f"correlated_{uuid.uuid4().hex[:8]}"
            correlated = CorrelatedState(
                id=state_id,
                name=component.name,
                component=component,
                state_variables=component_state_vars,
                runtime_state=runtime_state,
                elements=[
                    e
                    for e in all_elements
                    if any(ev.runtime_reference == e.id for ev in evidence)
                ],
                evidence=strong_evidence,
                screenshot_id=(
                    first_capture.screenshot_path.name
                    if first_capture.screenshot_path
                    else None
                ),
            )

            # Compute confidence
            correlated.confidence = compute_state_confidence(
                correlated, strong_evidence
            )

            logger.info(
                f"Correlated state: {correlated.name} "
                f"(confidence: {correlated.confidence:.2f}, "
                f"evidence: {len(strong_evidence)} items)"
            )

            correlated_states.append(correlated)

        # Match state variables to visibility changes
        for state_var in static.state_variables:
            logger.debug(f"Analyzing state variable: {state_var.name}")

            # This would use conditional renders if we had them extracted
            # For now, we'll skip this part as ConditionalRender extraction
            # would happen during static analysis

        logger.info(f"Correlation complete: {len(correlated_states)} correlated states")
        return correlated_states

    async def verify_transitions(
        self,
        transitions: list[InferredTransition],
        runtime_extractor: "RuntimeExtractor",
    ) -> list[VerifiedTransition]:
        """Verify inferred transitions by executing them at runtime.

        Process:
        1. For each inferred transition:
           a. Get current state
           b. Execute the transition
           c. Capture new state
           d. Compare expected vs actual changes
        2. Return verified transitions with confidence scores

        Args:
            transitions: List of transitions inferred from static analysis.
            runtime_extractor: Connected runtime extractor for verification.

        Returns:
            List of verified transitions with results.
        """
        logger.info(f"Verifying {len(transitions)} transitions...")
        verified: list[VerifiedTransition] = []

        verifier = TransitionVerifier(runtime_extractor)

        for transition in transitions:
            logger.debug(f"Verifying transition: {transition.name}")

            try:
                # Get current state
                current_state = await runtime_extractor.extract_current_state()

                # Verify the transition
                result = await verifier.verify_transition(transition, current_state)

                # Compute confidence
                result.confidence = compute_transition_confidence(result)

                logger.info(
                    f"Verified transition: {transition.name} "
                    f"(verified: {result.verified}, "
                    f"confidence: {result.confidence:.2f})"
                )

                verified.append(result)

            except Exception as e:
                logger.error(
                    f"Error verifying transition {transition.name}: {e}", exc_info=True
                )
                # Add failed verification
                verified.append(
                    VerifiedTransition(
                        inferred=transition,
                        verified=False,
                        error=str(e),
                        confidence=0.0,
                    )
                )

        logger.info(
            f"Verification complete: {sum(1 for v in verified if v.verified)}/{len(verified)} verified"
        )
        return verified

    def compute_confidence(self, state: CorrelatedState, evidence: list) -> float:
        """Compute confidence score for a state.

        Args:
            state: Correlated state to score.
            evidence: Matching evidence.

        Returns:
            Confidence score (0.0 to 1.0).
        """
        return compute_state_confidence(state, evidence)

    def _find_best_runtime_state(self, component, all_states, evidence):
        """Find the best matching runtime state for a component.

        Args:
            component: Component from static analysis.
            all_states: All runtime states.
            evidence: Matching evidence.

        Returns:
            Best matching runtime state or None.
        """
        # Get element IDs from evidence
        element_ids = {ev.runtime_reference for ev in evidence if ev.runtime_reference}

        if not element_ids:
            return None

        # Find states that contain these elements
        candidate_states = []
        for state in all_states:
            overlap = len(set(state.element_ids) & element_ids)
            if overlap > 0:
                candidate_states.append((state, overlap))

        if not candidate_states:
            return None

        # Return state with most overlapping elements
        best_state = max(candidate_states, key=lambda x: x[1])
        return best_state[0]
