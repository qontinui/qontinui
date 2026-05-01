"""
State variable matcher for correlating state to UI visibility.

Matches state variables from static analysis to UI visibility changes
observed at runtime.
"""

import logging
from difflib import SequenceMatcher

from ..models.correlated import EvidenceType, MatchingEvidence
from ..models.runtime import DetectedRegion, RuntimeStateCapture
from ..models.static import ConditionalRender, StateVariable

logger = logging.getLogger(__name__)


class StateVariableMatcher:
    """Matches state variables to UI visibility changes."""

    def match_state_to_visibility(
        self,
        state_var: StateVariable,
        conditional_renders: list[ConditionalRender],
        runtime_states: list[RuntimeStateCapture],
    ) -> list[MatchingEvidence]:
        """Determine which runtime states correspond to state variable values.

        Analyzes conditional rendering logic that depends on this state
        variable and matches it to observed UI states at runtime.

        Args:
            state_var: State variable from static analysis.
            conditional_renders: Conditional rendering patterns that depend on this state.
            runtime_states: Captured runtime states.

        Returns:
            List of matching evidence linking state variable to runtime observations.
        """
        evidence: list[MatchingEvidence] = []

        # Find conditionals that use this state variable
        relevant_conditionals = [
            cond
            for cond in conditional_renders
            if state_var.id in cond.controlling_variables
            or state_var.name in cond.condition
        ]

        if not relevant_conditionals:
            logger.debug(
                f"No conditional renders found for state variable: {state_var.name}"
            )
            return evidence

        # For each conditional, try to correlate with runtime states
        for conditional in relevant_conditionals:
            cond_evidence = self._match_conditional_to_states(
                state_var, conditional, runtime_states
            )
            evidence.extend(cond_evidence)

        return evidence

    def correlate_conditionals_to_regions(
        self, conditional: ConditionalRender, regions: list[DetectedRegion]
    ) -> list[tuple[str, DetectedRegion]]:
        """Match conditional render components to detected regions.

        Args:
            conditional: Conditional rendering pattern.
            regions: Detected UI regions.

        Returns:
            List of (component_name, region) pairs.
        """
        matches: list[tuple[str, DetectedRegion]] = []

        # Get all components rendered by this conditional
        all_components = conditional.renders_when_true + conditional.renders_when_false

        for component_name in all_components:
            for region in regions:
                # Try various matching strategies
                if self._component_matches_region(component_name, region):
                    matches.append((component_name, region))

        return matches

    def _match_conditional_to_states(
        self,
        state_var: StateVariable,
        conditional: ConditionalRender,
        runtime_states: list[RuntimeStateCapture],
    ) -> list[MatchingEvidence]:
        """Match a conditional render pattern to runtime states.

        Args:
            state_var: State variable controlling the conditional.
            conditional: Conditional rendering pattern.
            runtime_states: Captured runtime states.

        Returns:
            Matching evidence.
        """
        evidence: list[MatchingEvidence] = []

        # Look for runtime states that contain components from the conditional
        for runtime_state in runtime_states:
            # Check if any conditionally-rendered components appear in this state
            for component_name in conditional.renders_when_true:
                similarity = self._find_component_in_state(
                    component_name, runtime_state
                )
                if similarity > 0.6:
                    evidence.append(
                        MatchingEvidence(
                            evidence_type=EvidenceType.STRUCTURAL_MATCH,
                            strength=similarity * 0.7,  # Scale down since it's indirect
                            description=f"Component '{component_name}' appears when {state_var.name} is true",
                            static_reference=f"{state_var.name}=true",
                            runtime_reference=runtime_state.id,
                            metadata={
                                "conditional_id": conditional.id,
                                "component": component_name,
                                "branch": "true",
                            },
                        )
                    )

            for component_name in conditional.renders_when_false:
                similarity = self._find_component_in_state(
                    component_name, runtime_state
                )
                if similarity > 0.6:
                    evidence.append(
                        MatchingEvidence(
                            evidence_type=EvidenceType.STRUCTURAL_MATCH,
                            strength=similarity * 0.7,
                            description=f"Component '{component_name}' appears when {state_var.name} is false",
                            static_reference=f"{state_var.name}=false",
                            runtime_reference=runtime_state.id,
                            metadata={
                                "conditional_id": conditional.id,
                                "component": component_name,
                                "branch": "false",
                            },
                        )
                    )

        return evidence

    def _component_matches_region(
        self, component_name: str, region: DetectedRegion
    ) -> bool:
        """Check if a component name matches a detected region.

        Args:
            component_name: Component name from static analysis.
            region: Detected region from runtime.

        Returns:
            True if they likely match.
        """
        component_lower = component_name.lower()

        # Check selector
        if region.selector and component_lower in region.selector.lower():
            return True

        # Check aria label
        if region.aria_label and component_lower in region.aria_label.lower():
            return True

        # Check semantic role
        if region.semantic_role and component_lower in region.semantic_role.lower():
            return True

        # Check metadata
        if region.metadata:
            matched_selector = region.metadata.get("matched_selector", "")
            if component_lower in matched_selector.lower():
                return True

        return False

    def _find_component_in_state(
        self, component_name: str, runtime_state: RuntimeStateCapture
    ) -> float:
        """Find how well a component name matches elements in a runtime state.

        Args:
            component_name: Component name to find.
            runtime_state: Runtime state capture.

        Returns:
            Similarity score (0.0 to 1.0).
        """
        component_lower = component_name.lower()
        component_kebab = self._to_kebab_case(component_name)
        best_similarity = 0.0

        # Check in detected regions
        for region in runtime_state.regions:
            # Check region type name
            if region.region_type:
                similarity = SequenceMatcher(
                    None, component_lower, region.region_type.value.lower()
                ).ratio()
                best_similarity = max(best_similarity, similarity)

            # Check aria label
            if region.aria_label:
                similarity = SequenceMatcher(
                    None, component_lower, region.aria_label.lower()
                ).ratio()
                best_similarity = max(best_similarity, similarity)

        # Check in elements
        for element in runtime_state.elements:
            # Check test ID
            test_id = element.attributes.get("data-testid")
            if test_id:
                if (
                    component_lower in test_id.lower()
                    or component_kebab in test_id.lower()
                ):
                    best_similarity = max(best_similarity, 0.9)

            # Check class names
            for class_name in element.class_names:
                if (
                    component_lower in class_name.lower()
                    or component_kebab in class_name.lower()
                ):
                    best_similarity = max(best_similarity, 0.8)

            # Check aria label
            if element.aria_label:
                similarity = SequenceMatcher(
                    None, component_lower, element.aria_label.lower()
                ).ratio()
                best_similarity = max(best_similarity, similarity)

        return best_similarity

    @staticmethod
    def _to_kebab_case(name: str) -> str:
        """Convert PascalCase/camelCase to kebab-case."""
        import re

        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1-\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1-\2", s1).lower()
