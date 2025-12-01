"""
Component matcher for correlating components to DOM elements.

Matches React components from static analysis to DOM elements extracted
at runtime using multiple matching strategies.
"""

import logging
from difflib import SequenceMatcher

from ..models.correlated import EvidenceType, MatchingEvidence
from ..models.runtime import DetectedRegion, ExtractedElement
from ..models.static import ComponentDefinition

logger = logging.getLogger(__name__)


class ComponentMatcher:
    """Matches React components from static analysis to DOM elements."""

    def match_component_to_elements(
        self,
        component: ComponentDefinition,
        elements: list[ExtractedElement],
        regions: list[DetectedRegion],
    ) -> list[MatchingEvidence]:
        """Find DOM elements that correspond to a component.

        Matching strategies:
        1. data-testid attributes
        2. Class name matching
        3. Text content matching
        4. Structural hierarchy matching
        5. ARIA role/label matching

        Args:
            component: Component from static analysis.
            elements: Extracted DOM elements.
            regions: Detected UI regions.

        Returns:
            List of matching evidence items.
        """
        evidence: list[MatchingEvidence] = []

        # Strategy 1: Test ID matching
        test_id_evidence = self.match_by_test_id(component, elements)
        evidence.extend(test_id_evidence)

        # Strategy 2: Class name matching
        class_evidence = self.match_by_class_names(component, elements)
        evidence.extend(class_evidence)

        # Strategy 3: Text content matching
        text_evidence = self.match_by_text_content(component, elements)
        evidence.extend(text_evidence)

        # Strategy 4: ARIA matching
        aria_evidence = self.match_by_aria(component, elements)
        evidence.extend(aria_evidence)

        # Strategy 5: Name matching (component name in selectors)
        name_evidence = self.match_by_name(component, elements)
        evidence.extend(name_evidence)

        return evidence

    def match_by_test_id(
        self, component: ComponentDefinition, elements: list[ExtractedElement]
    ) -> list[MatchingEvidence]:
        """Match using data-testid attributes.

        Test IDs are the most reliable matching strategy since they're
        explicitly defined for testing purposes.

        Args:
            component: Component to match.
            elements: Available elements.

        Returns:
            List of matching evidence.
        """
        evidence: list[MatchingEvidence] = []

        # Convert component name to common test ID patterns
        component_lower = component.name.lower()
        test_id_patterns = [
            component_lower,
            component_lower.replace("_", "-"),
            component_lower.replace("_", ""),
            f"{component_lower}-container",
            f"{component_lower}-wrapper",
        ]

        for element in elements:
            test_id = element.attributes.get("data-testid")
            if not test_id:
                continue

            test_id_lower = test_id.lower()

            # Exact match
            if test_id_lower in test_id_patterns:
                evidence.append(
                    MatchingEvidence(
                        evidence_type=EvidenceType.TEST_ID_MATCH,
                        strength=1.0,
                        description=f"Exact test ID match: {test_id}",
                        static_reference=component.name,
                        runtime_reference=element.id,
                        metadata={"test_id": test_id},
                    )
                )
            # Partial match
            elif any(pattern in test_id_lower for pattern in test_id_patterns):
                evidence.append(
                    MatchingEvidence(
                        evidence_type=EvidenceType.TEST_ID_MATCH,
                        strength=0.8,
                        description=f"Partial test ID match: {test_id}",
                        static_reference=component.name,
                        runtime_reference=element.id,
                        metadata={"test_id": test_id},
                    )
                )

        return evidence

    def match_by_class_names(
        self, component: ComponentDefinition, elements: list[ExtractedElement]
    ) -> list[MatchingEvidence]:
        """Match using CSS class names from source.

        Class names often reflect component names, especially in
        CSS Modules or styled-components.

        Args:
            component: Component to match.
            elements: Available elements.

        Returns:
            List of matching evidence.
        """
        evidence: list[MatchingEvidence] = []

        component_lower = component.name.lower()
        component_kebab = self._to_kebab_case(component.name)
        component_snake = self._to_snake_case(component.name)

        for element in elements:
            if not element.class_names:
                continue

            for class_name in element.class_names:
                class_lower = class_name.lower()

                # Exact component name match
                if component_lower in class_lower or component_kebab in class_lower:
                    evidence.append(
                        MatchingEvidence(
                            evidence_type=EvidenceType.CLASS_NAME_MATCH,
                            strength=0.7,
                            description=f"Class name contains component name: {class_name}",
                            static_reference=component.name,
                            runtime_reference=element.id,
                            metadata={"class_name": class_name},
                        )
                    )
                    break  # Only add one evidence per element

        return evidence

    def match_by_text_content(
        self, component: ComponentDefinition, elements: list[ExtractedElement]
    ) -> list[MatchingEvidence]:
        """Match using text content from JSX.

        If we can extract text from JSX literals in the component,
        we can match against element text content.

        Args:
            component: Component to match.
            elements: Available elements.

        Returns:
            List of matching evidence.
        """
        evidence: list[MatchingEvidence] = []

        # Extract potential text from component metadata
        jsx_text = component.metadata.get("jsx_text", [])
        if not jsx_text:
            return evidence

        for element in elements:
            if not element.text_content:
                continue

            element_text = element.text_content.strip().lower()
            if not element_text:
                continue

            # Check for text matches
            for text in jsx_text:
                text_lower = text.lower()
                if text_lower == element_text:
                    evidence.append(
                        MatchingEvidence(
                            evidence_type=EvidenceType.TEXT_CONTENT_MATCH,
                            strength=0.6,
                            description=f"Exact text match: '{text}'",
                            static_reference=component.name,
                            runtime_reference=element.id,
                            metadata={"matched_text": text},
                        )
                    )
                elif text_lower in element_text or element_text in text_lower:
                    evidence.append(
                        MatchingEvidence(
                            evidence_type=EvidenceType.TEXT_CONTENT_MATCH,
                            strength=0.4,
                            description=f"Partial text match: '{text}'",
                            static_reference=component.name,
                            runtime_reference=element.id,
                            metadata={"matched_text": text},
                        )
                    )

        return evidence

    def match_by_aria(
        self, component: ComponentDefinition, elements: list[ExtractedElement]
    ) -> list[MatchingEvidence]:
        """Match using ARIA role/label.

        ARIA attributes provide semantic meaning and are good indicators
        of component type and identity.

        Args:
            component: Component to match.
            elements: Available elements.

        Returns:
            List of matching evidence.
        """
        evidence: list[MatchingEvidence] = []

        component_lower = component.name.lower()

        for element in elements:
            # Match aria-label
            if element.aria_label:
                aria_lower = element.aria_label.lower()
                similarity = SequenceMatcher(None, component_lower, aria_lower).ratio()

                if similarity > 0.8:
                    evidence.append(
                        MatchingEvidence(
                            evidence_type=EvidenceType.ARIA_MATCH,
                            strength=0.8 * similarity,
                            description=f"ARIA label similar to component name: {element.aria_label}",
                            static_reference=component.name,
                            runtime_reference=element.id,
                            metadata={
                                "aria_label": element.aria_label,
                                "similarity": similarity,
                            },
                        )
                    )

            # Match accessible name
            if element.name:
                name_lower = element.name.lower()
                similarity = SequenceMatcher(None, component_lower, name_lower).ratio()

                if similarity > 0.8:
                    evidence.append(
                        MatchingEvidence(
                            evidence_type=EvidenceType.ARIA_MATCH,
                            strength=0.7 * similarity,
                            description=f"Accessible name similar to component name: {element.name}",
                            static_reference=component.name,
                            runtime_reference=element.id,
                            metadata={
                                "accessible_name": element.name,
                                "similarity": similarity,
                            },
                        )
                    )

        return evidence

    def match_by_name(
        self, component: ComponentDefinition, elements: list[ExtractedElement]
    ) -> list[MatchingEvidence]:
        """Match by component name appearing in selectors or IDs.

        Args:
            component: Component to match.
            elements: Available elements.

        Returns:
            List of matching evidence.
        """
        evidence: list[MatchingEvidence] = []

        component_lower = component.name.lower()
        component_kebab = self._to_kebab_case(component.name)

        for element in elements:
            # Check selector
            if (
                component_lower in element.selector.lower()
                or component_kebab in element.selector
            ):
                evidence.append(
                    MatchingEvidence(
                        evidence_type=EvidenceType.NAME_MATCH,
                        strength=0.6,
                        description=f"Component name in selector: {element.selector}",
                        static_reference=component.name,
                        runtime_reference=element.id,
                        metadata={"selector": element.selector},
                    )
                )

            # Check element ID attribute
            element_id_attr = element.attributes.get("id")
            if element_id_attr and (
                component_lower in element_id_attr.lower()
                or component_kebab in element_id_attr.lower()
            ):
                evidence.append(
                    MatchingEvidence(
                        evidence_type=EvidenceType.NAME_MATCH,
                        strength=0.7,
                        description=f"Component name in element ID: {element_id_attr}",
                        static_reference=component.name,
                        runtime_reference=element.id,
                        metadata={"element_id": element_id_attr},
                    )
                )

        return evidence

    @staticmethod
    def _to_kebab_case(name: str) -> str:
        """Convert PascalCase/camelCase to kebab-case."""
        import re

        # Insert hyphens before uppercase letters
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1-\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1-\2", s1).lower()

    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert PascalCase/camelCase to snake_case."""
        import re

        # Insert underscores before uppercase letters
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
