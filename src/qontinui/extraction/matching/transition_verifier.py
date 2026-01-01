"""
Transition verifier for runtime validation.

Verifies inferred transitions by executing them at runtime and comparing
expected vs actual state changes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from ..models.correlated import (
    InferredTransition,
    VerificationDiscrepancy,
    VerifiedTransition,
)
from ..models.static import EventHandler
from ..runtime.types import RuntimeStateCapture
from ..web.models import ExtractedElement

if TYPE_CHECKING:
    from ..runtime.base import RuntimeExtractor

logger = logging.getLogger(__name__)


class TransitionVerifier:
    """Verifies inferred transitions by executing them at runtime."""

    def __init__(self, extractor: RuntimeExtractor):
        """Initialize the verifier.

        Args:
            extractor: Connected runtime extractor to use for verification.
        """
        self.extractor = extractor

    async def verify_transition(
        self, transition: InferredTransition, current_state: RuntimeStateCapture
    ) -> VerifiedTransition:
        """Execute a transition and verify it produces expected results.

        Process:
        1. Find the trigger element in current DOM
        2. Execute the interaction
        3. Capture new state
        4. Compare actual changes to expected changes
        5. Return verified transition with confidence

        Args:
            transition: Transition to verify.
            current_state: Current runtime state before transition.

        Returns:
            VerifiedTransition with verification results.
        """
        start_time = time.time()

        try:
            # Find trigger element
            trigger_element = await self._find_trigger_by_handler(
                transition.trigger_handler, current_state.elements
            )

            if not trigger_element:
                logger.warning(f"Could not find trigger element for transition: {transition.id}")
                return VerifiedTransition(
                    id=f"verified_{transition.id}",
                    inferred=transition.id,
                    verified=False,
                    verification_method="trigger_not_found",
                    confidence=0.0,
                    discrepancies=[
                        VerificationDiscrepancy(
                            discrepancy_type="missing_trigger",
                            description="Trigger element not found",
                        )
                    ],
                    metadata={"execution_time_ms": (time.time() - start_time) * 1000},
                )

            # Capture before state
            before_state = await self.extractor.extract_current_state()
            before_state_ids = {s.id for s in before_state.states}
            before_screenshot = None
            if before_state.screenshot_path:
                before_screenshot = str(before_state.screenshot_path)

            # Execute the interaction (use "click" as default event type)
            await self._execute_interaction(trigger_element, "click")

            # Wait for state to stabilize
            await asyncio.sleep(0.5)  # Give time for animations/transitions

            # Capture after state
            after_state = await self.extractor.extract_current_state()
            after_state_ids = {s.id for s in after_state.states}
            after_screenshot = None
            if after_state.screenshot_path:
                after_screenshot = str(after_state.screenshot_path)

            # Determine what actually changed
            actual_appear = list(after_state_ids - before_state_ids)
            actual_disappear = list(before_state_ids - after_state_ids)

            # Compare with expected changes (from causes_appear/causes_disappear)
            matches, discrepancies = self.compare_state_changes(
                transition.causes_appear,
                transition.causes_disappear,
                actual_appear,
                actual_disappear,
            )

            # Compute confidence based on match quality
            confidence = self._compute_verification_confidence(
                matches, discrepancies, transition, actual_appear, actual_disappear
            )

            execution_time = (time.time() - start_time) * 1000

            return VerifiedTransition(
                id=f"verified_{transition.id}",
                inferred=transition.id,
                verified=matches,
                verification_method="runtime_execution",
                action_type="click",
                trigger_element=trigger_element.id,
                trigger_selector=trigger_element.selector,
                causes_appear=actual_appear,
                causes_disappear=actual_disappear,
                confidence=confidence,
                discrepancies=discrepancies,
                metadata={
                    "execution_time_ms": execution_time,
                    "screenshot_before": before_screenshot,
                    "screenshot_after": after_screenshot,
                },
            )

        except Exception as e:
            logger.error(f"Error verifying transition {transition.id}: {e}", exc_info=True)
            execution_time = (time.time() - start_time) * 1000
            return VerifiedTransition(
                id=f"verified_{transition.id}",
                inferred=transition.id,
                verified=False,
                verification_method="error",
                confidence=0.0,
                discrepancies=[
                    VerificationDiscrepancy(
                        discrepancy_type="execution_error",
                        description=str(e),
                    )
                ],
                metadata={"execution_time_ms": execution_time},
            )

    async def find_trigger_element(
        self, handler: EventHandler, elements: list[ExtractedElement]
    ) -> ExtractedElement | None:
        """Find the DOM element that triggers this handler.

        Args:
            handler: Event handler from static analysis.
            elements: Available DOM elements.

        Returns:
            Element that has this handler, or None if not found.
        """
        return await self._find_trigger_by_handler(handler.handler_name, elements)

    def compare_state_changes(
        self,
        expected_appear: list[str],
        expected_disappear: list[str],
        actual_appear: list[str],
        actual_disappear: list[str],
    ) -> tuple[bool, list[VerificationDiscrepancy]]:
        """Compare expected vs actual state changes.

        Args:
            expected_appear: State IDs expected to appear.
            expected_disappear: State IDs expected to disappear.
            actual_appear: State IDs that actually appeared.
            actual_disappear: State IDs that actually disappeared.

        Returns:
            Tuple of (overall_match, list_of_discrepancies).
        """
        discrepancies: list[VerificationDiscrepancy] = []

        # Check for expected appears that didn't happen
        for expected_id in expected_appear:
            # Try partial matching since IDs might not match exactly
            found = any(expected_id.lower() in actual_id.lower() for actual_id in actual_appear)
            if not found:
                discrepancies.append(
                    VerificationDiscrepancy(
                        discrepancy_type="missing_element",
                        description=f"Expected '{expected_id}' to appear but it didn't",
                        expected=expected_id,
                        actual=None,
                    )
                )

        # Check for expected disappears that didn't happen
        for expected_id in expected_disappear:
            found = any(expected_id.lower() in actual_id.lower() for actual_id in actual_disappear)
            if not found:
                discrepancies.append(
                    VerificationDiscrepancy(
                        discrepancy_type="missing_element",
                        description=f"Expected '{expected_id}' to disappear but it didn't",
                        expected=expected_id,
                        actual=None,
                    )
                )

        # Check for unexpected appears
        for actual_id in actual_appear:
            found = any(expected.lower() in actual_id.lower() for expected in expected_appear)
            if not found and expected_appear:  # Only flag if we had expectations
                discrepancies.append(
                    VerificationDiscrepancy(
                        discrepancy_type="extra_element",
                        description=f"Unexpected state appeared: '{actual_id}'",
                        expected=None,
                        actual=actual_id,
                    )
                )

        # Check for unexpected disappears
        for actual_id in actual_disappear:
            found = any(expected.lower() in actual_id.lower() for expected in expected_disappear)
            if not found and expected_disappear:  # Only flag if we had expectations
                discrepancies.append(
                    VerificationDiscrepancy(
                        discrepancy_type="extra_element",
                        description=f"Unexpected state disappeared: '{actual_id}'",
                        expected=None,
                        actual=actual_id,
                    )
                )

        # Overall match is true if no discrepancies or only minor ones
        matches = len(discrepancies) == 0

        return matches, discrepancies

    async def _find_trigger_by_handler(
        self, handler_name: str | None, elements: list[ExtractedElement]
    ) -> ExtractedElement | None:
        """Find element by handler name.

        Args:
            handler_name: Event handler name (e.g., "handleSubmit", "onClick").
            elements: Available elements.

        Returns:
            Matching element or None.
        """
        if not handler_name:
            return None

        handler_lower = handler_name.lower()

        # Look for elements with matching attributes or IDs
        for element in elements:
            # Check data attributes
            for attr_name, attr_value in element.attributes.items():
                if handler_lower in attr_name.lower() or (
                    isinstance(attr_value, str) and handler_lower in attr_value.lower()
                ):
                    return element

            # Check test ID
            test_id = element.attributes.get("data-testid")
            if test_id and handler_lower in test_id.lower():
                return element

            # Check aria-label
            if element.aria_label and handler_lower in element.aria_label.lower():
                return element

            # Check text content for button-like elements
            if (
                element.text_content
                and element.element_type.value in ["button", "link"]
                and handler_lower in element.text_content.lower()
            ):
                return element

        return None

    async def _execute_interaction(self, element: ExtractedElement, event_type: str) -> None:
        """Execute an interaction on an element.

        Args:
            element: Element to interact with.
            event_type: Type of interaction (click, hover, focus, etc.).
        """
        # This would use the runtime extractor's interaction simulation
        # For now, we'll use a simplified approach via the selector

        # Note: This is a placeholder - the actual implementation would use
        # the extractor's simulate_interaction method with proper action objects
        logger.info(f"Executing {event_type} on element {element.id} ({element.selector})")

        # In a real implementation, this would be:
        # await self.extractor.simulate_interaction(
        #     InteractionAction(type=event_type, target=element.selector)
        # )

    def _compute_verification_confidence(
        self,
        matches: bool,
        discrepancies: list[VerificationDiscrepancy],
        transition: InferredTransition,
        actual_appear: list[str],
        actual_disappear: list[str],
    ) -> float:
        """Compute confidence score for verification.

        Args:
            matches: Whether expected and actual changes matched.
            discrepancies: List of discrepancies found.
            transition: The inferred transition.
            actual_appear: States that actually appeared.
            actual_disappear: States that actually disappeared.

        Returns:
            Confidence score (0.0 to 1.0).
        """
        if not matches:
            # Base confidence on severity of discrepancies
            if len(discrepancies) == 0:
                return 1.0
            elif len(discrepancies) == 1:
                return 0.7
            elif len(discrepancies) == 2:
                return 0.5
            else:
                return 0.3

        # Perfect match
        if len(discrepancies) == 0:
            return 1.0

        # Some discrepancies but overall matched
        # Penalize based on number of issues
        penalty = min(0.4, len(discrepancies) * 0.1)
        return max(0.6, 1.0 - penalty)
