"""Detect transitions between discovered states."""

import logging
from typing import Any

import numpy as np

from ...models import DiscoveredState, StateTransition

logger = logging.getLogger(__name__)


class TransitionDetector:
    """Detects potential transitions between discovered states."""

    def detect(
        self, states: list[DiscoveredState], screenshots: list[np.ndarray[Any, Any]]
    ) -> list[StateTransition]:
        """
        Find potential transitions between states.

        Analyzes sequential screenshots to identify state changes by:
        1. Mapping screenshots to states based on state_image matches
        2. Detecting consecutive different states
        3. Computing confidence based on visual difference

        Args:
            states: List of discovered states
            screenshots: Original screenshots

        Returns:
            List of state transitions
        """
        transitions: list[StateTransition] = []

        if len(states) < 2 or len(screenshots) < 2:
            logger.debug("Not enough states or screenshots for transition detection")
            return transitions

        # Create mapping of screenshot IDs to states
        screenshot_to_state: dict[str, DiscoveredState] = {}
        for state in states:
            for screenshot_id in state.screenshot_ids:
                screenshot_to_state[screenshot_id] = state

        # Detect transitions by analyzing sequence
        prev_state: DiscoveredState | None = None
        for i in range(len(screenshots)):
            screenshot_id = str(i)
            current_state = screenshot_to_state.get(screenshot_id)

            if current_state and prev_state and current_state.id != prev_state.id:
                # Detected a state transition
                confidence = self._calculate_transition_confidence(
                    screenshots[i - 1] if i > 0 else screenshots[i],
                    screenshots[i],
                )

                transition = StateTransition(
                    from_state=prev_state.id,
                    to_state=current_state.id,
                    trigger_image=None,  # Could be enhanced to identify trigger element
                    confidence=confidence,
                )

                # Avoid duplicate transitions
                if not self._is_duplicate_transition(transitions, transition):
                    transitions.append(transition)
                    logger.info(
                        f"Detected transition: {prev_state.name} -> {current_state.name} "
                        f"(confidence: {confidence:.2f})"
                    )

            if current_state:
                prev_state = current_state

        logger.info(f"Detected {len(transitions)} state transitions")
        return transitions

    def _calculate_transition_confidence(
        self, prev_screenshot: np.ndarray[Any, Any], current_screenshot: np.ndarray[Any, Any]
    ) -> float:
        """
        Calculate confidence score for a transition based on visual difference.

        Args:
            prev_screenshot: Previous screenshot
            current_screenshot: Current screenshot

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Ensure same shape
        if prev_screenshot.shape != current_screenshot.shape:
            return 0.5  # Medium confidence if shapes differ

        # Calculate pixel-wise difference
        diff = np.abs(current_screenshot.astype(np.float32) - prev_screenshot.astype(np.float32))
        diff_magnitude = np.mean(diff)

        # Normalize to 0-1 range (assuming 0-255 pixel values)
        # Higher difference = more distinct transition = higher confidence
        normalized_diff = min(diff_magnitude / 128.0, 1.0)  # 128 is mid-range of 0-255

        # Scale to reasonable confidence range (0.5 to 1.0)
        # Even small visual changes indicate a transition
        confidence = 0.5 + (normalized_diff * 0.5)

        return confidence

    def _is_duplicate_transition(
        self, transitions: list[StateTransition], new_transition: StateTransition
    ) -> bool:
        """
        Check if this transition already exists in the list.

        Args:
            transitions: Existing transitions
            new_transition: New transition to check

        Returns:
            True if duplicate exists
        """
        for transition in transitions:
            if (
                transition.from_state == new_transition.from_state
                and transition.to_state == new_transition.to_state
            ):
                return True
        return False
