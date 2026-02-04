"""Metadata and Confidence Score Generation for the Analysis Pipeline.

This module provides metadata generation functionality including confidence
scores and quality metrics for analysis results.
"""

import logging
from typing import Any

from qontinui.discovery.state_analysis.models import DetectedState, Transition

logger = logging.getLogger(__name__)


class MetadataGenerator:
    """Handles metadata generation and confidence scoring.

    This class calculates various quality metrics and confidence scores
    for the analysis results.
    """

    def __init__(self):
        """Initialize the MetadataGenerator."""
        pass

    def calculate_confidence_scores(
        self, states: list[DetectedState], transitions: list[Transition]
    ) -> dict[str, float]:
        """Calculate confidence scores for the analysis.

        Args:
            states: Detected states
            transitions: Detected transitions

        Returns:
            Dictionary of confidence metrics
        """
        scores = {}

        # Average state image count (more images = higher confidence)
        if states:
            avg_images_per_state = sum(len(s.state_images) for s in states) / len(states)  # type: ignore[misc,attr-defined]
            scores["avg_state_images"] = avg_images_per_state

            # State coverage (what percentage of frames are in states)
            total_state_frames = sum(len(s.frame_indices) for s in states)  # type: ignore[misc,attr-defined]
            scores["state_coverage"] = total_state_frames / max(
                1,
                max(s.end_frame_index for s in states),  # type: ignore[attr-defined]
            )

        # Transition coverage (what percentage of states have transitions)
        if states and transitions:
            states_with_transitions = set()
            for t in transitions:
                states_with_transitions.update(t.source_states)
                states_with_transitions.update(t.target_states)

            scores["transition_coverage"] = len(states_with_transitions) / len(states)

        # Average transition confidence
        if transitions:
            scores["avg_transition_confidence"] = sum(
                t.recognition_confidence for t in transitions
            ) / len(transitions)

        return scores

    def generate_recommendation(
        self, result1_data: dict[str, Any], result2_data: dict[str, Any]
    ) -> str:
        """Generate a recommendation on which result is better.

        Args:
            result1_data: First result metrics
            result2_data: Second result metrics

        Returns:
            Recommendation string
        """
        # Simple heuristic: prefer more states and transitions with fewer errors
        score1 = (
            result1_data["num_states"]
            + result1_data["num_transitions"] * 2
            - result1_data["errors"] * 10
        )
        score2 = (
            result2_data["num_states"]
            + result2_data["num_transitions"] * 2
            - result2_data["errors"] * 10
        )

        if score2 > score1:
            return "Result 2 appears better (more states/transitions, fewer errors)"
        elif score1 > score2:
            return "Result 1 appears better (more states/transitions, fewer errors)"
        else:
            return "Results are comparable in quality"
