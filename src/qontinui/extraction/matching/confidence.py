"""
Confidence scoring for state matching.

Computes confidence scores for correlated states and verified transitions
based on the quality and quantity of evidence.
"""

import logging

from ..models.correlated import (
    CorrelatedState,
    EvidenceType,
    MatchingEvidence,
    VerifiedTransition,
)

logger = logging.getLogger(__name__)


# Evidence type weights - how much we trust each type of evidence
EVIDENCE_WEIGHTS = {
    EvidenceType.TEST_ID_MATCH: 1.0,  # Highest confidence - explicit test IDs
    EvidenceType.RUNTIME_VERIFIED: 1.0,  # Runtime verification is definitive
    EvidenceType.ARIA_MATCH: 0.8,  # ARIA attributes are semantic and reliable
    EvidenceType.CLASS_NAME_MATCH: 0.7,  # Class names often reflect component names
    EvidenceType.NAME_MATCH: 0.7,  # Name matches are pretty reliable
    EvidenceType.TEXT_CONTENT_MATCH: 0.6,  # Text can be reliable but might change
    EvidenceType.STRUCTURAL_MATCH: 0.5,  # Structure is helpful but indirect
    EvidenceType.SELECTOR_MATCH: 0.6,  # Selectors are moderately reliable
}


def compute_state_confidence(
    state: CorrelatedState, evidence: list[MatchingEvidence]
) -> float:
    """Compute overall confidence score for a correlated state.

    Factors:
    - Number of matching evidence items
    - Strength of each evidence type
    - Consistency across multiple matches

    Args:
        state: Correlated state to score.
        evidence: List of matching evidence supporting this state.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    if not evidence:
        return 0.0

    # Collect weighted scores
    weighted_scores: list[float] = []

    for item in evidence:
        # Get base weight for this evidence type
        base_weight = EVIDENCE_WEIGHTS.get(item.evidence_type, 0.5)

        # Apply the item's own strength
        weighted_score = base_weight * item.strength

        weighted_scores.append(weighted_score)

    if not weighted_scores:
        return 0.0

    # Use a combination of average and max to balance between:
    # - Having multiple pieces of evidence (average)
    # - Having at least one strong piece of evidence (max)
    avg_score = sum(weighted_scores) / len(weighted_scores)
    max_score = max(weighted_scores)

    # Weighted combination (60% max, 40% avg)
    # This rewards having one strong match while still valuing multiple sources
    combined_score = 0.6 * max_score + 0.4 * avg_score

    # Bonus for having multiple independent evidence types
    unique_types = len({item.evidence_type for item in evidence})
    if unique_types > 1:
        # Up to 10% bonus for having multiple evidence types
        diversity_bonus = min(0.1, (unique_types - 1) * 0.03)
        combined_score = min(1.0, combined_score + diversity_bonus)

    # Penalty if we only have weak evidence
    if all(score < 0.5 for score in weighted_scores):
        combined_score *= 0.8

    return min(1.0, max(0.0, combined_score))


def compute_transition_confidence(transition: VerifiedTransition) -> float:
    """Compute confidence for a verified transition.

    Factors:
    - Was it verified at runtime?
    - Did actual changes match expected?
    - Number of discrepancies

    Args:
        transition: Verified transition to score.

    Returns:
        Confidence score between 0.0 and 1.0.
    """
    # If verification failed completely
    if transition.error:
        return 0.0

    # If not verified at all
    if not transition.verified:
        # Check if we at least executed something
        if transition.actual_appear or transition.actual_disappear:
            # Something happened, just not what we expected
            return 0.3
        return 0.0

    # Base confidence on number of discrepancies
    num_discrepancies = len(transition.discrepancies)

    if num_discrepancies == 0:
        # Perfect match - very high confidence
        confidence = 1.0
    elif num_discrepancies == 1:
        # One discrepancy - still good
        confidence = 0.8
    elif num_discrepancies == 2:
        # Two discrepancies - moderate confidence
        confidence = 0.6
    elif num_discrepancies == 3:
        # Three discrepancies - lower confidence
        confidence = 0.4
    else:
        # Many discrepancies - low confidence
        confidence = 0.2

    # Adjust based on whether we got any correct predictions
    expected_count = len(transition.inferred.expected_appear) + len(
        transition.inferred.expected_disappear
    )
    actual_count = len(transition.actual_appear) + len(transition.actual_disappear)

    if expected_count > 0:
        # Bonus if we got the rough magnitude right
        magnitude_ratio = min(actual_count, expected_count) / max(
            actual_count, expected_count, 1
        )
        if magnitude_ratio > 0.7:
            confidence = min(1.0, confidence + 0.1)

    # Penalty if we expected changes but nothing happened
    if expected_count > 0 and actual_count == 0:
        confidence *= 0.5

    # Bonus for fast execution (indicates reliable transition)
    if transition.execution_time_ms < 500:
        confidence = min(1.0, confidence + 0.05)

    return min(1.0, max(0.0, confidence))


def get_evidence_summary(evidence: list[MatchingEvidence]) -> dict[str, int]:
    """Get a summary of evidence types.

    Args:
        evidence: List of matching evidence.

    Returns:
        Dictionary mapping evidence type to count.
    """
    summary: dict[str, int] = {}

    for item in evidence:
        type_name = item.evidence_type.value
        summary[type_name] = summary.get(type_name, 0) + 1

    return summary


def get_strongest_evidence(evidence: list[MatchingEvidence]) -> MatchingEvidence | None:
    """Get the strongest piece of evidence.

    Args:
        evidence: List of matching evidence.

    Returns:
        Strongest evidence item or None.
    """
    if not evidence:
        return None

    def evidence_score(item: MatchingEvidence) -> float:
        base_weight = EVIDENCE_WEIGHTS.get(item.evidence_type, 0.5)
        return base_weight * item.strength

    return max(evidence, key=evidence_score)


def filter_weak_evidence(
    evidence: list[MatchingEvidence], threshold: float = 0.3
) -> list[MatchingEvidence]:
    """Filter out weak evidence below a threshold.

    Args:
        evidence: List of matching evidence.
        threshold: Minimum strength threshold (0.0 to 1.0).

    Returns:
        Filtered list of evidence.
    """
    return [item for item in evidence if item.strength >= threshold]


def combine_evidence(
    evidence_lists: list[list[MatchingEvidence]],
) -> list[MatchingEvidence]:
    """Combine multiple evidence lists, removing duplicates.

    Args:
        evidence_lists: List of evidence lists to combine.

    Returns:
        Combined and deduplicated evidence list.
    """
    seen: set[tuple[str, str, str]] = set()
    combined: list[MatchingEvidence] = []

    for evidence_list in evidence_lists:
        for item in evidence_list:
            # Create a unique key for deduplication
            key = (
                item.evidence_type.value,
                item.static_reference or "",
                item.runtime_reference or "",
            )

            if key not in seen:
                seen.add(key)
                combined.append(item)

    return combined
