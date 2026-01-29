"""Transition Detection from UI Bridge Actions.

This module detects transitions by tracking which actions cause state changes.
It builds a transition graph over time from observed action -> state-change
relationships.

Features:
- Real-time transition detection
- Continuous reliability score updates
- Bidirectional transition detection
- Dynamic pathfinding cost adjustment
- Event streaming for discovered transitions

Example:
    from qontinui.state_machine import TransitionDetector

    detector = TransitionDetector()

    # Record action and resulting state change
    detector.record_action(
        action={"type": "click", "elementId": "settings-btn"},
        before_states=["dashboard"],
        after_states=["dashboard", "settings_panel"],
    )

    # Get detected transitions
    transitions = detector.get_detected_transitions()

    # Real-time mode with callbacks
    detector.set_on_transition_discovered(callback_fn)
    detector.record_action_realtime(action, before, after)
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class TransitionReliability:
    """Reliability metrics for a transition."""

    transition_id: str
    success_count: int = 0
    failure_count: int = 0
    total_time_ms: float = 0.0
    last_success: datetime | None = None
    last_failure: datetime | None = None

    @property
    def total_attempts(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        if self.total_attempts == 0:
            return 1.0
        return self.success_count / self.total_attempts

    @property
    def average_time_ms(self) -> float:
        if self.total_attempts == 0:
            return 0.0
        return self.total_time_ms / self.total_attempts

    def record_success(self, duration_ms: float = 0.0) -> None:
        """Record successful execution."""
        self.success_count += 1
        self.total_time_ms += duration_ms
        self.last_success = datetime.utcnow()

    def record_failure(self, duration_ms: float = 0.0) -> None:
        """Record failed execution."""
        self.failure_count += 1
        self.total_time_ms += duration_ms
        self.last_failure = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "transitionId": self.transition_id,
            "successCount": self.success_count,
            "failureCount": self.failure_count,
            "totalAttempts": self.total_attempts,
            "successRate": self.success_rate,
            "averageTimeMs": self.average_time_ms,
            "lastSuccess": self.last_success.isoformat() if self.last_success else None,
            "lastFailure": self.last_failure.isoformat() if self.last_failure else None,
        }


@dataclass
class DetectedTransition:
    """A transition detected from observed action -> state-change."""

    id: str
    name: str
    from_states: list[str]
    to_states: list[str]
    actions: list[dict[str, Any]]
    activate_states: list[str]
    exit_states: list[str]
    observation_count: int = 1
    confidence: float = 0.0
    reliability: TransitionReliability | None = None
    first_observed: datetime = field(default_factory=datetime.utcnow)
    last_observed: datetime = field(default_factory=datetime.utcnow)
    is_bidirectional: bool = False
    reverse_transition_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "fromStates": self.from_states,
            "toStates": self.to_states,
            "actions": self.actions,
            "activateStates": self.activate_states,
            "exitStates": self.exit_states,
            "observationCount": self.observation_count,
            "confidence": self.confidence,
            "reliability": self.reliability.to_dict() if self.reliability else None,
            "firstObserved": self.first_observed.isoformat(),
            "lastObserved": self.last_observed.isoformat(),
            "isBidirectional": self.is_bidirectional,
            "reverseTransitionId": self.reverse_transition_id,
            "metadata": self.metadata,
        }

    def update_observation(self, success: bool, duration_ms: float = 0.0) -> None:
        """Update observation with result."""
        self.observation_count += 1
        self.last_observed = datetime.utcnow()
        if self.reliability:
            if success:
                self.reliability.record_success(duration_ms)
            else:
                self.reliability.record_failure(duration_ms)


@dataclass
class TransitionDetectorConfig:
    """Configuration for transition detection."""

    # Minimum observations to confirm a transition
    min_observation_count: int = 2

    # Confidence threshold for confirmed transitions
    confidence_threshold: float = 0.7

    # Maximum age (seconds) for recent transitions
    recent_window_seconds: int = 300

    # Whether to track reliability metrics
    track_reliability: bool = True

    # Maximum transitions to keep in memory
    max_transitions: int = 1000

    # Real-time mode settings
    enable_realtime_mode: bool = True
    action_correlation_window_ms: int = 2000

    # Bidirectional detection
    detect_bidirectional: bool = True

    # Event callbacks for real-time notifications
    on_transition_discovered: Callable[[DetectedTransition], None] | None = None
    on_transition_updated: Callable[[DetectedTransition], None] | None = None
    on_reliability_changed: Callable[[str, float], None] | None = None


@dataclass
class ActionRecord:
    """Record of an action and its state change."""

    action: dict[str, Any]
    before_states: set[str]
    after_states: set[str]
    timestamp: datetime
    duration_ms: float = 0.0
    success: bool = True

    @property
    def activated_states(self) -> set[str]:
        """States that were activated by this action."""
        return self.after_states - self.before_states

    @property
    def exited_states(self) -> set[str]:
        """States that were exited by this action."""
        return self.before_states - self.after_states


class TransitionDetector:
    """Detects transitions from observed action -> state-change relationships.

    This class builds a transition graph by:
    1. Recording actions and their resulting state changes
    2. Grouping similar action -> state-change patterns
    3. Calculating confidence based on observation frequency
    4. Tracking reliability metrics for pathfinding cost

    The detected transitions can be used for:
    - Automatic transition discovery
    - Pathfinding cost optimization
    - Transition reliability monitoring
    """

    def __init__(self, config: TransitionDetectorConfig | None = None) -> None:
        """Initialize transition detector.

        Args:
            config: Detection configuration
        """
        self.config = config or TransitionDetectorConfig()

        # Detected transitions: transition_id -> DetectedTransition
        self._transitions: dict[str, DetectedTransition] = {}

        # Action history for pattern analysis
        self._action_history: list[ActionRecord] = []

        # Pattern signatures: signature -> list of matching action records
        self._pattern_signatures: dict[str, list[ActionRecord]] = {}

    def record_action(
        self,
        action: dict[str, Any],
        before_states: list[str] | set[str],
        after_states: list[str] | set[str],
        duration_ms: float = 0.0,
        success: bool = True,
    ) -> DetectedTransition | None:
        """Record an action and its resulting state change.

        Args:
            action: Action that was executed
            before_states: States before action
            after_states: States after action
            duration_ms: Action duration in milliseconds
            success: Whether action succeeded

        Returns:
            Detected transition if pattern is confirmed
        """
        before_set = set(before_states) if isinstance(before_states, list) else before_states
        after_set = set(after_states) if isinstance(after_states, list) else after_states

        # Create action record
        record = ActionRecord(
            action=action.copy(),
            before_states=before_set,
            after_states=after_set,
            timestamp=datetime.utcnow(),
            duration_ms=duration_ms,
            success=success,
        )

        # Store in history
        self._action_history.append(record)

        # Trim history if too large
        if len(self._action_history) > self.config.max_transitions * 10:
            self._action_history = self._action_history[-self.config.max_transitions * 5 :]

        # Generate pattern signature
        signature = self._generate_pattern_signature(record)

        # Add to pattern tracking
        if signature not in self._pattern_signatures:
            self._pattern_signatures[signature] = []
        self._pattern_signatures[signature].append(record)

        # Check if we have enough observations
        return self._update_transition(signature, record)

    def _generate_pattern_signature(self, record: ActionRecord) -> str:
        """Generate a signature for an action -> state-change pattern.

        Args:
            record: Action record

        Returns:
            Pattern signature string
        """
        # Include action type and target element
        action_type = record.action.get("type", "click")
        element_id = record.action.get("elementId") or record.action.get("target", "unknown")

        # Include state changes (sorted for consistency)
        activated = "|".join(sorted(record.activated_states))
        exited = "|".join(sorted(record.exited_states))

        # Create signature
        sig_parts = [
            f"action:{action_type}",
            f"element:{element_id}",
            f"activated:{activated}",
            f"exited:{exited}",
        ]

        return hashlib.sha256("::".join(sig_parts).encode()).hexdigest()[:16]

    def _update_transition(self, signature: str, record: ActionRecord) -> DetectedTransition | None:
        """Update or create transition based on pattern observations.

        Args:
            signature: Pattern signature
            record: Latest action record

        Returns:
            Confirmed transition if threshold met
        """
        records = self._pattern_signatures.get(signature, [])

        if len(records) < self.config.min_observation_count:
            return None

        # Generate transition ID from signature
        transition_id = f"trans_{signature}"

        if transition_id in self._transitions:
            # Update existing transition
            trans = self._transitions[transition_id]
            trans.observation_count = len(records)
            trans.last_observed = record.timestamp
            trans.confidence = self._calculate_confidence(records)

            # Update reliability
            if trans.reliability and self.config.track_reliability:
                if record.success:
                    trans.reliability.record_success(record.duration_ms)
                else:
                    trans.reliability.record_failure(record.duration_ms)

            return trans if trans.confidence >= self.config.confidence_threshold else None

        # Create new transition
        trans = self._create_transition(transition_id, records)
        self._transitions[transition_id] = trans

        # Prune old transitions if too many
        if len(self._transitions) > self.config.max_transitions:
            self._prune_transitions()

        return trans if trans.confidence >= self.config.confidence_threshold else None

    def _create_transition(
        self, transition_id: str, records: list[ActionRecord]
    ) -> DetectedTransition:
        """Create a detected transition from observation records.

        Args:
            transition_id: Transition ID
            records: Observation records

        Returns:
            DetectedTransition
        """
        # Use first record as template
        first_record = records[0]
        last_record = records[-1]

        # Get action from first record
        action = first_record.action.copy()

        # Generate name
        action_type = action.get("type", "click")
        element_id = action.get("elementId") or action.get("target", "element")
        name = f"{action_type.title()} {element_id}"

        # Calculate state changes from all records
        from_states = set()
        activated_states = set()
        exited_states = set()

        for record in records:
            from_states.update(record.before_states)
            activated_states.update(record.activated_states)
            exited_states.update(record.exited_states)

        # Create reliability tracker
        reliability: TransitionReliability | None = None
        if self.config.track_reliability:
            reliability = TransitionReliability(transition_id=transition_id)
            for record in records:
                if record.success:
                    reliability.record_success(record.duration_ms)
                else:
                    reliability.record_failure(record.duration_ms)

        return DetectedTransition(
            id=transition_id,
            name=name,
            from_states=sorted(from_states),
            to_states=sorted(activated_states | (from_states - exited_states)),
            actions=[action],
            activate_states=sorted(activated_states),
            exit_states=sorted(exited_states),
            observation_count=len(records),
            confidence=self._calculate_confidence(records),
            reliability=reliability,
            first_observed=first_record.timestamp,
            last_observed=last_record.timestamp,
        )

    def _calculate_confidence(self, records: list[ActionRecord]) -> float:
        """Calculate confidence score for a transition pattern.

        Based on:
        - Number of observations
        - Consistency of state changes
        - Success rate

        Args:
            records: Observation records

        Returns:
            Confidence score (0.0-1.0)
        """
        if not records:
            return 0.0

        # Factor 1: Observation count (log scale)
        import math

        observation_score = min(1.0, math.log10(len(records) + 1) / 2)

        # Factor 2: Consistency of state changes
        consistency_score = self._calculate_consistency(records)

        # Factor 3: Success rate
        success_count = sum(1 for r in records if r.success)
        success_score = success_count / len(records)

        # Weighted combination
        weights = (0.3, 0.4, 0.3)  # observations, consistency, success
        confidence = (
            weights[0] * observation_score
            + weights[1] * consistency_score
            + weights[2] * success_score
        )

        return min(1.0, confidence)

    def _calculate_consistency(self, records: list[ActionRecord]) -> float:
        """Calculate consistency of state changes across records.

        Args:
            records: Observation records

        Returns:
            Consistency score (0.0-1.0)
        """
        if len(records) < 2:
            return 1.0

        # Compare state changes across all records
        activated_sets = [r.activated_states for r in records]
        exited_sets = [r.exited_states for r in records]

        # Calculate Jaccard similarity for activated states
        activated_sim = self._average_set_similarity(activated_sets)

        # Calculate Jaccard similarity for exited states
        exited_sim = self._average_set_similarity(exited_sets)

        return (activated_sim + exited_sim) / 2

    def _average_set_similarity(self, sets: list[set[str]]) -> float:
        """Calculate average pairwise Jaccard similarity.

        Args:
            sets: List of sets

        Returns:
            Average similarity (0.0-1.0)
        """
        if len(sets) < 2:
            return 1.0

        total_sim = 0.0
        pair_count = 0

        for i, s1 in enumerate(sets):
            for s2 in sets[i + 1 :]:
                if not s1 and not s2:
                    sim = 1.0  # Both empty = same
                elif not s1 or not s2:
                    sim = 0.0  # One empty = different
                else:
                    intersection = len(s1 & s2)
                    union = len(s1 | s2)
                    sim = intersection / union if union > 0 else 0.0

                total_sim += sim
                pair_count += 1

        return total_sim / pair_count if pair_count > 0 else 1.0

    def _prune_transitions(self) -> None:
        """Prune old/low-confidence transitions to stay under limit."""
        if len(self._transitions) <= self.config.max_transitions:
            return

        # Sort by (confidence, last_observed) descending
        sorted_trans = sorted(
            self._transitions.items(),
            key=lambda x: (x[1].confidence, x[1].last_observed),
            reverse=True,
        )

        # Keep top N
        keep_ids = {t[0] for t in sorted_trans[: self.config.max_transitions]}
        self._transitions = {
            tid: trans for tid, trans in self._transitions.items() if tid in keep_ids
        }

    def get_detected_transitions(
        self, min_confidence: float | None = None
    ) -> list[DetectedTransition]:
        """Get all detected transitions.

        Args:
            min_confidence: Minimum confidence threshold (uses config default if None)

        Returns:
            List of detected transitions
        """
        threshold = min_confidence or self.config.confidence_threshold

        return [trans for trans in self._transitions.values() if trans.confidence >= threshold]

    def get_transition(self, transition_id: str) -> DetectedTransition | None:
        """Get a specific transition by ID.

        Args:
            transition_id: Transition ID

        Returns:
            DetectedTransition or None
        """
        return self._transitions.get(transition_id)

    def get_transitions_from_state(self, state_id: str) -> list[DetectedTransition]:
        """Get transitions that can execute from a state.

        Args:
            state_id: State ID

        Returns:
            List of available transitions
        """
        return [
            trans
            for trans in self._transitions.values()
            if state_id in trans.from_states
            and trans.confidence >= self.config.confidence_threshold
        ]

    def get_transitions_to_state(self, state_id: str) -> list[DetectedTransition]:
        """Get transitions that lead to a state.

        Args:
            state_id: Target state ID

        Returns:
            List of transitions that activate this state
        """
        return [
            trans
            for trans in self._transitions.values()
            if state_id in trans.activate_states
            and trans.confidence >= self.config.confidence_threshold
        ]

    def record_transition_result(
        self, transition_id: str, success: bool, duration_ms: float = 0.0
    ) -> None:
        """Record the result of executing a transition.

        Args:
            transition_id: Transition that was executed
            success: Whether it succeeded
            duration_ms: Execution duration
        """
        trans = self._transitions.get(transition_id)
        if not trans or not trans.reliability:
            return

        if success:
            trans.reliability.record_success(duration_ms)
        else:
            trans.reliability.record_failure(duration_ms)

        trans.last_observed = datetime.utcnow()

    def get_transition_cost(self, transition_id: str, base_cost: float = 1.0) -> float:
        """Get adjusted pathfinding cost for a transition.

        Cost is adjusted based on reliability.

        Args:
            transition_id: Transition ID
            base_cost: Base path cost

        Returns:
            Adjusted cost
        """
        trans = self._transitions.get(transition_id)
        if not trans or not trans.reliability:
            return base_cost

        # Adjust cost based on success rate
        success_rate = trans.reliability.success_rate

        if success_rate >= 0.9:
            return base_cost  # No penalty
        elif success_rate >= 0.7:
            return base_cost * 1.5
        elif success_rate >= 0.5:
            return base_cost * 2.0
        else:
            return base_cost * 3.0  # High penalty for unreliable transitions

    def get_statistics(self) -> dict[str, Any]:
        """Get detector statistics.

        Returns:
            Dictionary with statistics
        """
        confirmed = [
            t
            for t in self._transitions.values()
            if t.confidence >= self.config.confidence_threshold
        ]

        return {
            "total_transitions": len(self._transitions),
            "confirmed_transitions": len(confirmed),
            "pending_patterns": len(self._pattern_signatures) - len(self._transitions),
            "action_history_size": len(self._action_history),
            "average_confidence": (
                sum(t.confidence for t in self._transitions.values()) / len(self._transitions)
                if self._transitions
                else 0.0
            ),
            "average_observations": (
                sum(t.observation_count for t in self._transitions.values())
                / len(self._transitions)
                if self._transitions
                else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset all detection state."""
        self._transitions.clear()
        self._action_history.clear()
        self._pattern_signatures.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize detector state to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "transitions": {tid: t.to_dict() for tid, t in self._transitions.items()},
            "statistics": self.get_statistics(),
        }

    # =========================================================================
    # Real-Time Learning
    # =========================================================================

    def record_action_realtime(
        self,
        action: dict[str, Any],
        before_states: list[str] | set[str],
        after_states: list[str] | set[str],
        duration_ms: float = 0.0,
        success: bool = True,
    ) -> DetectedTransition | None:
        """Record action with real-time transition detection.

        Similar to record_action but emits events for discovered transitions.

        Args:
            action: Action that was executed
            before_states: States before action
            after_states: States after action
            duration_ms: Action duration in milliseconds
            success: Whether action succeeded

        Returns:
            Detected transition if confirmed
        """
        # Record the action
        transition = self.record_action(
            action=action,
            before_states=before_states,
            after_states=after_states,
            duration_ms=duration_ms,
            success=success,
        )

        # Emit events if transition was detected/updated
        if transition:
            is_new = transition.observation_count == self.config.min_observation_count

            if is_new and self.config.on_transition_discovered:
                self.config.on_transition_discovered(transition)

                # Check for bidirectional
                if self.config.detect_bidirectional:
                    self._detect_bidirectional_transition(transition)
            elif not is_new and self.config.on_transition_updated:
                self.config.on_transition_updated(transition)

            # Emit reliability change
            if self.config.on_reliability_changed and transition.reliability:
                self.config.on_reliability_changed(
                    transition.id, transition.reliability.success_rate
                )

        return transition

    def _detect_bidirectional_transition(self, transition: DetectedTransition) -> None:
        """Detect if this transition has a reverse (bidirectional).

        Args:
            transition: Transition to check
        """
        for existing_id, existing in self._transitions.items():
            if existing_id == transition.id:
                continue

            # Check if this is the reverse transition
            is_reverse = set(existing.activate_states) == set(transition.exit_states) and set(
                existing.exit_states
            ) == set(transition.activate_states)

            if is_reverse:
                transition.is_bidirectional = True
                transition.reverse_transition_id = existing_id
                existing.is_bidirectional = True
                existing.reverse_transition_id = transition.id

                logger.info(
                    f"Detected bidirectional transitions: {transition.id} <-> {existing_id}"
                )
                break

    def update_reliability_continuous(
        self,
        transition_id: str,
        success: bool,
        duration_ms: float = 0.0,
    ) -> float | None:
        """Update reliability score continuously.

        Called after each transition execution to maintain accurate scores.

        Args:
            transition_id: Transition that was executed
            success: Whether execution succeeded
            duration_ms: Execution duration

        Returns:
            New reliability score or None if transition not found
        """
        trans = self._transitions.get(transition_id)
        if not trans:
            return None

        # Update reliability
        self.record_transition_result(transition_id, success, duration_ms)

        # Update last observed
        trans.last_observed = datetime.utcnow()

        # Recalculate confidence
        records = self._pattern_signatures.get(self._get_signature_for_transition(trans), [])
        trans.confidence = self._calculate_confidence(records) if records else 0.5

        # Emit callback
        if self.config.on_reliability_changed and trans.reliability:
            self.config.on_reliability_changed(transition_id, trans.reliability.success_rate)

        return trans.reliability.success_rate if trans.reliability else None

    def _get_signature_for_transition(self, trans: DetectedTransition) -> str:
        """Get pattern signature for a detected transition.

        Args:
            trans: Transition to get signature for

        Returns:
            Pattern signature
        """
        if not trans.actions:
            return ""

        action = trans.actions[0]
        action_type = action.get("type", "click")
        element_id = action.get("elementId") or action.get("target", "unknown")
        activated = "|".join(sorted(trans.activate_states))
        exited = "|".join(sorted(trans.exit_states))

        sig_parts = [
            f"action:{action_type}",
            f"element:{element_id}",
            f"activated:{activated}",
            f"exited:{exited}",
        ]

        return hashlib.sha256("::".join(sig_parts).encode()).hexdigest()[:16]

    # =========================================================================
    # Event Callbacks
    # =========================================================================

    def set_on_transition_discovered(
        self, callback: Callable[[DetectedTransition], None] | None
    ) -> None:
        """Set callback for when new transition is discovered.

        Args:
            callback: Function to call with discovered transition
        """
        self.config.on_transition_discovered = callback

    def set_on_transition_updated(
        self, callback: Callable[[DetectedTransition], None] | None
    ) -> None:
        """Set callback for when existing transition is updated.

        Args:
            callback: Function to call with updated transition
        """
        self.config.on_transition_updated = callback

    def set_on_reliability_changed(self, callback: Callable[[str, float], None] | None) -> None:
        """Set callback for when transition reliability changes.

        Args:
            callback: Function to call with (transition_id, new_reliability)
        """
        self.config.on_reliability_changed = callback

    # =========================================================================
    # Advanced Queries
    # =========================================================================

    def get_bidirectional_pairs(self) -> list[tuple[DetectedTransition, DetectedTransition]]:
        """Get all bidirectional transition pairs.

        Returns:
            List of (forward, reverse) transition pairs
        """
        pairs: list[tuple[DetectedTransition, DetectedTransition]] = []
        seen: set[str] = set()

        for trans in self._transitions.values():
            if not trans.is_bidirectional or trans.id in seen:
                continue

            reverse_id = trans.reverse_transition_id
            if reverse_id and reverse_id in self._transitions:
                reverse = self._transitions[reverse_id]
                pairs.append((trans, reverse))
                seen.add(trans.id)
                seen.add(reverse_id)

        return pairs

    def get_transitions_by_element(self, element_id: str) -> list[DetectedTransition]:
        """Get all transitions triggered by a specific element.

        Args:
            element_id: Element ID to search for

        Returns:
            List of transitions involving this element
        """
        return [
            trans
            for trans in self._transitions.values()
            if any(
                action.get("elementId") == element_id or action.get("target") == element_id
                for action in trans.actions
            )
        ]

    def get_most_reliable_path(
        self,
        from_state: str,
        to_state: str,
    ) -> list[DetectedTransition]:
        """Find most reliable transition path between states.

        Uses simple BFS weighted by reliability.

        Args:
            from_state: Starting state
            to_state: Target state

        Returns:
            List of transitions forming the path
        """
        from collections import deque

        # BFS with reliability weighting
        queue: deque[tuple[str, list[DetectedTransition], float]] = deque([(from_state, [], 1.0)])
        visited: set[str] = set()

        best_path: list[DetectedTransition] = []
        best_reliability = 0.0

        while queue:
            current_state, path, path_reliability = queue.popleft()

            if current_state in visited:
                continue
            visited.add(current_state)

            if current_state == to_state:
                if path_reliability > best_reliability:
                    best_path = path
                    best_reliability = path_reliability
                continue

            # Get available transitions
            for trans in self.get_transitions_from_state(current_state):
                reliability = trans.reliability.success_rate if trans.reliability else 0.5
                new_reliability = path_reliability * reliability

                for activated in trans.activate_states:
                    if activated not in visited:
                        queue.append(
                            (
                                activated,
                                path + [trans],
                                new_reliability,
                            )
                        )

        return best_path

    def merge_similar_transitions(
        self,
        similarity_threshold: float = 0.9,
    ) -> int:
        """Merge transitions with very similar patterns.

        Args:
            similarity_threshold: Minimum similarity to merge

        Returns:
            Number of transitions merged
        """
        merged_count = 0
        to_remove: list[str] = []

        transitions = list(self._transitions.values())

        for i, trans1 in enumerate(transitions):
            if trans1.id in to_remove:
                continue

            for trans2 in transitions[i + 1 :]:
                if trans2.id in to_remove:
                    continue

                # Calculate similarity
                similarity = self._calculate_transition_similarity(trans1, trans2)

                if similarity >= similarity_threshold:
                    # Merge trans2 into trans1
                    trans1.observation_count += trans2.observation_count
                    if trans1.reliability and trans2.reliability:
                        trans1.reliability.success_count += trans2.reliability.success_count
                        trans1.reliability.failure_count += trans2.reliability.failure_count
                        trans1.reliability.total_time_ms += trans2.reliability.total_time_ms
                    trans1.confidence = self._calculate_merged_confidence(trans1, trans2)

                    to_remove.append(trans2.id)
                    merged_count += 1

        # Remove merged transitions
        for tid in to_remove:
            del self._transitions[tid]

        logger.info(f"Merged {merged_count} similar transitions")
        return merged_count

    def _calculate_transition_similarity(
        self,
        trans1: DetectedTransition,
        trans2: DetectedTransition,
    ) -> float:
        """Calculate similarity between two transitions.

        Args:
            trans1: First transition
            trans2: Second transition

        Returns:
            Similarity score (0.0-1.0)
        """
        # Compare from_states
        from1 = set(trans1.from_states)
        from2 = set(trans2.from_states)
        from_sim = len(from1 & from2) / len(from1 | from2) if (from1 | from2) else 1.0

        # Compare activate_states
        act1 = set(trans1.activate_states)
        act2 = set(trans2.activate_states)
        act_sim = len(act1 & act2) / len(act1 | act2) if (act1 | act2) else 1.0

        # Compare exit_states
        exit1 = set(trans1.exit_states)
        exit2 = set(trans2.exit_states)
        exit_sim = len(exit1 & exit2) / len(exit1 | exit2) if (exit1 | exit2) else 1.0

        # Compare actions
        if trans1.actions and trans2.actions:
            a1 = trans1.actions[0]
            a2 = trans2.actions[0]
            action_sim = (
                1.0
                if (a1.get("type") == a2.get("type") and a1.get("elementId") == a2.get("elementId"))
                else 0.0
            )
        else:
            action_sim = 1.0

        # Weighted average
        return from_sim * 0.2 + act_sim * 0.3 + exit_sim * 0.3 + action_sim * 0.2

    def _calculate_merged_confidence(
        self,
        trans1: DetectedTransition,
        trans2: DetectedTransition,
    ) -> float:
        """Calculate confidence for merged transition.

        Args:
            trans1: Target transition
            trans2: Source transition

        Returns:
            New confidence score
        """
        total_obs = trans1.observation_count + trans2.observation_count
        weighted = (
            trans1.confidence * trans1.observation_count
            + trans2.confidence * trans2.observation_count
        )
        return weighted / total_obs if total_obs > 0 else 0.5
