"""Path reliability tracking for informed pathfinding.

Tracks actual transition success/failure rates at runtime to inform
pathfinding decisions. More reliable paths are preferred over
frequently-failing ones.

This module learns from actual execution history, not just static
transition metadata.
"""

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TransitionAttempt:
    """Record of a single transition attempt."""

    timestamp: float
    """Unix timestamp when attempt occurred."""

    success: bool
    """Whether the transition succeeded."""

    duration_ms: float | None = None
    """How long the transition took (milliseconds)."""

    failure_reason: str | None = None
    """Description of failure if success=False."""


@dataclass
class TransitionStats:
    """Aggregated statistics for a transition."""

    from_state: str
    """Source state identifier."""

    to_state: str
    """Target state identifier."""

    total_attempts: int
    """Total number of attempts."""

    successes: int
    """Number of successful attempts."""

    failures: int
    """Number of failed attempts."""

    avg_duration_ms: float | None
    """Average duration of successful attempts."""

    last_success: float | None
    """Timestamp of last successful attempt."""

    last_failure: float | None
    """Timestamp of last failed attempt."""

    @property
    def success_rate(self) -> float:
        """Calculate raw success rate (0.0-1.0)."""
        if self.total_attempts == 0:
            return 0.5  # Unknown, neutral
        return self.successes / self.total_attempts

    @property
    def is_failing(self) -> bool:
        """Check if transition is currently failing (recent failures)."""
        if self.last_failure is None:
            return False
        if self.last_success is None:
            return True
        return self.last_failure > self.last_success


class TransitionReliability:
    """Track and query transition reliability based on execution history.

    Maintains a history of transition attempts and provides reliability
    scores that can be used to inform pathfinding decisions.

    Features:
    - Recency weighting (recent attempts matter more)
    - Persistence across sessions
    - Thread-safe access
    - Configurable history limits

    Attributes:
        persistence_path: Path to JSON file for persistence.
        max_history_per_transition: Maximum attempts to store per transition.
        recency_decay: Decay factor for recency weighting (0.0-1.0).
    """

    def __init__(
        self,
        persistence_path: Path | str | None = None,
        max_history_per_transition: int = 100,
        recency_decay: float = 0.95,
    ) -> None:
        """Initialize reliability tracker.

        Args:
            persistence_path: Path to JSON file for persistence.
                            Defaults to ~/.qontinui/transition_reliability.json
            max_history_per_transition: Max attempts stored per transition.
            recency_decay: Weight decay per attempt (older = less weight).
                          0.95 means each older attempt is 95% as important.
        """
        if persistence_path is None:
            persistence_path = Path.home() / ".qontinui" / "transition_reliability.json"
        self.persistence_path = Path(persistence_path)
        self.max_history_per_transition = max_history_per_transition
        self.recency_decay = recency_decay

        # History storage: (from_state, to_state) -> list[TransitionAttempt]
        self._history: dict[tuple[str, str], list[TransitionAttempt]] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Load existing data
        self._load()

    def _transition_key(self, from_state: str, to_state: str) -> tuple[str, str]:
        """Create normalized key for transition."""
        return (str(from_state), str(to_state))

    def record_attempt(
        self,
        from_state: str,
        to_state: str,
        success: bool,
        duration_ms: float | None = None,
        failure_reason: str | None = None,
    ) -> None:
        """Record a transition attempt.

        Args:
            from_state: Source state identifier.
            to_state: Target state identifier.
            success: Whether the transition succeeded.
            duration_ms: Duration of attempt in milliseconds.
            failure_reason: Description if failed.
        """
        key = self._transition_key(from_state, to_state)
        attempt = TransitionAttempt(
            timestamp=time.time(),
            success=success,
            duration_ms=duration_ms,
            failure_reason=failure_reason,
        )

        with self._lock:
            if key not in self._history:
                self._history[key] = []

            self._history[key].append(attempt)

            # Trim old entries
            if len(self._history[key]) > self.max_history_per_transition:
                self._history[key] = self._history[key][-self.max_history_per_transition :]

        logger.debug(
            f"Recorded {'success' if success else 'failure'} for " f"{from_state} -> {to_state}"
        )

    def record_success(
        self,
        from_state: str,
        to_state: str,
        duration_ms: float | None = None,
    ) -> None:
        """Convenience method to record successful transition.

        Args:
            from_state: Source state identifier.
            to_state: Target state identifier.
            duration_ms: Duration of transition.
        """
        self.record_attempt(from_state, to_state, True, duration_ms)

    def record_failure(
        self,
        from_state: str,
        to_state: str,
        reason: str | None = None,
        duration_ms: float | None = None,
    ) -> None:
        """Convenience method to record failed transition.

        Args:
            from_state: Source state identifier.
            to_state: Target state identifier.
            reason: Description of failure.
            duration_ms: Duration before failure.
        """
        self.record_attempt(from_state, to_state, False, duration_ms, reason)

    def get_reliability(
        self,
        from_state: str,
        to_state: str,
        use_recency_weighting: bool = True,
    ) -> float:
        """Get reliability score for a transition.

        Args:
            from_state: Source state identifier.
            to_state: Target state identifier.
            use_recency_weighting: Apply recency weighting.

        Returns:
            Reliability score 0.0-1.0.
            - 1.0 = always succeeds
            - 0.5 = unknown or 50/50
            - 0.0 = always fails
        """
        key = self._transition_key(from_state, to_state)

        with self._lock:
            attempts = self._history.get(key, [])

        if not attempts:
            return 0.5  # Unknown, neutral

        if not use_recency_weighting:
            # Simple success rate
            successes = sum(1 for a in attempts if a.success)
            return successes / len(attempts)

        # Recency-weighted success rate
        # Most recent attempts have highest weight
        total_weight = 0.0
        weighted_successes = 0.0

        for i, attempt in enumerate(reversed(attempts)):
            weight = self.recency_decay**i
            total_weight += weight
            if attempt.success:
                weighted_successes += weight

        if total_weight == 0:
            return 0.5

        return weighted_successes / total_weight

    def get_cost_multiplier(
        self,
        from_state: str,
        to_state: str,
        min_multiplier: float = 1.0,
        max_multiplier: float = 10.0,
    ) -> float:
        """Get cost multiplier for pathfinding.

        Low reliability = high cost = path avoided.

        Args:
            from_state: Source state identifier.
            to_state: Target state identifier.
            min_multiplier: Multiplier for 100% reliable transitions.
            max_multiplier: Multiplier for 0% reliable transitions.

        Returns:
            Cost multiplier to apply to transition cost.
        """
        reliability = self.get_reliability(from_state, to_state)
        # Linear interpolation: reliability 1.0 -> min, 0.0 -> max
        return min_multiplier + (1.0 - reliability) * (max_multiplier - min_multiplier)

    def get_stats(self, from_state: str, to_state: str) -> TransitionStats | None:
        """Get aggregated statistics for a transition.

        Args:
            from_state: Source state identifier.
            to_state: Target state identifier.

        Returns:
            TransitionStats or None if no history.
        """
        key = self._transition_key(from_state, to_state)

        with self._lock:
            attempts = self._history.get(key, [])

        if not attempts:
            return None

        successes = [a for a in attempts if a.success]
        failures = [a for a in attempts if not a.success]

        # Calculate average duration of successes
        durations = [a.duration_ms for a in successes if a.duration_ms is not None]
        avg_duration = sum(durations) / len(durations) if durations else None

        return TransitionStats(
            from_state=from_state,
            to_state=to_state,
            total_attempts=len(attempts),
            successes=len(successes),
            failures=len(failures),
            avg_duration_ms=avg_duration,
            last_success=max((a.timestamp for a in successes), default=None),
            last_failure=max((a.timestamp for a in failures), default=None),
        )

    def get_all_stats(self) -> list[TransitionStats]:
        """Get statistics for all tracked transitions.

        Returns:
            List of TransitionStats for all transitions.
        """
        stats = []
        with self._lock:
            keys = list(self._history.keys())

        for from_state, to_state in keys:
            stat = self.get_stats(from_state, to_state)
            if stat:
                stats.append(stat)

        return stats

    def get_failing_transitions(self, min_failures: int = 2) -> list[TransitionStats]:
        """Get transitions that are currently failing.

        Args:
            min_failures: Minimum failures to be considered.

        Returns:
            List of failing transitions.
        """
        all_stats = self.get_all_stats()
        return [s for s in all_stats if s.is_failing and s.failures >= min_failures]

    def clear_history(self, from_state: str | None = None, to_state: str | None = None) -> int:
        """Clear transition history.

        Args:
            from_state: If provided, only clear transitions from this state.
            to_state: If provided, only clear transitions to this state.
                     If both None, clears all history.

        Returns:
            Number of transitions cleared.
        """
        with self._lock:
            if from_state is None and to_state is None:
                count = len(self._history)
                self._history.clear()
                return count

            to_remove = []
            for key in self._history.keys():
                if from_state is not None and key[0] != from_state:
                    continue
                if to_state is not None and key[1] != to_state:
                    continue
                to_remove.append(key)

            for key in to_remove:
                del self._history[key]

            return len(to_remove)

    def _load(self) -> None:
        """Load history from persistence file."""
        if not self.persistence_path.exists():
            return

        try:
            with open(self.persistence_path, encoding="utf-8") as f:
                data = json.load(f)

            with self._lock:
                for key_str, attempts_data in data.get("history", {}).items():
                    # Parse key (stored as "from_state|to_state")
                    parts = key_str.split("|", 1)
                    if len(parts) != 2:
                        continue
                    key = (parts[0], parts[1])

                    attempts = []
                    for a in attempts_data:
                        attempts.append(
                            TransitionAttempt(
                                timestamp=a["timestamp"],
                                success=a["success"],
                                duration_ms=a.get("duration_ms"),
                                failure_reason=a.get("failure_reason"),
                            )
                        )

                    self._history[key] = attempts[-self.max_history_per_transition :]

            logger.info(f"Loaded reliability data for {len(self._history)} transitions")

        except (json.JSONDecodeError, KeyError, TypeError, OSError) as e:
            logger.warning(f"Failed to load reliability data: {e}")

    def save(self) -> bool:
        """Save history to persistence file.

        Returns:
            True if saved successfully.
        """
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)

            with self._lock:
                history_data = {}
                for (from_state, to_state), attempts in self._history.items():
                    key_str = f"{from_state}|{to_state}"
                    history_data[key_str] = [
                        {
                            "timestamp": a.timestamp,
                            "success": a.success,
                            "duration_ms": a.duration_ms,
                            "failure_reason": a.failure_reason,
                        }
                        for a in attempts
                    ]

            data = {
                "version": 1,
                "history": history_data,
                "saved_at": time.time(),
            }

            # Atomic write
            temp_path = self.persistence_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(self.persistence_path)

            logger.debug(f"Saved reliability data for {len(self._history)} transitions")
            return True

        except OSError as e:
            logger.warning(f"Failed to save reliability data: {e}")
            return False

    def __str__(self) -> str:
        """String representation."""
        with self._lock:
            return f"TransitionReliability(transitions={len(self._history)})"


# Global instance
_default_reliability: TransitionReliability | None = None


def get_transition_reliability() -> TransitionReliability:
    """Get the global transition reliability tracker.

    Returns:
        The global TransitionReliability instance.
    """
    global _default_reliability
    if _default_reliability is None:
        _default_reliability = TransitionReliability()
    return _default_reliability


def set_transition_reliability(tracker: TransitionReliability) -> None:
    """Set the global transition reliability tracker.

    Args:
        tracker: TransitionReliability instance to use globally.
    """
    global _default_reliability
    _default_reliability = tracker
