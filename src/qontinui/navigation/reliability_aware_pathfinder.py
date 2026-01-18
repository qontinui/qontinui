"""Reliability-aware pathfinding integration.

Provides a mixin and wrapper to integrate TransitionReliability
with existing pathfinders.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .path_reliability import TransitionReliability, get_transition_reliability

if TYPE_CHECKING:
    from ..model.state import State
    from ..model.state.path import Path
    from ..model.state.path_finder import PathFinder

logger = logging.getLogger(__name__)


@dataclass
class ReliabilityAwarePathFinder:
    """Wrapper that adds runtime reliability tracking to PathFinder.

    Uses TransitionReliability to track actual success/failure rates
    and adjusts path costs accordingly. More reliable paths are preferred.

    Attributes:
        path_finder: Underlying PathFinder to wrap.
        reliability: TransitionReliability tracker.
        reliability_weight: How much reliability affects path cost (0.0-1.0).
        min_cost_multiplier: Minimum cost multiplier for reliable transitions.
        max_cost_multiplier: Maximum cost multiplier for unreliable transitions.
    """

    path_finder: "PathFinder"
    reliability: TransitionReliability = field(default_factory=get_transition_reliability)

    # Configuration
    reliability_weight: float = 0.5
    min_cost_multiplier: float = 1.0
    max_cost_multiplier: float = 5.0

    def find_path(self, start: "State", end: "State") -> "Path | None":
        """Find path considering reliability.

        Finds the path using the underlying PathFinder, then records
        which path was chosen for future reliability updates.

        Args:
            start: Starting state.
            end: Target state.

        Returns:
            Path or None if no path exists.
        """
        path = self.path_finder.find_path(start, end)

        if path:
            # Log path selection for debugging
            reliability_info = self._get_path_reliability_info(path)
            logger.debug(
                f"Selected path from {start.name} to {end.name}: "
                f"reliability={reliability_info['avg_reliability']:.2f}"
            )

        return path

    def find_path_with_reliability(
        self, start: "State", end: "State"
    ) -> tuple["Path | None", dict]:
        """Find path and return reliability information.

        Args:
            start: Starting state.
            end: Target state.

        Returns:
            Tuple of (path, reliability_info).
        """
        path = self.path_finder.find_path(start, end)

        if path:
            reliability_info = self._get_path_reliability_info(path)
        else:
            reliability_info = {"avg_reliability": 0.0, "transitions": []}

        return path, reliability_info

    def find_all_paths_ranked(
        self, start: "State", end: "State", max_paths: int = 10
    ) -> list[tuple["Path", float]]:
        """Find all paths ranked by reliability.

        Args:
            start: Starting state.
            end: Target state.
            max_paths: Maximum paths to return.

        Returns:
            List of (path, reliability_score) tuples, sorted by reliability.
        """
        all_paths = self.path_finder.find_all_paths(start, end, max_paths)

        # Calculate reliability for each path
        ranked = []
        for path in all_paths:
            info = self._get_path_reliability_info(path)
            ranked.append((path, info["avg_reliability"]))

        # Sort by reliability (highest first)
        ranked.sort(key=lambda x: x[1], reverse=True)

        return ranked

    def record_transition_result(
        self,
        from_state: str,
        to_state: str,
        success: bool,
        duration_ms: float | None = None,
        failure_reason: str | None = None,
    ) -> None:
        """Record the result of a transition attempt.

        Call this after executing a transition to update reliability data.

        Args:
            from_state: Source state name/ID.
            to_state: Target state name/ID.
            success: Whether the transition succeeded.
            duration_ms: How long the transition took.
            failure_reason: Reason for failure if applicable.
        """
        self.reliability.record_attempt(
            from_state=from_state,
            to_state=to_state,
            success=success,
            duration_ms=duration_ms,
            failure_reason=failure_reason,
        )

        if not success:
            # Clear path cache since reliability changed
            self.path_finder._clear_cache()
            logger.info(
                f"Recorded transition failure {from_state} -> {to_state}: "
                f"{failure_reason or 'unknown'}"
            )

    def get_transition_reliability(self, from_state: str, to_state: str) -> float:
        """Get reliability score for a specific transition.

        Args:
            from_state: Source state name/ID.
            to_state: Target state name/ID.

        Returns:
            Reliability score 0.0-1.0.
        """
        return self.reliability.get_reliability(from_state, to_state)

    def get_failing_transitions(self) -> list[tuple[str, str, float]]:
        """Get list of transitions that are currently failing.

        Returns:
            List of (from_state, to_state, success_rate) tuples.
        """
        stats = self.reliability.get_failing_transitions()
        return [(s.from_state, s.to_state, s.success_rate) for s in stats]

    def _get_path_reliability_info(self, path: "Path") -> dict:
        """Calculate reliability information for a path.

        Args:
            path: Path to analyze.

        Returns:
            Dictionary with reliability information.
        """
        if not path.transitions:
            return {"avg_reliability": 1.0, "transitions": []}

        transition_info = []
        total_reliability = 0.0

        for i, _transition in enumerate(path.transitions):
            if i < len(path.states) - 1:
                from_state = path.states[i].name
                to_state = path.states[i + 1].name

                reliability = self.reliability.get_reliability(from_state, to_state)
                transition_info.append(
                    {
                        "from": from_state,
                        "to": to_state,
                        "reliability": reliability,
                    }
                )
                total_reliability += reliability

        avg_reliability = total_reliability / len(path.transitions) if path.transitions else 1.0

        return {
            "avg_reliability": avg_reliability,
            "transitions": transition_info,
        }

    def save_reliability_data(self) -> bool:
        """Save reliability data to disk.

        Returns:
            True if saved successfully.
        """
        return self.reliability.save()


def create_reliability_aware_pathfinder(
    path_finder: "PathFinder",
    reliability: TransitionReliability | None = None,
    reliability_weight: float = 0.5,
) -> ReliabilityAwarePathFinder:
    """Create a reliability-aware pathfinder wrapper.

    Convenience function to wrap an existing PathFinder with
    reliability tracking.

    Args:
        path_finder: PathFinder to wrap.
        reliability: Optional TransitionReliability tracker.
        reliability_weight: How much reliability affects path cost.

    Returns:
        ReliabilityAwarePathFinder instance.
    """
    return ReliabilityAwarePathFinder(
        path_finder=path_finder,
        reliability=reliability or get_transition_reliability(),
        reliability_weight=reliability_weight,
    )
