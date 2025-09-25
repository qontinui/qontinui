"""State transition aspect - ported from Qontinui framework.

Tracks and analyzes state transitions for navigation insights.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TransitionStats:
    """Statistics for a specific state transition."""

    from_state: str
    """Source state name."""

    to_state: str
    """Target state name."""

    total_attempts: int = 0
    """Total transition attempts."""

    successful_transitions: int = 0
    """Number of successful transitions."""

    failed_transitions: int = 0
    """Number of failed transitions."""

    total_time_ms: float = 0.0
    """Total time spent in transitions."""

    min_time_ms: float = float("inf")
    """Minimum transition time."""

    max_time_ms: float = 0.0
    """Maximum transition time."""

    last_transition_time: datetime | None = None
    """Timestamp of last transition."""

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_attempts == 0:
            return 0.0
        return (self.successful_transitions / self.total_attempts) * 100

    @property
    def average_time_ms(self) -> float:
        """Calculate average transition time."""
        if self.successful_transitions == 0:
            return 0.0
        return self.total_time_ms / self.successful_transitions


@dataclass
class StateNode:
    """Node in the state graph."""

    name: str
    """State name."""

    visit_count: int = 0
    """Number of times this state was visited."""

    total_time_in_state_ms: float = 0.0
    """Total time spent in this state."""

    entry_time: float | None = None
    """Time when state was entered."""

    outgoing_transitions: set[str] = field(default_factory=set)
    """Set of states this state can transition to."""

    incoming_transitions: set[str] = field(default_factory=set)
    """Set of states that can transition to this state."""

    is_initial: bool = False
    """Whether this is an initial state."""

    is_terminal: bool = False
    """Whether this is a terminal state."""

    @property
    def average_time_in_state_ms(self) -> float:
        """Calculate average time spent in state."""
        if self.visit_count == 0:
            return 0.0
        return self.total_time_in_state_ms / self.visit_count


class StateTransitionAspect:
    """Tracks and analyzes state transitions.

    Port of StateTransitionAspect from Qontinui framework.

    Features:
    - Real-time state transition graph building
    - Success/failure rate tracking
    - Transition timing analysis
    - State machine visualization generation (DOT format)
    - Unreachable state detection
    - Navigation pattern analytics
    """

    def __init__(
        self,
        enabled: bool = True,
        track_success_rates: bool = True,
        generate_visualizations: bool = True,
    ):
        """Initialize the aspect.

        Args:
            enabled: Whether tracking is enabled
            track_success_rates: Track transition success rates
            generate_visualizations: Generate state graph visualizations
        """
        self.enabled = enabled
        self.track_success_rates = track_success_rates
        self.generate_visualizations = generate_visualizations

        # State graph
        self._state_graph: dict[str, StateNode] = {}

        # Transition statistics
        self._transition_stats: dict[tuple[str, str], TransitionStats] = {}

        # Current state tracking
        self._current_state: str | None = None

        # Navigation patterns
        self._navigation_paths: list[list[str]] = []
        self._current_path: list[str] = []

        # Transition history
        self._transition_history: list[dict[str, Any]] = []

    def track_transition(self, func):
        """Decorator to track state transitions.

        Args:
            func: Transition function to wrap

        Returns:
            Wrapped function
        """

        @wraps(func)
        def wrapper(transition_instance, *args, **kwargs):
            if not self.enabled:
                return func(transition_instance, *args, **kwargs)

            # Extract transition info from instance metadata
            from_state = self._get_from_state(transition_instance)
            to_state = self._get_to_state(transition_instance)

            if not from_state or not to_state:
                return func(transition_instance, *args, **kwargs)

            # Start timing
            start_time = time.time()

            # Record transition attempt
            self._record_transition_attempt(from_state, to_state)

            success = False
            try:
                # Execute transition
                result = func(transition_instance, *args, **kwargs)

                # Check if transition succeeded
                success = self._is_successful_result(result)

                if success:
                    # Update current state
                    self._enter_state(to_state)
                    self._leave_state(from_state)

                return result

            finally:
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000

                # Record transition result
                self._record_transition_result(from_state, to_state, success, duration_ms)

        return wrapper

    def _get_from_state(self, transition_instance) -> str | None:
        """Extract source state from transition instance.

        Args:
            transition_instance: Transition object

        Returns:
            Source state name or None
        """
        # Check for annotation metadata
        if hasattr(transition_instance, "_qontinui_transition_from"):
            states = transition_instance._qontinui_transition_from
            if states:
                # Return first state for simplicity
                return states[0].__name__

        return None

    def _get_to_state(self, transition_instance) -> str | None:
        """Extract target state from transition instance.

        Args:
            transition_instance: Transition object

        Returns:
            Target state name or None
        """
        # Check for annotation metadata
        if hasattr(transition_instance, "_qontinui_transition_to"):
            states = transition_instance._qontinui_transition_to
            if states:
                # Return first state for simplicity
                return states[0].__name__

        return None

    def _is_successful_result(self, result: Any) -> bool:
        """Check if transition result indicates success.

        Args:
            result: Transition result

        Returns:
            True if successful
        """
        if isinstance(result, bool):
            return result
        # Could check for StateTransition object
        return result is not None

    def _record_transition_attempt(self, from_state: str, to_state: str) -> None:
        """Record a transition attempt.

        Args:
            from_state: Source state
            to_state: Target state
        """
        # Ensure states exist in graph
        if from_state not in self._state_graph:
            self._state_graph[from_state] = StateNode(from_state)
        if to_state not in self._state_graph:
            self._state_graph[to_state] = StateNode(to_state)

        # Update graph connections
        self._state_graph[from_state].outgoing_transitions.add(to_state)
        self._state_graph[to_state].incoming_transitions.add(from_state)

        # Get or create transition stats
        key = (from_state, to_state)
        if key not in self._transition_stats:
            self._transition_stats[key] = TransitionStats(from_state, to_state)

        self._transition_stats[key].total_attempts += 1

    def _record_transition_result(
        self, from_state: str, to_state: str, success: bool, duration_ms: float
    ) -> None:
        """Record transition result.

        Args:
            from_state: Source state
            to_state: Target state
            success: Whether transition succeeded
            duration_ms: Transition duration
        """
        key = (from_state, to_state)
        stats = self._transition_stats[key]

        if success:
            stats.successful_transitions += 1
            stats.total_time_ms += duration_ms
            stats.min_time_ms = min(stats.min_time_ms, duration_ms)
            stats.max_time_ms = max(stats.max_time_ms, duration_ms)

            # Update navigation path
            self._current_path.append(to_state)
        else:
            stats.failed_transitions += 1

        stats.last_transition_time = datetime.now()

        # Add to history
        self._transition_history.append(
            {
                "from": from_state,
                "to": to_state,
                "success": success,
                "duration_ms": duration_ms,
                "timestamp": datetime.now(),
            }
        )

        # Limit history size
        if len(self._transition_history) > 1000:
            self._transition_history.pop(0)

    def _enter_state(self, state_name: str) -> None:
        """Record entering a state.

        Args:
            state_name: State being entered
        """
        if state_name in self._state_graph:
            node = self._state_graph[state_name]
            node.visit_count += 1
            node.entry_time = time.time()

        self._current_state = state_name

    def _leave_state(self, state_name: str) -> None:
        """Record leaving a state.

        Args:
            state_name: State being left
        """
        if state_name in self._state_graph:
            node = self._state_graph[state_name]
            if node.entry_time is not None:
                duration_ms = (time.time() - node.entry_time) * 1000
                node.total_time_in_state_ms += duration_ms
                node.entry_time = None

    def get_state_graph(self) -> dict[str, StateNode]:
        """Get the state graph.

        Returns:
            Dictionary of state nodes
        """
        return dict(self._state_graph)

    def get_transition_stats(self) -> dict[tuple[str, str], TransitionStats]:
        """Get transition statistics.

        Returns:
            Dictionary of transition stats
        """
        return dict(self._transition_stats)

    def get_unreachable_states(self, initial_states: set[str]) -> set[str]:
        """Find states that are unreachable from initial states.

        Args:
            initial_states: Set of initial state names

        Returns:
            Set of unreachable state names
        """
        # Perform BFS from initial states
        visited = set()
        queue = list(initial_states)

        while queue:
            state = queue.pop(0)
            if state in visited:
                continue

            visited.add(state)

            if state in self._state_graph:
                for next_state in self._state_graph[state].outgoing_transitions:
                    if next_state not in visited:
                        queue.append(next_state)

        # Find unreachable states
        all_states = set(self._state_graph.keys())
        unreachable = all_states - visited

        return unreachable

    def generate_dot_graph(self) -> str:
        """Generate DOT format graph for visualization.

        Returns:
            DOT format string
        """
        lines = ["digraph StateTransitions {"]
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=ellipse];")

        # Add nodes with visit counts
        for state_name, node in self._state_graph.items():
            label = f"{state_name}\\nvisits: {node.visit_count}"
            color = "green" if node.is_initial else "red" if node.is_terminal else "black"
            lines.append(f'  "{state_name}" [label="{label}", color={color}];')

        # Add edges with success rates
        for (from_state, to_state), stats in self._transition_stats.items():
            if stats.total_attempts > 0:
                label = f"{stats.success_rate:.1f}%\\n{stats.average_time_ms:.0f}ms"
                color = (
                    "green"
                    if stats.success_rate > 80
                    else "red" if stats.success_rate < 50 else "orange"
                )
                lines.append(
                    f'  "{from_state}" -> "{to_state}" ' f'[label="{label}", color={color}];'
                )

        lines.append("}")

        return "\\n".join(lines)

    def get_navigation_patterns(self, min_length: int = 3) -> dict[tuple, int]:
        """Find common navigation patterns.

        Args:
            min_length: Minimum pattern length

        Returns:
            Dictionary of patterns to occurrence counts
        """
        patterns = defaultdict(int)

        for path in self._navigation_paths:
            if len(path) < min_length:
                continue

            # Extract all subsequences of min_length
            for i in range(len(path) - min_length + 1):
                pattern = tuple(path[i : i + min_length])
                patterns[pattern] += 1

        return dict(patterns)

    def reset_tracking(self) -> None:
        """Reset all tracking data."""
        self._state_graph.clear()
        self._transition_stats.clear()
        self._current_state = None
        self._navigation_paths.clear()
        self._current_path.clear()
        self._transition_history.clear()
        logger.info("State transition tracking reset")


# Global instance
_state_transition_aspect = StateTransitionAspect()


def track_state_transition(func):
    """Decorator for tracking state transitions.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function
    """
    return _state_transition_aspect.track_transition(func)


def get_state_transition_aspect() -> StateTransitionAspect:
    """Get the global state transition aspect.

    Returns:
        The state transition aspect
    """
    return _state_transition_aspect
