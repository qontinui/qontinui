"""HybridPathFinder - Combines Qontinui's efficiency with Brobot's completeness.

This module implements a hybrid pathfinding approach that:
- Supports multiple start and target states (Brobot feature)
- Uses efficient algorithms (Qontinui advantage)
- Finds paths that activate ALL target states
- Supports multiple pathfinding strategies
"""

import heapq
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from qontinui.model.transition.enhanced_joint_table import StateTransitionsJointTable
from qontinui.model.transition.enhanced_state_transition import StateTransition

logger = logging.getLogger(__name__)


class PathStrategy(Enum):
    """Pathfinding strategy selection."""

    SHORTEST = "bfs"  # Qontinui's BFS for shortest path
    OPTIMAL = "astar"  # Qontinui's A* for optimal path
    ALL_PATHS = "recursive"  # Brobot's recursive all-paths
    MOST_RELIABLE = "success"  # Based on success rates


@dataclass
class Path:
    """Represents a path through states with transitions."""

    states: list[int] = field(default_factory=list)
    transitions: list[StateTransition] = field(default_factory=list)
    total_cost: float = 0.0
    total_probability: float = 1.0

    def add_state(self, state_id: int, transition: StateTransition | None = None) -> None:
        """Add a state and optional transition to the path."""
        self.states.append(state_id)
        if transition:
            self.transitions.append(transition)

    def copy(self) -> "Path":
        """Create a deep copy of the path."""
        new_path = Path()
        new_path.states = self.states.copy()
        new_path.transitions = self.transitions.copy()
        new_path.total_cost = self.total_cost
        new_path.total_probability = self.total_probability
        return new_path

    def reverse(self) -> None:
        """Reverse the path in place."""
        self.states.reverse()
        self.transitions.reverse()

    def reaches_all(self, target_states: set[int]) -> bool:
        """Check if path reaches all target states.

        This is crucial for multi-state activation - the path must
        enable reaching ALL required states.

        Args:
            target_states: States that must all be reachable

        Returns:
            True if all states are reachable via this path
        """
        # Check if final transition activates all targets
        if self.transitions:
            final_transition = self.transitions[-1]
            return final_transition.activates_all(target_states)

        # If no transitions, check if we're already at all targets
        return target_states.issubset(set(self.states))

    def __len__(self) -> int:
        """Get path length."""
        return len(self.states)


@dataclass
class HybridPathFinder:
    """Hybrid pathfinder combining Qontinui efficiency with Brobot completeness.

    Key features:
    - Multiple start states (current active states)
    - Multiple target states (all states to activate)
    - Configurable strategy selection
    - Path must enable reaching ALL target states
    """

    joint_table: StateTransitionsJointTable
    state_service: Optional["StateService"] = None  # For state metadata

    # Configuration
    max_depth: int = 20
    enable_caching: bool = True
    strategy: PathStrategy = PathStrategy.OPTIMAL

    # Scoring weights (configurable)
    state_cost_weight: float = 0.3
    transition_cost_weight: float = 0.3
    probability_weight: float = 0.2
    reliability_weight: float = 0.2

    # Cache
    _path_cache: dict[tuple[frozenset, frozenset], Path] = field(default_factory=dict)

    def find_path_to_states(
        self, start_states: set[int], target_states: set[int], strategy: PathStrategy | None = None
    ) -> Path | None:
        """Find path considering multiple active states and targets.

        This is the main entry point that ensures ALL target states
        can be activated from the current active states.

        Args:
            start_states: Current active state IDs
            target_states: ALL states that must be activated
            strategy: Override default strategy

        Returns:
            Path that reaches all targets or None
        """
        if not start_states or not target_states:
            logger.warning("Empty start or target states")
            return None

        # Check cache
        cache_key = (frozenset(start_states), frozenset(target_states))
        if self.enable_caching and cache_key in self._path_cache:
            logger.debug("Returning cached path")
            return self._path_cache[cache_key]

        # Select strategy
        strategy = strategy or self.strategy

        logger.info(
            f"Finding path from {len(start_states)} states to "
            f"{len(target_states)} targets using {strategy.value}"
        )

        # Execute appropriate strategy
        path = None
        if strategy == PathStrategy.SHORTEST:
            path = self._find_shortest_path(start_states, target_states)
        elif strategy == PathStrategy.OPTIMAL:
            path = self._find_optimal_path(start_states, target_states)
        elif strategy == PathStrategy.ALL_PATHS:
            paths = self._find_all_paths_recursive(start_states, target_states)
            if paths:
                path = self._select_best_path(paths)
        elif strategy == PathStrategy.MOST_RELIABLE:
            path = self._find_most_reliable_path(start_states, target_states)

        # Cache result
        if self.enable_caching and path:
            self._path_cache[cache_key] = path

        if path:
            logger.info(f"Found path with {len(path)} states, cost={path.total_cost:.2f}")
        else:
            logger.warning("No path found")

        return path

    def find_all_paths(
        self, start_states: set[int], target_states: set[int], max_paths: int = 10
    ) -> list[Path]:
        """Find all paths to activate target states (Brobot style).

        Args:
            start_states: Current active state IDs
            target_states: ALL states that must be activated
            max_paths: Maximum number of paths to return

        Returns:
            List of valid paths
        """
        return self._find_all_paths_recursive(start_states, target_states, max_paths)

    def _find_shortest_path(self, start_states: set[int], target_states: set[int]) -> Path | None:
        """Find shortest path using BFS (Qontinui's efficient approach).

        Args:
            start_states: Starting state IDs
            target_states: Target state IDs

        Returns:
            Shortest path or None
        """
        # BFS to find shortest path
        queue = deque()

        # Initialize with all start states
        for state_id in start_states:
            path = Path()
            path.add_state(state_id)
            queue.append((state_id, path))

        visited = set(start_states)

        while queue:
            current_state, current_path = queue.popleft()

            # Check depth limit
            if len(current_path) > self.max_depth:
                continue

            # Get transitions from current state
            transitions = self.joint_table.get_transitions_from(current_state)

            for transition in transitions:
                # Check if this transition activates ALL target states
                if transition.activates_all(target_states):
                    # Found a valid path
                    final_path = current_path.copy()
                    final_path.add_state(-1, transition)  # -1 placeholder for targets
                    self._calculate_path_score(final_path)
                    return final_path

                # Otherwise, explore states this transition activates
                for next_state in transition.activate:
                    if next_state not in visited:
                        visited.add(next_state)
                        new_path = current_path.copy()
                        new_path.add_state(next_state, transition)
                        queue.append((next_state, new_path))

        return None

    def _find_optimal_path(self, start_states: set[int], target_states: set[int]) -> Path | None:
        """Find optimal path using A* (Qontinui's approach with enhancements).

        Args:
            start_states: Starting state IDs
            target_states: Target state IDs

        Returns:
            Optimal path or None
        """
        # Priority queue: (negative_score, counter, state_id, path)
        heap = []
        counter = 0

        # Initialize with all start states
        for state_id in start_states:
            path = Path()
            path.add_state(state_id)
            heapq.heappush(heap, (0.0, counter, state_id, path))
            counter += 1

        visited = {}  # state_id -> best_cost

        while heap:
            neg_score, _, current_state, current_path = heapq.heappop(heap)
            current_cost = -neg_score

            # Check if we've seen this state with better cost
            if current_state in visited and visited[current_state] <= current_cost:
                continue

            visited[current_state] = current_cost

            # Check depth limit
            if len(current_path) > self.max_depth:
                continue

            # Get transitions from current state
            transitions = self.joint_table.get_transitions_from(current_state)

            for transition in transitions:
                # Check if this transition activates ALL target states
                if transition.activates_all(target_states):
                    # Found a valid path
                    final_path = current_path.copy()
                    final_path.add_state(-1, transition)
                    self._calculate_path_score(final_path)
                    return final_path

                # Explore states this transition activates
                for next_state in transition.activate:
                    new_path = current_path.copy()
                    new_path.add_state(next_state, transition)

                    # Calculate path cost
                    path_cost = self._calculate_path_cost(new_path)

                    # Add heuristic (estimated cost to target)
                    heuristic = self._estimate_cost_to_targets(next_state, target_states)
                    total_cost = path_cost + heuristic

                    counter += 1
                    heapq.heappush(heap, (-total_cost, counter, next_state, new_path))

        return None

    def _find_all_paths_recursive(
        self, start_states: set[int], target_states: set[int], max_paths: int = 10
    ) -> list[Path]:
        """Find all paths using recursive DFS (Brobot's approach).

        Args:
            start_states: Starting state IDs
            target_states: Target state IDs
            max_paths: Maximum paths to find

        Returns:
            List of all valid paths
        """
        all_paths = []

        def recurse(current_state: int, current_path: Path, visited: set[int]):
            if len(all_paths) >= max_paths:
                return

            if len(current_path) > self.max_depth:
                return

            # Get transitions from current state
            transitions = self.joint_table.get_transitions_from(current_state)

            for transition in transitions:
                # Check if this transition activates ALL target states
                if transition.activates_all(target_states):
                    # Found a valid path
                    final_path = current_path.copy()
                    final_path.add_state(-1, transition)
                    self._calculate_path_score(final_path)
                    all_paths.append(final_path)

                    if len(all_paths) >= max_paths:
                        return

                # Explore further
                for next_state in transition.activate:
                    if next_state not in visited:
                        new_visited = visited.copy()
                        new_visited.add(next_state)
                        new_path = current_path.copy()
                        new_path.add_state(next_state, transition)
                        recurse(next_state, new_path, new_visited)

        # Start recursion from all start states
        for start_state in start_states:
            initial_path = Path()
            initial_path.add_state(start_state)
            recurse(start_state, initial_path, {start_state})

        return all_paths

    def _find_most_reliable_path(
        self, start_states: set[int], target_states: set[int]
    ) -> Path | None:
        """Find path with highest success rate.

        Args:
            start_states: Starting state IDs
            target_states: Target state IDs

        Returns:
            Most reliable path or None
        """
        # Temporarily adjust weights to prioritize reliability
        old_reliability = self.reliability_weight
        self.reliability_weight = 0.7
        self.state_cost_weight = 0.1
        self.transition_cost_weight = 0.1
        self.probability_weight = 0.1

        path = self._find_optimal_path(start_states, target_states)

        # Restore weights
        self.reliability_weight = old_reliability
        self.state_cost_weight = 0.3
        self.transition_cost_weight = 0.3
        self.probability_weight = 0.2

        return path

    def _calculate_path_score(self, path: Path) -> None:
        """Calculate comprehensive path score.

        Combines:
        - State costs (from Brobot)
        - Transition costs (from Brobot)
        - Probability weights (from Qontinui)
        - Success rates (hybrid)

        Args:
            path: Path to score
        """
        state_cost = 0.0
        transition_cost = 0.0
        total_probability = 1.0
        total_reliability = 1.0

        # Sum state costs
        if self.state_service:
            for state_id in path.states:
                if state_id != -1:  # Skip placeholder
                    state = self.state_service.get_state(state_id)
                    if state:
                        state_cost += state.path_cost

        # Sum transition costs and calculate probabilities
        for transition in path.transitions:
            transition_cost += transition.path_cost
            total_reliability *= transition.get_success_rate()

        # Calculate weighted score
        path.total_cost = (
            state_cost * self.state_cost_weight
            + transition_cost * self.transition_cost_weight
            + (1.0 - total_probability) * self.probability_weight * 10
            + (1.0 - total_reliability) * self.reliability_weight * 10
        )
        path.total_probability = total_probability

    def _calculate_path_cost(self, path: Path) -> float:
        """Calculate simple path cost for A*.

        Args:
            path: Path to evaluate

        Returns:
            Path cost
        """
        cost = float(len(path.states))  # Base cost is path length

        # Add transition costs
        for transition in path.transitions:
            cost += transition.get_total_cost()

        return cost

    def _estimate_cost_to_targets(self, from_state: int, target_states: set[int]) -> float:
        """Heuristic estimate of cost to reach all targets.

        Args:
            from_state: Current state
            target_states: Target states

        Returns:
            Estimated cost (admissible heuristic)
        """
        # Simple heuristic: assume at least one more transition needed
        # This is admissible (never overestimates)
        return 1.0

    def _select_best_path(self, paths: list[Path]) -> Path | None:
        """Select best path from a list.

        Args:
            paths: List of paths to choose from

        Returns:
            Best path or None
        """
        if not paths:
            return None

        # Sort by total cost (lower is better)
        paths.sort(key=lambda p: p.total_cost)
        return paths[0]

    def clear_cache(self) -> None:
        """Clear the path cache."""
        self._path_cache.clear()

    def __str__(self) -> str:
        """String representation."""
        return (
            f"HybridPathFinder(strategy={self.strategy.value}, "
            f"max_depth={self.max_depth}, "
            f"cache_size={len(self._path_cache)})"
        )


# Placeholder for StateService
class StateService:
    """Placeholder for state service."""

    def get_state(self, state_id: int):
        """Get state by ID."""
        return None
