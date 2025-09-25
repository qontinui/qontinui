"""PathFinder - ported from Qontinui framework.

Finds paths between states using various algorithms.
"""

import heapq
from collections import deque
from dataclasses import dataclass, field

from .path import Path
from .state import State


@dataclass
class PathFinder:
    """Finds paths between states.

    Port of PathFinder from Qontinui framework class.
    Implements various path finding algorithms.
    """

    states: dict[str, State] = field(default_factory=dict)

    # Path finding configuration
    _max_depth: int = 20  # Maximum path depth
    _use_probability: bool = True  # Consider transition probabilities
    _use_score: bool = True  # Consider transition scores
    _allow_loops: bool = False  # Allow loops in paths

    # Cache
    _path_cache: dict[tuple[str, str], Path] = field(default_factory=dict)
    _cache_enabled: bool = True

    def add_state(self, state: State) -> "PathFinder":
        """Add a state to the path finder (fluent).

        Args:
            state: State to add

        Returns:
            Self for chaining
        """
        self.states[state.name] = state
        self._clear_cache()
        return self

    def find_path(self, start: State, end: State) -> Path | None:
        """Find path between two states.

        Args:
            start: Starting state
            end: Target state

        Returns:
            Path or None if no path exists
        """
        # Check cache
        cache_key = (start.name, end.name)
        if self._cache_enabled and cache_key in self._path_cache:
            return self._path_cache[cache_key]

        # Find path using configured algorithm
        if self._use_score and self._use_probability:
            path = self._find_best_path(start, end)
        elif self._use_score:
            path = self._find_highest_score_path(start, end)
        elif self._use_probability:
            path = self._find_most_probable_path(start, end)
        else:
            path = self._find_shortest_path(start, end)

        # Cache result
        if self._cache_enabled and path:
            self._path_cache[cache_key] = path

        return path

    def find_shortest_path(self, start: State, end: State) -> Path | None:
        """Find shortest path using BFS.

        Args:
            start: Starting state
            end: Target state

        Returns:
            Shortest path or None
        """
        return self._find_shortest_path(start, end)

    def find_all_paths(self, start: State, end: State, max_paths: int = 10) -> list[Path]:
        """Find all paths between states.

        Args:
            start: Starting state
            end: Target state
            max_paths: Maximum number of paths to return

        Returns:
            List of paths
        """
        paths = []
        visited = set()
        current_path = Path()

        self._find_all_paths_recursive(start, end, visited, current_path, paths, max_paths)

        return paths[:max_paths]

    def _find_shortest_path(self, start: State, end: State) -> Path | None:
        """Find shortest path using BFS.

        Args:
            start: Starting state
            end: Target state

        Returns:
            Shortest path or None
        """
        if start == end:
            path = Path()
            path.add_state(start)
            return path

        queue = deque([(start, Path().add_state(start))])
        visited = {start.name}

        while queue:
            current_state, current_path = queue.popleft()

            if len(current_path.states) > self._max_depth:
                continue

            for transition in current_state.transitions:
                next_state = transition.to_state
                if not next_state:
                    continue

                if next_state == end:
                    # Found path
                    final_path = Path()
                    for state in current_path.states:
                        final_path.add_state(state)
                    final_path.add_state(next_state, transition)
                    return final_path

                if next_state.name not in visited or self._allow_loops:
                    visited.add(next_state.name)
                    new_path = Path()
                    for i, state in enumerate(current_path.states):
                        trans = (
                            current_path.transitions[i]
                            if i < len(current_path.transitions)
                            else None
                        )
                        new_path.add_state(state, trans)
                    new_path.add_state(next_state, transition)
                    queue.append((next_state, new_path))

        return None

    def _find_best_path(self, start: State, end: State) -> Path | None:
        """Find best path using A* with score and probability.

        Args:
            start: Starting state
            end: Target state

        Returns:
            Best path or None
        """
        if start == end:
            path = Path()
            path.add_state(start)
            return path

        # Priority queue: (negative_score, path)
        heap = [(-1000.0, 0, Path().add_state(start))]
        visited = {}
        counter = 0

        while heap:
            neg_score, _, current_path = heapq.heappop(heap)
            current_state = current_path.get_last_state()

            if current_state == end:
                return current_path

            state_key = current_state.name
            if state_key in visited and visited[state_key] <= -neg_score:
                continue

            visited[state_key] = -neg_score

            if len(current_path.states) >= self._max_depth:
                continue

            for transition in current_state.transitions:
                next_state = transition.to_state
                if not next_state:
                    continue

                # Calculate path score
                new_path = Path()
                for i, state in enumerate(current_path.states):
                    trans = (
                        current_path.transitions[i] if i < len(current_path.transitions) else None
                    )
                    new_path.add_state(state, trans)
                new_path.add_state(next_state, transition)

                # Combined score (higher is better)
                combined_score = new_path.get_score() * new_path.get_probability()

                counter += 1
                heapq.heappush(heap, (-combined_score, counter, new_path))

        return None

    def _find_highest_score_path(self, start: State, end: State) -> Path | None:
        """Find path with highest total score.

        Args:
            start: Starting state
            end: Target state

        Returns:
            Highest score path or None
        """
        self._use_probability = False
        path = self._find_best_path(start, end)
        self._use_probability = True
        return path

    def _find_most_probable_path(self, start: State, end: State) -> Path | None:
        """Find path with highest probability.

        Args:
            start: Starting state
            end: Target state

        Returns:
            Most probable path or None
        """
        self._use_score = False
        path = self._find_best_path(start, end)
        self._use_score = True
        return path

    def _find_all_paths_recursive(
        self,
        current: State,
        end: State,
        visited: set[str],
        current_path: Path,
        all_paths: list[Path],
        max_paths: int,
    ):
        """Recursively find all paths (DFS).

        Args:
            current: Current state
            end: Target state
            visited: Visited states
            current_path: Current path being built
            all_paths: List to store found paths
            max_paths: Maximum paths to find
        """
        if len(all_paths) >= max_paths:
            return

        if len(current_path.states) > self._max_depth:
            return

        visited.add(current.name)
        current_path.add_state(current)

        if current == end:
            # Found a path
            path_copy = Path()
            path_copy.states = current_path.states.copy()
            path_copy.transitions = current_path.transitions.copy()
            all_paths.append(path_copy)
        else:
            # Explore neighbors
            for transition in current.transitions:
                next_state = transition.to_state
                if next_state and (next_state.name not in visited or self._allow_loops):
                    self._find_all_paths_recursive(
                        next_state, end, visited.copy(), current_path, all_paths, max_paths
                    )

        # Backtrack
        current_path.states.pop()
        if current_path.transitions:
            current_path.transitions.pop()

    def has_path(self, start: State, end: State) -> bool:
        """Check if path exists between states.

        Args:
            start: Starting state
            end: Target state

        Returns:
            True if path exists
        """
        return self.find_path(start, end) is not None

    def get_distance(self, start: State, end: State) -> int:
        """Get minimum distance between states.

        Args:
            start: Starting state
            end: Target state

        Returns:
            Minimum number of transitions or -1 if no path
        """
        path = self.find_shortest_path(start, end)
        if path:
            return len(path.states) - 1
        return -1

    def find_cycles(self) -> list[Path]:
        """Find all cycles in the state graph.

        Returns:
            List of cyclic paths
        """
        cycles = []

        for state in self.states.values():
            # Try to find path back to itself
            for transition in state.transitions:
                if transition.to_state:
                    path = self.find_path(transition.to_state, state)
                    if path and len(path.states) > 1:
                        # Add the completing transition
                        cycle = Path()
                        for i, s in enumerate(path.states):
                            trans = path.transitions[i] if i < len(path.transitions) else None
                            cycle.add_state(s, trans)
                        cycle.add_state(state, transition)
                        cycles.append(cycle)

        return cycles

    def set_max_depth(self, depth: int) -> "PathFinder":
        """Set maximum path depth (fluent).

        Args:
            depth: Maximum depth

        Returns:
            Self for chaining
        """
        self._max_depth = depth
        self._clear_cache()
        return self

    def set_use_probability(self, use: bool = True) -> "PathFinder":
        """Enable/disable probability consideration (fluent).

        Args:
            use: True to use probability

        Returns:
            Self for chaining
        """
        self._use_probability = use
        self._clear_cache()
        return self

    def set_use_score(self, use: bool = True) -> "PathFinder":
        """Enable/disable score consideration (fluent).

        Args:
            use: True to use score

        Returns:
            Self for chaining
        """
        self._use_score = use
        self._clear_cache()
        return self

    def set_allow_loops(self, allow: bool = True) -> "PathFinder":
        """Enable/disable loops in paths (fluent).

        Args:
            allow: True to allow loops

        Returns:
            Self for chaining
        """
        self._allow_loops = allow
        self._clear_cache()
        return self

    def _clear_cache(self):
        """Clear path cache."""
        self._path_cache.clear()

    def __str__(self) -> str:
        """String representation."""
        return f"PathFinder(states={len(self.states)}, cache={len(self._path_cache)})"
