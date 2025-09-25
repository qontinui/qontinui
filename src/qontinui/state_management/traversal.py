"""State traversal and navigation strategies."""

import logging
import random
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .models import StateGraph, Transition, TransitionType

logger = logging.getLogger(__name__)


class TraversalStrategy(Enum):
    """Types of traversal strategies."""

    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    RANDOM = "random"
    SHORTEST_PATH = "shortest_path"
    EXPLORATORY = "exploratory"
    GOAL_ORIENTED = "goal_oriented"


@dataclass
class TraversalResult:
    """Result of a traversal operation."""

    path: list[str]
    transitions: list[Transition]
    cost: float
    visited_states: set[str]
    success: bool
    metadata: dict[str, Any]


class StateTraversal:
    """Handles state graph traversal and navigation."""

    def __init__(self, state_graph: StateGraph):
        """Initialize StateTraversal.

        Args:
            state_graph: The state graph to traverse
        """
        self.state_graph = state_graph
        self.visited_states: set[str] = set()
        self.traversal_history: list[tuple[str, Transition]] = []
        self.cost_function: Callable | None = None

    def find_path(
        self, start: str, goal: str, strategy: TraversalStrategy = TraversalStrategy.SHORTEST_PATH
    ) -> TraversalResult | None:
        """Find a path from start to goal state.

        Args:
            start: Starting state name
            goal: Goal state name
            strategy: Traversal strategy to use

        Returns:
            TraversalResult if path found, None otherwise
        """
        if start not in self.state_graph.states or goal not in self.state_graph.states:
            logger.error(f"Invalid states: start={start}, goal={goal}")
            return None

        if start == goal:
            return TraversalResult(
                path=[start],
                transitions=[],
                cost=0.0,
                visited_states={start},
                success=True,
                metadata={"strategy": strategy.value},
            )

        if strategy == TraversalStrategy.BREADTH_FIRST:
            return self._bfs_search(start, goal)
        elif strategy == TraversalStrategy.DEPTH_FIRST:
            return self._dfs_search(start, goal)
        elif strategy == TraversalStrategy.SHORTEST_PATH:
            return self._dijkstra_search(start, goal)
        elif strategy == TraversalStrategy.RANDOM:
            return self._random_search(start, goal)
        elif strategy == TraversalStrategy.EXPLORATORY:
            return self._exploratory_search(start, goal)
        elif strategy == TraversalStrategy.GOAL_ORIENTED:
            return self._a_star_search(start, goal)
        else:
            return self._bfs_search(start, goal)

    def _bfs_search(self, start: str, goal: str) -> TraversalResult | None:
        """Breadth-first search for finding path.

        Args:
            start: Starting state
            goal: Goal state

        Returns:
            TraversalResult if path found
        """
        queue = deque([(start, [], [])])
        visited = {start}

        while queue:
            current, path, transitions = queue.popleft()

            state = self.state_graph.get_state(current)
            if not state:
                continue

            for transition in state.transitions:
                next_state = transition.to_state

                if next_state == goal:
                    final_path = path + [current, goal]
                    final_transitions = transitions + [transition]

                    return TraversalResult(
                        path=final_path,
                        transitions=final_transitions,
                        cost=len(final_transitions),
                        visited_states=visited | {goal},
                        success=True,
                        metadata={"strategy": "breadth_first", "nodes_explored": len(visited)},
                    )

                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, path + [current], transitions + [transition]))

        return TraversalResult(
            path=[],
            transitions=[],
            cost=float("inf"),
            visited_states=visited,
            success=False,
            metadata={"strategy": "breadth_first", "nodes_explored": len(visited)},
        )

    def _dfs_search(self, start: str, goal: str, max_depth: int = 100) -> TraversalResult | None:
        """Depth-first search for finding path.

        Args:
            start: Starting state
            goal: Goal state
            max_depth: Maximum search depth

        Returns:
            TraversalResult if path found
        """
        stack = [(start, [], [], 0)]
        visited = set()

        while stack:
            current, path, transitions, depth = stack.pop()

            if current in visited or depth > max_depth:
                continue

            visited.add(current)

            if current == goal:
                return TraversalResult(
                    path=path + [goal],
                    transitions=transitions,
                    cost=len(transitions),
                    visited_states=visited,
                    success=True,
                    metadata={"strategy": "depth_first", "nodes_explored": len(visited)},
                )

            state = self.state_graph.get_state(current)
            if not state:
                continue

            for transition in reversed(state.transitions):
                next_state = transition.to_state
                if next_state not in visited:
                    stack.append(
                        (next_state, path + [current], transitions + [transition], depth + 1)
                    )

        return TraversalResult(
            path=[],
            transitions=[],
            cost=float("inf"),
            visited_states=visited,
            success=False,
            metadata={"strategy": "depth_first", "nodes_explored": len(visited)},
        )

    def _dijkstra_search(self, start: str, goal: str) -> TraversalResult | None:
        """Dijkstra's algorithm for finding shortest path.

        Args:
            start: Starting state
            goal: Goal state

        Returns:
            TraversalResult if path found
        """
        import heapq

        # Priority queue: (cost, state, path, transitions)
        pq = [(0, start, [start], [])]
        visited = set()
        costs = {start: 0}

        while pq:
            current_cost, current, path, transitions = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current == goal:
                return TraversalResult(
                    path=path,
                    transitions=transitions,
                    cost=current_cost,
                    visited_states=visited,
                    success=True,
                    metadata={"strategy": "dijkstra", "nodes_explored": len(visited)},
                )

            state = self.state_graph.get_state(current)
            if not state:
                continue

            for transition in state.transitions:
                next_state = transition.to_state

                # Calculate transition cost
                transition_cost = self._get_transition_cost(transition)
                new_cost = current_cost + transition_cost

                if next_state not in costs or new_cost < costs[next_state]:
                    costs[next_state] = new_cost
                    heapq.heappush(
                        pq, (new_cost, next_state, path + [next_state], transitions + [transition])
                    )

        return TraversalResult(
            path=[],
            transitions=[],
            cost=float("inf"),
            visited_states=visited,
            success=False,
            metadata={"strategy": "dijkstra", "nodes_explored": len(visited)},
        )

    def _a_star_search(self, start: str, goal: str) -> TraversalResult | None:
        """A* search for finding optimal path.

        Args:
            start: Starting state
            goal: Goal state

        Returns:
            TraversalResult if path found
        """
        import heapq

        # Priority queue: (f_score, cost, state, path, transitions)
        pq = [(0, 0, start, [start], [])]
        visited = set()
        g_scores = {start: 0}

        while pq:
            _, current_cost, current, path, transitions = heapq.heappop(pq)

            if current in visited:
                continue

            visited.add(current)

            if current == goal:
                return TraversalResult(
                    path=path,
                    transitions=transitions,
                    cost=current_cost,
                    visited_states=visited,
                    success=True,
                    metadata={"strategy": "a_star", "nodes_explored": len(visited)},
                )

            state = self.state_graph.get_state(current)
            if not state:
                continue

            for transition in state.transitions:
                next_state = transition.to_state

                # Calculate g score (cost so far)
                transition_cost = self._get_transition_cost(transition)
                new_g_score = current_cost + transition_cost

                if next_state not in g_scores or new_g_score < g_scores[next_state]:
                    g_scores[next_state] = new_g_score

                    # Calculate h score (heuristic)
                    h_score = self._heuristic(next_state, goal)
                    f_score = new_g_score + h_score

                    heapq.heappush(
                        pq,
                        (
                            f_score,
                            new_g_score,
                            next_state,
                            path + [next_state],
                            transitions + [transition],
                        ),
                    )

        return TraversalResult(
            path=[],
            transitions=[],
            cost=float("inf"),
            visited_states=visited,
            success=False,
            metadata={"strategy": "a_star", "nodes_explored": len(visited)},
        )

    def _random_search(
        self, start: str, goal: str, max_steps: int = 1000
    ) -> TraversalResult | None:
        """Random walk search for finding path.

        Args:
            start: Starting state
            goal: Goal state
            max_steps: Maximum number of steps

        Returns:
            TraversalResult if path found
        """
        current = start
        path = [start]
        transitions = []
        visited = {start}

        for step in range(max_steps):
            if current == goal:
                return TraversalResult(
                    path=path,
                    transitions=transitions,
                    cost=len(transitions),
                    visited_states=visited,
                    success=True,
                    metadata={"strategy": "random", "steps": step},
                )

            state = self.state_graph.get_state(current)
            if not state or not state.transitions:
                break

            # Random transition
            transition = random.choice(state.transitions)
            current = transition.to_state
            path.append(current)
            transitions.append(transition)
            visited.add(current)

        return TraversalResult(
            path=[],
            transitions=[],
            cost=float("inf"),
            visited_states=visited,
            success=False,
            metadata={"strategy": "random", "steps": max_steps},
        )

    def _exploratory_search(self, start: str, goal: str) -> TraversalResult | None:
        """Exploratory search that prioritizes unvisited states.

        Args:
            start: Starting state
            goal: Goal state

        Returns:
            TraversalResult if path found
        """
        queue = deque([(start, [], [], set())])
        global_visited = set()

        while queue:
            current, path, transitions, local_visited = queue.popleft()

            if current == goal:
                return TraversalResult(
                    path=path + [goal],
                    transitions=transitions,
                    cost=len(transitions),
                    visited_states=global_visited | {goal},
                    success=True,
                    metadata={"strategy": "exploratory", "nodes_explored": len(global_visited)},
                )

            global_visited.add(current)
            local_visited.add(current)

            state = self.state_graph.get_state(current)
            if not state:
                continue

            # Sort transitions by novelty (prefer unvisited states)
            sorted_transitions = sorted(
                state.transitions,
                key=lambda t: (t.to_state in local_visited, t.to_state in global_visited),
            )

            for transition in sorted_transitions:
                next_state = transition.to_state
                if next_state not in local_visited:
                    queue.append(
                        (
                            next_state,
                            path + [current],
                            transitions + [transition],
                            local_visited.copy(),
                        )
                    )

        return TraversalResult(
            path=[],
            transitions=[],
            cost=float("inf"),
            visited_states=global_visited,
            success=False,
            metadata={"strategy": "exploratory", "nodes_explored": len(global_visited)},
        )

    def _get_transition_cost(self, transition: Transition) -> float:
        """Get cost of a transition.

        Args:
            transition: Transition to evaluate

        Returns:
            Cost value
        """
        if self.cost_function:
            return self.cost_function(transition)

        # Default costs based on action type
        costs = {
            TransitionType.CLICK: 1.0,
            TransitionType.TYPE: 2.0,
            TransitionType.HOVER: 0.5,
            TransitionType.DRAG: 3.0,
            TransitionType.SCROLL: 1.5,
            TransitionType.KEY_PRESS: 1.0,
            TransitionType.WAIT: 0.1,
            TransitionType.CUSTOM: 5.0,
        }

        return costs.get(transition.action_type, 1.0) / transition.probability

    def _heuristic(self, state: str, goal: str) -> float:
        """Heuristic function for A* search.

        Args:
            state: Current state
            goal: Goal state

        Returns:
            Heuristic value
        """
        # Simple heuristic: minimum number of transitions
        # In practice, could use more sophisticated heuristics
        path = self.state_graph.find_path(state, goal)
        if path:
            return len(path) - 1
        return float("inf")

    def set_cost_function(self, cost_func: Callable[[Transition], float]):
        """Set custom cost function for transitions.

        Args:
            cost_func: Function that takes a Transition and returns cost
        """
        self.cost_function = cost_func

    def explore_graph(self, start: str, max_depth: int = 10) -> set[str]:
        """Explore the graph from a starting state.

        Args:
            start: Starting state
            max_depth: Maximum exploration depth

        Returns:
            Set of discovered state names
        """
        discovered = set()
        queue = deque([(start, 0)])

        while queue:
            current, depth = queue.popleft()

            if current in discovered or depth >= max_depth:
                continue

            discovered.add(current)

            state = self.state_graph.get_state(current)
            if state:
                for transition in state.transitions:
                    if transition.to_state not in discovered:
                        queue.append((transition.to_state, depth + 1))

        return discovered

    def get_reachable_states(self, start: str) -> set[str]:
        """Get all states reachable from a starting state.

        Args:
            start: Starting state

        Returns:
            Set of reachable state names
        """
        return self.explore_graph(start, max_depth=float("inf"))
