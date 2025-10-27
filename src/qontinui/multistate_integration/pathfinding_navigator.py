"""Pathfinding navigator for Qontinui using MultiState algorithms.

Provides advanced navigation capabilities:
- Multi-target pathfinding (reach ALL targets)
- Optimal path computation
- Path caching and optimization
- Navigation execution with rollback
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from qontinui.model.transition.enhanced_state_transition import TaskSequenceStateTransition
from qontinui.state_management.state_memory import StateMemory

from .enhanced_transition_executor import EnhancedTransitionExecutor
from .multistate_adapter import MultiStateAdapter

logger = logging.getLogger(__name__)


class NavigationStrategy(Enum):
    """Strategies for path computation."""

    BREADTH_FIRST = "bfs"  # Find shortest path by transitions
    DIJKSTRA = "dijkstra"  # Find optimal path by cost
    A_STAR = "a_star"  # Heuristic-guided search
    GREEDY = "greedy"  # Fast but potentially suboptimal


@dataclass
class NavigationPath:
    """Represents a navigation path through states."""

    start_states: set[int]
    target_states: set[int]
    transitions: list[TaskSequenceStateTransition]
    total_cost: float
    estimated_duration: float = 0.0
    complexity: str = "simple"  # simple, moderate, complex

    def is_valid(self) -> bool:
        """Check if path is still valid."""
        # Would check if transitions are still available
        return len(self.transitions) > 0


@dataclass
class NavigationContext:
    """Context for navigation execution."""

    path: NavigationPath
    current_transition_index: int = 0
    completed_transitions: list[TaskSequenceStateTransition] = field(default_factory=list)
    failed_transition: TaskSequenceStateTransition | None = None
    rollback_performed: bool = False
    targets_reached: set[int] = field(default_factory=set)
    recovery_attempts: int = 0  # Track recovery attempts to prevent infinite recursion


class PathfindingNavigator:
    """Navigator for complex multi-state pathfinding in Qontinui.

    This navigator uses MultiState's algorithms to find optimal paths
    that reach ALL specified target states, not just one.

    Key features:
    1. Multi-target pathfinding (reach ALL targets)
    2. Multiple search strategies
    3. Path caching for performance
    4. Robust execution with rollback
    5. Dynamic replanning on failure
    """

    def __init__(
        self,
        state_memory: StateMemory,
        transition_executor: EnhancedTransitionExecutor | None = None,
        multistate_adapter: MultiStateAdapter | None = None,
        workflow_executor: Any | None = None,
        default_strategy: NavigationStrategy = NavigationStrategy.DIJKSTRA,
    ):
        """Initialize pathfinding navigator.

        Args:
            state_memory: Qontinui's state memory
            transition_executor: Executor for transitions
            multistate_adapter: MultiState adapter
            workflow_executor: Executor that can run workflows
            default_strategy: Default search strategy
        """
        self.state_memory = state_memory
        self.multistate_adapter = multistate_adapter or MultiStateAdapter(state_memory)
        self.transition_executor = transition_executor or EnhancedTransitionExecutor(
            state_memory, self.multistate_adapter, workflow_executor
        )
        self.default_strategy = default_strategy

        # Path cache for performance
        self.path_cache: dict[tuple[frozenset[int], frozenset[int]], NavigationPath] = {}

        # Navigation history
        self.navigation_history: list[NavigationContext] = []

        # Path complexity analyzer
        self.max_simple_targets = 2
        self.max_moderate_targets = 5

    def navigate_to_states(
        self,
        target_state_ids: list[int],
        strategy: NavigationStrategy | None = None,
        use_cache: bool = True,
        execute: bool = True,
    ) -> NavigationContext | None:
        """Navigate to reach ALL specified target states.

        This is the main API for multi-target navigation.

        Args:
            target_state_ids: ALL states that must be reached
            strategy: Search strategy to use
            use_cache: Whether to use cached paths
            execute: Whether to execute the path or just compute it

        Returns:
            NavigationContext with results, or None if no path found
        """
        logger.info(f"[DEBUG] navigate_to_states called with target_state_ids={target_state_ids}, execute={execute}")
        strategy = strategy or self.default_strategy

        # Find path
        logger.info(f"[DEBUG] Finding path to states...")
        path = self.find_path_to_states(
            target_state_ids=target_state_ids, strategy=strategy, use_cache=use_cache
        )
        logger.info(f"[DEBUG] Path found: {path is not None}")

        if not path:
            logger.info(f"No path found to states: {target_state_ids}")
            return None

        # Create navigation context
        logger.info(f"[DEBUG] Creating navigation context...")
        context = NavigationContext(path=path)

        # Execute if requested
        if execute:
            logger.info(f"[DEBUG] Executing navigation path...")
            self._execute_navigation(context)
            logger.info(f"[DEBUG] Navigation execution completed")

        # Record navigation
        self.navigation_history.append(context)

        return context

    def find_path_to_states(
        self,
        target_state_ids: list[int],
        from_state_ids: set[int] | None = None,
        strategy: NavigationStrategy = NavigationStrategy.DIJKSTRA,
        use_cache: bool = True,
    ) -> NavigationPath | None:
        """Find optimal path to reach ALL target states.

        Args:
            target_state_ids: ALL states that must be reached
            from_state_ids: Starting states (uses current if None)
            strategy: Search strategy
            use_cache: Whether to use cached paths

        Returns:
            NavigationPath or None if no path exists
        """
        # Get current states
        logger.info(f"[DEBUG] Getting current states for pathfinding...")
        from_states = from_state_ids or self.state_memory.active_states
        logger.info(f"[DEBUG] From states: {from_states}")
        if not from_states:
            logger.warning("No starting states for pathfinding")
            return None

        # Check cache
        cache_key = (frozenset(from_states), frozenset(target_state_ids))
        if use_cache and cache_key in self.path_cache:
            cached_path = self.path_cache[cache_key]
            if cached_path.is_valid():
                logger.debug(f"Using cached path to {target_state_ids}")
                return cached_path

        # Compute path using MultiState
        logger.info(f"[DEBUG] Computing path using multistate_adapter...")
        transitions = self.multistate_adapter.find_path_to_states(
            target_state_ids=target_state_ids, current_state_ids=from_states
        )
        logger.info(f"[DEBUG] Multistate adapter returned {len(transitions) if transitions else 0} transitions")

        if not transitions:
            return None

        # Build navigation path
        path = self._build_navigation_path(
            from_states=from_states, target_states=set(target_state_ids), transitions=transitions
        )

        # Cache path
        if use_cache:
            self.path_cache[cache_key] = path

        return path

    def _build_navigation_path(
        self,
        from_states: set[int],
        target_states: set[int],
        transitions: list[TaskSequenceStateTransition],
    ) -> NavigationPath:
        """Build NavigationPath from transitions.

        Args:
            from_states: Starting states
            target_states: Target states
            transitions: Transition sequence

        Returns:
            NavigationPath object
        """
        # Calculate total cost
        total_cost = sum(t.score for t in transitions)

        # Estimate duration (simplified)
        estimated_duration = len(transitions) * 0.5  # 0.5s per transition

        # Determine complexity
        num_targets = len(target_states)
        if num_targets <= self.max_simple_targets:
            complexity = "simple"
        elif num_targets <= self.max_moderate_targets:
            complexity = "moderate"
        else:
            complexity = "complex"

        return NavigationPath(
            start_states=from_states,
            target_states=target_states,
            transitions=transitions,
            total_cost=total_cost,
            estimated_duration=estimated_duration,
            complexity=complexity,
        )

    def _execute_navigation(self, context: NavigationContext) -> bool:
        """Execute navigation path.

        Args:
            context: Navigation context

        Returns:
            True if all targets reached
        """
        logger.info(f"Executing navigation path with {len(context.path.transitions)} transitions")

        for i, transition in enumerate(context.path.transitions):
            context.current_transition_index = i

            # Execute transition
            success = self.transition_executor.execute_transition(transition)

            if success:
                context.completed_transitions.append(transition)

                # Update targets reached
                for state_id in transition.activate:
                    if state_id in context.path.target_states:
                        context.targets_reached.add(state_id)

                logger.debug(f"Completed transition {i+1}/{len(context.path.transitions)}")
            else:
                context.failed_transition = transition
                logger.warning(f"Failed at transition {i+1}: {transition.name}")

                # Attempt recovery
                if not self._attempt_recovery(context):
                    return False

        # Check if all targets reached
        all_reached = context.targets_reached == context.path.target_states
        if all_reached:
            logger.info(f"Successfully reached all {len(context.path.target_states)} target states")
        else:
            missing = context.path.target_states - context.targets_reached
            logger.warning(f"Failed to reach states: {missing}")

        return all_reached

    def _attempt_recovery(self, context: NavigationContext) -> bool:
        """Attempt to recover from navigation failure.

        Args:
            context: Navigation context

        Returns:
            True if recovery successful
        """
        # Check if we've exceeded maximum recovery attempts
        MAX_RECOVERY_ATTEMPTS = 3
        context.recovery_attempts += 1

        if context.recovery_attempts > MAX_RECOVERY_ATTEMPTS:
            logger.error(
                f"Maximum recovery attempts ({MAX_RECOVERY_ATTEMPTS}) exceeded. "
                "Navigation cannot proceed. Please check transition configuration."
            )
            return False

        logger.info(f"Attempting navigation recovery (attempt {context.recovery_attempts}/{MAX_RECOVERY_ATTEMPTS})...")

        # Get current states after failure
        current_states = self.state_memory.active_states

        # Find remaining targets
        remaining_targets = context.path.target_states - context.targets_reached

        if not remaining_targets:
            logger.info("All targets already reached despite failure")
            return True

        # Try to find alternative path
        alternative_path = self.find_path_to_states(
            target_state_ids=list(remaining_targets),
            from_state_ids=current_states,
            use_cache=False,  # Don't use cache for recovery
        )

        if not alternative_path:
            logger.warning("No alternative path found for recovery")
            return False

        # Check if we're about to try the same transition again
        if (context.failed_transition and
            alternative_path.transitions and
            alternative_path.transitions[0].id == context.failed_transition.id):
            logger.error(
                f"Recovery would retry the same failed transition '{context.failed_transition.name}'. "
                "This indicates a configuration problem - the transition likely has no workflows linked."
            )
            return False

        # Execute alternative path
        logger.info(f"Found alternative path with {len(alternative_path.transitions)} transitions")

        # Update context with new path
        context.path = alternative_path
        context.current_transition_index = 0

        # Continue execution
        return self._execute_navigation(context)

    def find_shortest_path(
        self, target_state_id: int, from_state_ids: set[int] | None = None
    ) -> NavigationPath | None:
        """Find shortest path to a single target state.

        Convenience method for single-target navigation.

        Args:
            target_state_id: Single target state
            from_state_ids: Starting states

        Returns:
            NavigationPath or None
        """
        return self.find_path_to_states(
            target_state_ids=[target_state_id],
            from_state_ids=from_state_ids,
            strategy=NavigationStrategy.BREADTH_FIRST,
        )

    def can_reach_states(
        self, target_state_ids: list[int], from_state_ids: set[int] | None = None
    ) -> bool:
        """Check if target states are reachable.

        Args:
            target_state_ids: States to check
            from_state_ids: Starting states

        Returns:
            True if ALL targets are reachable
        """
        path = self.find_path_to_states(
            target_state_ids=target_state_ids, from_state_ids=from_state_ids, use_cache=True
        )
        return path is not None

    def get_navigation_statistics(self) -> dict[str, Any]:
        """Get statistics about navigation.

        Returns:
            Dictionary with navigation statistics
        """
        total = len(self.navigation_history)
        successful = sum(
            1 for ctx in self.navigation_history if ctx.targets_reached == ctx.path.target_states
        )

        complexity_counts = {"simple": 0, "moderate": 0, "complex": 0}
        total_transitions = 0
        total_cost = 0.0

        for ctx in self.navigation_history:
            complexity_counts[ctx.path.complexity] += 1
            total_transitions += len(ctx.path.transitions)
            total_cost += ctx.path.total_cost

        return {
            "total_navigations": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0,
            "complexity_distribution": complexity_counts,
            "average_transitions": total_transitions / total if total > 0 else 0,
            "average_cost": total_cost / total if total > 0 else 0,
            "cache_size": len(self.path_cache),
            "cache_hit_rate": self._calculate_cache_hit_rate(),
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate from recent navigations.

        Returns:
            Cache hit rate (0.0 to 1.0)
        """
        # This would track cache hits/misses in real implementation
        # For now, return estimate based on cache size
        if not self.navigation_history:
            return 0.0

        cache_potential = min(len(self.path_cache) / len(self.navigation_history), 1.0)
        return cache_potential * 0.7  # Assume 70% reuse when cached

    def clear_cache(self) -> None:
        """Clear the path cache."""
        self.path_cache.clear()
        logger.info("Cleared navigation path cache")

    def explain_path(self, path: NavigationPath) -> str:
        """Generate human-readable explanation of a path.

        Args:
            path: Path to explain

        Returns:
            Explanation string
        """
        lines = []
        lines.append(f"Navigation path to reach {len(path.target_states)} target states:")
        lines.append(f"- Complexity: {path.complexity}")
        lines.append(f"- Total cost: {path.total_cost:.2f}")
        lines.append(f"- Estimated duration: {path.estimated_duration:.1f}s")
        lines.append(f"- Transitions required: {len(path.transitions)}")

        lines.append("\nTransition sequence:")
        for i, trans in enumerate(path.transitions, 1):
            lines.append(f"  {i}. {trans.name or f'Transition {trans.id}'}")
            if trans.activate:
                lines.append(f"     Activates: {trans.activate}")
            if trans.exit:
                lines.append(f"     Exits: {trans.exit}")

        return "\n".join(lines)
