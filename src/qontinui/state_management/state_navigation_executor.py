"""State navigation and pathfinding orchestration.

This module handles multi-state navigation with pathfinding, execution,
result building, and event emission coordination.

Architecture:
    - StateNavigationExecutor: Orchestrates navigation operations
    - Delegates to PathfindingNavigator for pathfinding
    - Builds rich result objects with path information
    - Coordinates event emission for navigation events

Key Features:
    1. Multi-target pathfinding via PathfindingNavigator
    2. Optional path execution
    3. Result building with path and targets reached
    4. Event emission coordination
    5. Partial success tracking
    6. Error handling and logging

Example:
    >>> executor = StateNavigationExecutor(navigator, event_emitter)
    >>> result = executor.navigate_to_states([1, 2, 3], execute=True, callback)
    >>> if result.success:
    ...     print(f"Reached all targets via {result.path_length} transitions")
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from qontinui.exceptions import StateException

logger = logging.getLogger(__name__)


@dataclass
class NavigationResult:
    """Result of a navigation operation.

    Attributes:
        success: Whether operation succeeded
        error_message: Error message if failed
        context: Additional context information
        target_state_ids: Target states to reach
        path: Sequence of transitions executed
        targets_reached: States that were successfully reached
        path_length: Number of transitions in path
        total_cost: Total cost of path
        partial_success: True if some but not all targets reached
    """

    success: bool
    error_message: Optional[str] = None
    context: dict[str, Any] = field(default_factory=dict)
    target_state_ids: list[int] = field(default_factory=list)
    path: list[Any] = field(default_factory=list)  # List of TaskSequenceStateTransition
    targets_reached: set[int] = field(default_factory=set)
    path_length: int = 0
    total_cost: float = 0.0
    partial_success: bool = False


class StateNavigationExecutor:
    """Orchestrates state navigation with pathfinding and result building.

    Handles the complete navigation flow including pathfinding, optional
    execution, result building, and event emission.

    Example:
        >>> executor = StateNavigationExecutor(navigator, event_emitter)
        >>> result = executor.navigate_to_states([1, 2, 3], execute=True)
        >>> if result.success:
        ...     print(f"Reached all targets")
        >>> elif result.partial_success:
        ...     print(f"Reached {len(result.targets_reached)} targets")
    """

    def __init__(self, navigator: Any, event_emitter: Any) -> None:
        """Initialize StateNavigationExecutor.

        Args:
            navigator: PathfindingNavigator for pathfinding
            event_emitter: StateEventEmitter for event emission
        """
        self.navigator = navigator
        self.event_emitter = event_emitter

    def navigate_to_states(
        self,
        target_state_ids: list[int],
        execute: bool = True,
        emit_event_callback: Optional[Callable[[str, dict], None]] = None,
    ) -> NavigationResult:
        """Navigate to reach ALL specified target states.

        Uses PathfindingNavigator to find optimal path and optionally execute it.
        This is multi-target navigation - ALL states must be reached for success.

        Args:
            target_state_ids: IDs of ALL states to reach
            execute: If True, execute the path; if False, just compute it
            emit_event_callback: Optional callback for emitting events

        Returns:
            NavigationResult with path and execution details

        Example:
            >>> # Find path only (don't execute)
            >>> result = executor.navigate_to_states([1, 2, 3], execute=False)
            >>> if result.success:
            ...     print(f"Path: {[t.name for t in result.path]}")
            >>>
            >>> # Find and execute path
            >>> result = executor.navigate_to_states([1, 2, 3], execute=True)
            >>> if result.success:
            ...     print(f"Reached all targets")
            >>> elif result.partial_success:
            ...     print(f"Reached {len(result.targets_reached)} of {len(result.target_state_ids)}")
        """
        logger.info(f"Navigating to states: {target_state_ids}, execute={execute}")

        # Emit starting event
        self.event_emitter.emit_navigation_start(target_state_ids, execute, emit_event_callback)

        try:
            # Execute pathfinding navigation
            nav_context = self._execute_navigation(target_state_ids, execute)

            if not nav_context:
                return self._handle_navigation_not_found(target_state_ids, emit_event_callback)

            # Build result from navigation context
            result = self._build_result(target_state_ids, nav_context)

            # Emit completion events and log
            self.event_emitter.emit_navigation_complete(result, nav_context, emit_event_callback)
            self._log_result(result, nav_context)

            return result

        except StateException as e:
            logger.error(f"State error navigating to states {target_state_ids}: {e}")
            self.event_emitter.emit_navigation_failed(target_state_ids, str(e), emit_event_callback)
            return NavigationResult(
                success=False, target_state_ids=target_state_ids, error_message=str(e)
            )

        except Exception as e:
            logger.error(f"Unexpected error navigating to states {target_state_ids}: {e}", exc_info=True)
            error_msg = f"Unexpected error: {e}"
            self.event_emitter.emit_navigation_failed(target_state_ids, error_msg, emit_event_callback)
            return NavigationResult(
                success=False, target_state_ids=target_state_ids, error_message=error_msg
            )

    def _execute_navigation(self, target_state_ids: list[int], execute: bool) -> Optional[Any]:
        """Execute pathfinding navigation to target states.

        Args:
            target_state_ids: IDs of states to reach
            execute: If True, execute path; if False, just compute it

        Returns:
            Navigation context if path found, None otherwise
        """
        return self.navigator.navigate_to_states(target_state_ids=target_state_ids, execute=execute)

    def _handle_navigation_not_found(
        self, target_state_ids: list[int], emit_event_callback: Optional[Callable[[str, dict], None]]
    ) -> NavigationResult:
        """Handle case where no path is found to target states.

        Args:
            target_state_ids: IDs of states that couldn't be reached
            emit_event_callback: Optional callback for emitting events

        Returns:
            Failed NavigationResult
        """
        error_msg = f"No path found to states: {target_state_ids}"
        logger.warning(error_msg)
        self.event_emitter.emit_navigation_failed(target_state_ids, error_msg, emit_event_callback)
        return NavigationResult(
            success=False,
            target_state_ids=target_state_ids,
            error_message=error_msg,
        )

    def _build_result(
        self, target_state_ids: list[int], nav_context: Any
    ) -> NavigationResult:
        """Build navigation result from navigation context.

        Args:
            target_state_ids: IDs of target states
            nav_context: Navigation context from PathfindingNavigator

        Returns:
            NavigationResult with execution details
        """
        all_reached = nav_context.targets_reached == nav_context.path.target_states
        some_reached = len(nav_context.targets_reached) > 0

        return NavigationResult(
            success=all_reached,
            target_state_ids=target_state_ids,
            path=nav_context.path.transitions,
            targets_reached=nav_context.targets_reached,
            path_length=len(nav_context.path.transitions),
            total_cost=nav_context.path.total_cost,
            partial_success=some_reached and not all_reached,
        )

    def _log_result(self, result: NavigationResult, nav_context: Any) -> None:
        """Log navigation execution result.

        Args:
            result: Navigation result
            nav_context: Navigation context from PathfindingNavigator
        """
        if result.success:
            logger.info(
                f"Navigation successful. Reached all {len(result.target_state_ids)} target states "
                f"in {len(nav_context.path.transitions)} transitions"
            )
        elif result.partial_success:
            logger.warning(
                f"Navigation partially successful. Reached {len(nav_context.targets_reached)} "
                f"of {len(result.target_state_ids)} target states"
            )
        else:
            logger.warning("Navigation failed. No target states reached")
