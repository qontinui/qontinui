"""Top-level orchestrator for HTN plan execution.

Provides a single entry point that:
1. Creates a WorldState from current UI Bridge + StateManager state
2. Creates an HTNPlanner with all registered methods
3. Plans and executes, with replanning on state divergence
4. Returns the execution result
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from multistate.planning.blackboard import Blackboard
from multistate.planning.executor import ExecutionResult, PlanExecutor
from multistate.planning.planner import HTNPlanner
from multistate.planning.registry import create_default_registry
from multistate.planning.world_adapter import WorldStateAdapter

from qontinui.planning_integration.action_handlers import create_action_handlers
from qontinui.planning_integration.world_state_bridge import (
    create_world_state_snapshot,
    populate_blackboard,
    run_async_safe,
)

if TYPE_CHECKING:
    from qontinui.discovery.target_connection import TargetConnection

logger = logging.getLogger(__name__)


async def run_htn_plan(
    task: tuple[Any, ...],
    adapter: WorldStateAdapter,
    connection: TargetConnection,
    hal: Any,
    planner: HTNPlanner | None = None,
) -> ExecutionResult:
    """Plan and execute an HTN task against a live UI Bridge connection.

    This is the main entry point that ties together world-state capture,
    HTN planning, blackboard population, and plan execution with
    automatic replanning on state divergence.

    Args:
        task: A single HTN task tuple, e.g. ``("fill_form", "login")``.
        adapter: A :class:`WorldStateAdapter` backed by a live
            :class:`StateManager`.
        connection: An active UI Bridge target connection.
        hal: An initialized HAL container for low-level input.
        planner: Optional pre-configured :class:`HTNPlanner`.  If ``None``,
            a planner is built from :func:`create_default_registry`.

    Returns:
        An :class:`ExecutionResult` summarising the outcome.
    """
    # 1. Capture initial world state from UI Bridge + StateManager
    logger.info("Capturing initial world state for task %s", task)
    world_state = await create_world_state_snapshot(adapter, connection)

    # 2. Build planner if not provided
    if planner is None:
        registry = create_default_registry()
        planner = registry.build_planner()

    # 3. Find plan
    logger.info("Searching for plan to accomplish %s", task)
    plan_result = planner.find_plan(world_state, [task])

    if not plan_result.success:
        logger.warning(
            "No plan found for task %s: %s (explored %d nodes in %.1fms)",
            task,
            plan_result.error,
            plan_result.nodes_explored,
            plan_result.planning_time_ms,
        )
        return ExecutionResult(
            success=False,
            error=plan_result.error or "No plan found",
        )

    logger.info(
        "Plan found: %d actions (explored %d nodes in %.1fms)",
        len(plan_result.actions),
        plan_result.nodes_explored,
        plan_result.planning_time_ms,
    )

    # 4. Create blackboard and populate with UI data
    blackboard = Blackboard()
    element_visible = dict(world_state.element_visible)
    element_values = dict(world_state.element_values)
    populate_blackboard(
        blackboard,
        element_visible,
        element_values,
        connection=connection,
        state_manager=adapter.manager,
    )

    # 5. Create action handlers
    action_handlers = create_action_handlers(
        hal=hal,
        ui_connection=connection,
        state_manager=adapter.manager,
    )

    # 6. Execute plan
    executor = PlanExecutor(
        planner=planner,
        adapter=adapter,
        action_handlers=action_handlers,
    )

    logger.info("Executing plan with %d actions", len(plan_result.actions))
    result = executor.execute(
        plan=plan_result.actions,
        initial_state=world_state,
        original_tasks=[task],
        blackboard=blackboard,
    )

    # 7. Report outcome
    if result.success:
        logger.info(
            "Plan execution succeeded: %d steps, %d replans",
            len(result.steps_executed),
            result.replans,
        )
    else:
        logger.warning(
            "Plan execution failed: %s (%d steps executed, %d replans)",
            result.error,
            len(result.steps_executed),
            result.replans,
        )

    return result


def run_htn_plan_sync(
    task: tuple[Any, ...],
    adapter: WorldStateAdapter,
    connection: TargetConnection,
    hal: Any,
    planner: HTNPlanner | None = None,
) -> ExecutionResult:
    """Synchronous wrapper around :func:`run_htn_plan`.

    Uses :func:`run_async_safe` to handle the async/sync boundary,
    working correctly whether or not an event loop is already running.

    Args:
        task: A single HTN task tuple.
        adapter: A WorldStateAdapter backed by a live StateManager.
        connection: An active UI Bridge target connection.
        hal: An initialized HAL container.
        planner: Optional pre-configured HTNPlanner.

    Returns:
        An ExecutionResult summarising the outcome.
    """
    return run_async_safe(run_htn_plan(task, adapter, connection, hal, planner=planner))
