"""Tests for the orchestrator module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from multistate.planning.executor import ExecutionResult
from multistate.planning.planner import HTNPlanner, PlanResult
from multistate.planning.world_adapter import WorldStateAdapter

from qontinui.discovery.target_connection import Element
from qontinui.planning_integration.orchestrator import (
    run_htn_plan,
    run_htn_plan_sync,
)


def _make_element(
    id: str,
    tag_name: str,
    is_visible: bool = True,
    text_content: str | None = None,
    attributes: dict[str, str] | None = None,
) -> Element:
    return Element(
        id=id,
        tag_name=tag_name,
        is_visible=is_visible,
        text_content=text_content,
        attributes=attributes or {},
    )


def _mock_connection() -> AsyncMock:
    conn = AsyncMock()
    conn.find_elements.return_value = [
        _make_element(id="btn-go", tag_name="button", is_visible=True, text_content="Go"),
    ]
    return conn


def _mock_adapter() -> WorldStateAdapter:
    manager = MagicMock()
    manager.get_active_states.return_value = {"state_main"}
    manager.get_available_transitions.return_value = []
    return WorldStateAdapter(manager)


def _mock_hal() -> MagicMock:
    hal = MagicMock()
    hal.mouse_controller = MagicMock()
    hal.keyboard_controller = MagicMock()
    return hal


# ------------------------------------------------------------------
# test_run_htn_plan_success
# ------------------------------------------------------------------


def test_run_htn_plan_success() -> None:
    """Plan found and execution succeeds."""
    conn = _mock_connection()
    adapter = _mock_adapter()
    hal = _mock_hal()
    planner = MagicMock(spec=HTNPlanner)

    planner.find_plan.return_value = PlanResult(
        success=True,
        actions=[("click_element", "btn-go")],
        planning_time_ms=1.0,
        nodes_explored=5,
    )

    success_result = ExecutionResult(success=True)
    with patch(
        "qontinui.planning_integration.orchestrator.PlanExecutor"
    ) as MockExecutor:
        instance = MockExecutor.return_value
        instance.execute.return_value = success_result

        result = asyncio.run(
            run_htn_plan(
                task=("do_thing",),
                adapter=adapter,
                connection=conn,
                hal=hal,
                planner=planner,
            )
        )

    assert result.success is True
    planner.find_plan.assert_called_once()
    instance.execute.assert_called_once()


# ------------------------------------------------------------------
# test_run_htn_plan_no_plan_found
# ------------------------------------------------------------------


def test_run_htn_plan_no_plan_found() -> None:
    """When the planner finds no plan, return a failure result."""
    conn = _mock_connection()
    adapter = _mock_adapter()
    hal = _mock_hal()
    planner = MagicMock(spec=HTNPlanner)

    planner.find_plan.return_value = PlanResult(
        success=False,
        actions=[],
        planning_time_ms=2.0,
        nodes_explored=100,
        error="No plan found within search limits",
    )

    result = asyncio.run(
        run_htn_plan(
            task=("impossible_task",),
            adapter=adapter,
            connection=conn,
            hal=hal,
            planner=planner,
        )
    )

    assert result.success is False
    assert result.error is not None
    assert "No plan found" in result.error


# ------------------------------------------------------------------
# test_run_htn_plan_sync_wrapper
# ------------------------------------------------------------------


def test_run_htn_plan_sync_wrapper() -> None:
    """Sync wrapper calls run_htn_plan and returns the result."""
    conn = _mock_connection()
    adapter = _mock_adapter()
    hal = _mock_hal()
    planner = MagicMock(spec=HTNPlanner)

    planner.find_plan.return_value = PlanResult(
        success=True,
        actions=[],
        planning_time_ms=0.5,
        nodes_explored=1,
    )

    success_result = ExecutionResult(success=True)
    with patch(
        "qontinui.planning_integration.orchestrator.PlanExecutor"
    ) as MockExecutor:
        instance = MockExecutor.return_value
        instance.execute.return_value = success_result

        result = run_htn_plan_sync(
            task=("simple_task",),
            adapter=adapter,
            connection=conn,
            hal=hal,
            planner=planner,
        )

    assert result.success is True
