"""CLI entry point for HTN plan execution.

Usage: python -m qontinui.planning_integration

Reads JSON config from stdin:
    {
        "task": "login" or ["login", "admin", "pass"],
        "ui_bridge_url": "http://localhost:1420",
        "target_type": "web",
        "state_machine_path": "/path/to/sm.json" or null,
        "planning_timeout_ms": 5000,
        "max_replans": 5
    }

Outputs JSON result on stdout (last line):
    {
        "plan_found": bool,
        "execution_success": bool,
        "plan_actions": int,
        "steps_succeeded": int,
        "replans": int,
        "total_time_ms": float,
        "summary": str,
        "error": str or null
    }
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from typing import Any

logger = logging.getLogger(__name__)
# Log to stderr so stdout stays clean for the JSON result
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _emit(result: dict[str, Any]) -> None:
    """Print result as JSON on stdout."""
    # Ensure ASCII so Windows console doesn't choke on unicode
    print(json.dumps(result, ensure_ascii=True))


def _failure(start: float, summary: str, error: str) -> dict[str, Any]:
    return {
        "plan_found": False,
        "execution_success": False,
        "plan_actions": 0,
        "steps_succeeded": 0,
        "replans": 0,
        "total_time_ms": (time.time() - start) * 1000,
        "summary": summary,
        "error": error,
    }


def main() -> None:
    start = time.time()

    # Parse stdin config
    try:
        raw = sys.stdin.read()
        cfg = json.loads(raw)
    except Exception as exc:
        _emit(_failure(start, f"Bad stdin config: {exc}", str(exc)))
        return

    task_raw = cfg.get("task", "")
    ui_bridge_url = cfg.get("ui_bridge_url")
    target_type = cfg.get("target_type", "web")
    state_machine_path = cfg.get("state_machine_path")
    methods_directory = cfg.get("methods_directory")
    max_replans = int(cfg.get("max_replans", 5))

    # Normalize task: string -> single-element tuple, list -> tuple
    if isinstance(task_raw, str):
        task_tuple: tuple[Any, ...] = (task_raw,)
    elif isinstance(task_raw, list):
        task_tuple = tuple(task_raw)
    else:
        _emit(
            _failure(
                start,
                "Invalid task type",
                f"task must be str or list, got {type(task_raw).__name__}",
            )
        )
        return

    # Imports - defer to catch missing deps gracefully
    try:
        from multistate.manager import StateManager
        from multistate.planning.blackboard import Blackboard
        from multistate.planning.executor import PlanExecutor
        from multistate.planning.registry import create_default_registry
        from multistate.planning.world_adapter import WorldStateAdapter

        from qontinui.planning_integration.action_handlers import (
            create_action_handlers,
        )
        from qontinui.planning_integration.world_state_bridge import (
            create_world_state_snapshot,
            populate_blackboard,
            run_async_safe,
        )
    except ImportError as exc:
        _emit(_failure(start, f"Import error: {exc}", str(exc)))
        return

    # Load or create StateManager
    try:
        if state_machine_path:
            with open(state_machine_path, encoding="utf-8") as f:
                sm_data = json.load(f)
            state_manager = StateManager.from_dict(sm_data)
            logger.info(
                "Loaded StateManager from %s: %d states, %d transitions",
                state_machine_path,
                len(state_manager.states),
                len(state_manager.transitions),
            )
        else:
            state_manager = StateManager()
            logger.info("Using empty StateManager (no state_machine_path)")
    except Exception as exc:
        _emit(_failure(start, f"Failed to load StateManager: {exc}", str(exc)))
        return

    adapter = WorldStateAdapter(state_manager)

    # Initialize HAL
    hal: Any = None
    try:
        from qontinui.hal import initialize_hal

        hal = initialize_hal()
        logger.info("HAL initialized")
    except Exception as exc:
        logger.warning("HAL init failed: %s - continuing without HAL", exc)
        # Continue anyway; planning may still succeed, execution will fail on
        # action handlers.

    # Connect to UI Bridge (optional)
    connection: Any = None
    connection_cm: Any = None
    try:
        if ui_bridge_url:
            try:
                from qontinui.discovery.target_connection import (
                    ExplorationConfig,
                    create_connection,
                )

                conn_config = ExplorationConfig(
                    target_type=target_type,
                    connection_url=ui_bridge_url,
                    timeout_seconds=30.0,
                )
                # create_connection returns a TargetConnection which is an async
                # context manager; enter it manually so we can keep it alive for
                # the duration of orchestration.
                connection_cm = create_connection(conn_config)
                connection = run_async_safe(connection_cm.__aenter__())
                logger.info("Connected to UI Bridge at %s", ui_bridge_url)
            except Exception as exc:
                logger.warning("UI Bridge connection failed: %s - continuing without", exc)
                connection = None
                connection_cm = None

        # Build planner with default + custom methods
        registry = create_default_registry()
        if methods_directory:
            try:
                from multistate.planning.methods.loader import MethodLoader

                loaded = MethodLoader.load_from_directory(methods_directory)
                for task_name, method_list in loaded.items():
                    registry.register_methods(task_name, method_list)
                logger.info(
                    "Loaded %d methods from %s",
                    sum(len(v) for v in loaded.values()),
                    methods_directory,
                )
            except Exception as exc:
                logger.warning("Failed to load methods from %s: %s", methods_directory, exc)
        planner = registry.build_planner()

        # If no connection, do plan-only (no execution)
        if connection is None:
            try:
                # Minimal world state: just what we know from StateManager
                world_state = adapter.snapshot()
                plan_result = planner.find_plan(world_state, [task_tuple])

                if not plan_result.success:
                    _emit(
                        _failure(
                            start,
                            f"No plan found (no UI Bridge): {plan_result.error}",
                            plan_result.error or "No plan",
                        )
                    )
                    return

                if not plan_result.actions:
                    # Goal already satisfied — nothing to execute
                    elapsed = (time.time() - start) * 1000
                    _emit(
                        {
                            "plan_found": True,
                            "execution_success": True,
                            "plan_actions": 0,
                            "steps_succeeded": 0,
                            "replans": 0,
                            "total_time_ms": elapsed,
                            "summary": "Goal already satisfied (no actions needed, no UI Bridge)",
                            "error": None,
                        }
                    )
                    return

                elapsed = (time.time() - start) * 1000
                # Plan-only: report as "plan_found but not executed"
                actions_summary = "; ".join(
                    " ".join(str(a) for a in action) for action in plan_result.actions[:5]
                )
                if len(plan_result.actions) > 5:
                    actions_summary += f"... (+{len(plan_result.actions) - 5} more)"

                _emit(
                    {
                        "plan_found": True,
                        "execution_success": False,  # Not executed without UI Bridge
                        "plan_actions": len(plan_result.actions),
                        "steps_succeeded": 0,
                        "replans": 0,
                        "total_time_ms": elapsed,
                        "summary": f"Plan found (not executed, no UI Bridge): {actions_summary}",
                        "error": "No UI Bridge connection - plan not executed",
                    }
                )
                return
            except Exception as exc:
                _emit(
                    _failure(
                        start,
                        f"Planning error: {exc}",
                        f"{exc}\n{traceback.format_exc()}",
                    )
                )
                return

        # Full orchestration: snapshot world state from UI Bridge + plan + execute
        try:
            world_state = run_async_safe(create_world_state_snapshot(adapter, connection))
        except Exception as exc:
            _emit(_failure(start, f"World state snapshot failed: {exc}", str(exc)))
            return

        plan_result = planner.find_plan(world_state, [task_tuple])

        if not plan_result.success:
            _emit(
                _failure(
                    start,
                    f"No plan found: {plan_result.error}",
                    plan_result.error or "No plan",
                )
            )
            return

        if not plan_result.actions:
            # Goal already satisfied — nothing to execute
            elapsed = (time.time() - start) * 1000
            _emit(
                {
                    "plan_found": True,
                    "execution_success": True,
                    "plan_actions": 0,
                    "steps_succeeded": 0,
                    "replans": 0,
                    "total_time_ms": elapsed,
                    "summary": "Goal already satisfied (no actions needed)",
                    "error": None,
                }
            )
            return

        # Create blackboard and populate
        blackboard = Blackboard()
        populate_blackboard(
            blackboard,
            dict(world_state.element_visible),
            dict(world_state.element_values),
            connection=connection,
            state_manager=state_manager,
        )

        # Action handlers need HAL; if HAL init failed, execution will fail cleanly
        if hal is None:
            elapsed = (time.time() - start) * 1000
            _emit(
                {
                    "plan_found": True,
                    "execution_success": False,
                    "plan_actions": len(plan_result.actions),
                    "steps_succeeded": 0,
                    "replans": 0,
                    "total_time_ms": elapsed,
                    "summary": "Plan found but HAL unavailable",
                    "error": "HAL initialization failed",
                }
            )
            return

        action_handlers = create_action_handlers(
            hal=hal,
            ui_connection=connection,
            state_manager=state_manager,
        )

        executor = PlanExecutor(
            planner=planner,
            adapter=adapter,
            action_handlers=action_handlers,
            max_replans=max_replans,
        )

        try:
            exec_result = executor.execute(
                plan=plan_result.actions,
                initial_state=world_state,
                original_tasks=[task_tuple],
                blackboard=blackboard,
            )
        except Exception as exc:
            _emit(
                _failure(
                    start,
                    f"Plan execution crashed: {exc}",
                    f"{exc}\n{traceback.format_exc()}",
                )
            )
            return

        elapsed = (time.time() - start) * 1000
        steps_succeeded = sum(
            1 for s in exec_result.steps_executed if getattr(s.status, "value", "") == "success"
        )

        actions_summary = "; ".join(
            " ".join(str(a) for a in action) for action in plan_result.actions[:5]
        )
        if len(plan_result.actions) > 5:
            actions_summary += f"... (+{len(plan_result.actions) - 5} more)"

        _emit(
            {
                "plan_found": True,
                "execution_success": exec_result.success,
                "plan_actions": len(plan_result.actions),
                "steps_succeeded": steps_succeeded,
                "replans": exec_result.replans,
                "total_time_ms": elapsed,
                "summary": (
                    f"Executed {steps_succeeded}/{len(plan_result.actions)} actions, "
                    f"{exec_result.replans} replans: {actions_summary}"
                ),
                "error": exec_result.error,
            }
        )
    finally:
        if connection_cm is not None:
            try:
                run_async_safe(connection_cm.__aexit__(None, None, None))
            except Exception:
                pass


if __name__ == "__main__":
    main()
