"""
Connection Routing Examples

Demonstrates the Connection Router system for graph-based workflows.
Shows routing logic for different action types.
"""

from qontinui.config.schema import (
    Action,
    Connection,
    Connections,
    Workflow,
)
from qontinui.execution import ConnectionRouter, RoutingContext

# ============================================================================
# Example 1: IF Action Routing
# ============================================================================


def example_if_routing():
    """
    Example: Routing through an IF action.

    Flow:
        start -> if_check -> (true: process_data, false: skip_processing) -> end
    """
    print("\n=== Example 1: IF Action Routing ===\n")

    # Create workflow
    workflow = Workflow(
        id="if_example",
        name="IF Routing Example",
        version="1.0.0",
        format="graph",
        actions=[
            Action(id="start", type="CLICK", config={}, position=(0, 100)),
            Action(
                id="if_check",
                type="IF",
                config={
                    "condition": {"type": "expression", "expression": "value > 10"},
                    "thenActions": ["process_data"],
                    "elseActions": ["skip_processing"],
                },
                position=(200, 100),
            ),
            Action(
                id="process_data",
                type="SET_VARIABLE",
                config={"variableName": "processed", "value": True},
                position=(400, 50),
            ),
            Action(
                id="skip_processing",
                type="WAIT",
                config={"waitFor": "time", "duration": 100},
                position=(400, 150),
            ),
            Action(id="end", type="SCREENSHOT", config={}, position=(600, 100)),
        ],
        connections=Connections(
            root={
                "start": {
                    "main": [[Connection(action="if_check", type="main", index=0)]]
                },
                "if_check": {
                    "true": [[Connection(action="process_data", type="true", index=0)]],
                    "false": [
                        [Connection(action="skip_processing", type="false", index=0)]
                    ],
                },
                "process_data": {
                    "main": [[Connection(action="end", type="main", index=0)]]
                },
                "skip_processing": {
                    "main": [[Connection(action="end", type="main", index=0)]]
                },
            }
        ),
    )

    # Create router with context tracking
    context = RoutingContext()
    router = ConnectionRouter(workflow=workflow, context=context)

    # Simulate execution: IF evaluates to TRUE
    print("Scenario: Condition evaluates to TRUE")
    if_action = workflow.actions[1]
    result_true = {"success": True, "condition_result": True}

    next_actions = router.route(if_action, result_true, workflow.connections)
    print(f"  Output type: {router.get_action_output_type(if_action, result_true)}")
    print(f"  Next actions: {[aid for aid, _, _ in next_actions]}")

    # Simulate execution: IF evaluates to FALSE
    print("\nScenario: Condition evaluates to FALSE")
    result_false = {"success": True, "condition_result": False}

    next_actions = router.route(if_action, result_false, workflow.connections)
    print(f"  Output type: {router.get_action_output_type(if_action, result_false)}")
    print(f"  Next actions: {[aid for aid, _, _ in next_actions]}")

    # Show routing history
    print("\nRouting history:")
    for record in context.get_route_history():
        print(f"  {record}")


# ============================================================================
# Example 2: LOOP Action Routing
# ============================================================================


def example_loop_routing():
    """
    Example: Routing through a LOOP action.

    Flow:
        start -> loop -> (loop: process_item, main: end)
    """
    print("\n=== Example 2: LOOP Action Routing ===\n")

    workflow = Workflow(
        id="loop_example",
        name="LOOP Routing Example",
        version="1.0.0",
        format="graph",
        actions=[
            Action(id="start", type="CLICK", config={}, position=(0, 100)),
            Action(
                id="loop",
                type="LOOP",
                config={
                    "loopType": "FOR",
                    "iterations": 3,
                    "iteratorVariable": "i",
                    "actions": ["process_item"],
                },
                position=(200, 100),
            ),
            Action(
                id="process_item",
                type="TYPE",
                config={"text": "item"},
                position=(400, 100),
            ),
            Action(id="end", type="SCREENSHOT", config={}, position=(600, 100)),
        ],
        connections=Connections(
            root={
                "start": {"main": [[Connection(action="loop", type="main", index=0)]]},
                "loop": {
                    "loop": [[Connection(action="process_item", type="loop", index=0)]],
                    "main": [[Connection(action="end", type="main", index=0)]],
                },
                "process_item": {
                    "main": [[Connection(action="loop", type="main", index=0)]]
                },
            }
        ),
    )

    context = RoutingContext()
    router = ConnectionRouter(workflow=workflow, context=context)
    loop_action = workflow.actions[1]

    # Iteration 1: Continue looping
    print("Iteration 1: Continue looping")
    result_continue = {
        "success": True,
        "continue_loop": True,
        "iteration": 1,
        "max_iterations": 3,
    }
    next_actions = router.route(loop_action, result_continue, workflow.connections)
    print(f"  Output: {router.get_action_output_type(loop_action, result_continue)}")
    print(f"  Next: {[aid for aid, _, _ in next_actions]}")

    # Final iteration: Exit loop
    print("\nIteration 3: Exit loop")
    result_exit = {
        "success": True,
        "continue_loop": False,
        "iteration": 3,
        "max_iterations": 3,
    }
    next_actions = router.route(loop_action, result_exit, workflow.connections)
    print(f"  Output: {router.get_action_output_type(loop_action, result_exit)}")
    print(f"  Next: {[aid for aid, _, _ in next_actions]}")


# ============================================================================
# Example 3: SWITCH Action Routing
# ============================================================================


def example_switch_routing():
    """
    Example: Routing through a SWITCH action.

    Routes to different actions based on case matching.
    """
    print("\n=== Example 3: SWITCH Action Routing ===\n")

    workflow = Workflow(
        id="switch_example",
        name="SWITCH Routing Example",
        version="1.0.0",
        format="graph",
        actions=[
            Action(
                id="start",
                type="GET_VARIABLE",
                config={"variableName": "status"},
                position=(0, 100),
            ),
            Action(
                id="switch",
                type="SWITCH",
                config={
                    "expression": "status",
                    "cases": [
                        {"value": "ready", "actions": ["handle_ready"]},
                        {"value": "error", "actions": ["handle_error"]},
                        {"value": "pending", "actions": ["handle_pending"]},
                    ],
                    "defaultActions": ["handle_unknown"],
                },
                position=(200, 100),
            ),
            Action(id="handle_ready", type="CLICK", config={}, position=(400, 0)),
            Action(
                id="handle_error", type="SCREENSHOT", config={}, position=(400, 100)
            ),
            Action(
                id="handle_pending",
                type="WAIT",
                config={"waitFor": "time", "duration": 1000},
                position=(400, 200),
            ),
            Action(
                id="handle_unknown", type="SCREENSHOT", config={}, position=(400, 300)
            ),
        ],
        connections=Connections(
            root={
                "start": {
                    "main": [[Connection(action="switch", type="main", index=0)]]
                },
                "switch": {
                    "case_0": [
                        [Connection(action="handle_ready", type="case_0", index=0)]
                    ],
                    "case_1": [
                        [Connection(action="handle_error", type="case_1", index=0)]
                    ],
                    "case_2": [
                        [Connection(action="handle_pending", type="case_2", index=0)]
                    ],
                    "default": [
                        [Connection(action="handle_unknown", type="default", index=0)]
                    ],
                },
            }
        ),
    )

    router = ConnectionRouter(workflow=workflow)
    switch_action = workflow.actions[1]

    # Case 0: ready
    print("Case 0: status = 'ready'")
    result_case0 = {"success": True, "case_index": 0, "matched_value": "ready"}
    next_actions = router.route(switch_action, result_case0, workflow.connections)
    print(f"  Output: {router.get_action_output_type(switch_action, result_case0)}")
    print(f"  Next: {[aid for aid, _, _ in next_actions]}")

    # Case 1: error
    print("\nCase 1: status = 'error'")
    result_case1 = {"success": True, "case_index": 1, "matched_value": "error"}
    next_actions = router.route(switch_action, result_case1, workflow.connections)
    print(f"  Output: {router.get_action_output_type(switch_action, result_case1)}")
    print(f"  Next: {[aid for aid, _, _ in next_actions]}")

    # Default case
    print("\nDefault: status = 'unknown'")
    result_default = {"success": True, "case_index": None}
    next_actions = router.route(switch_action, result_default, workflow.connections)
    print(f"  Output: {router.get_action_output_type(switch_action, result_default)}")
    print(f"  Next: {[aid for aid, _, _ in next_actions]}")


# ============================================================================
# Example 4: TRY_CATCH Action Routing
# ============================================================================


def example_try_catch_routing():
    """
    Example: Routing through a TRY_CATCH action.

    Routes to error handler on exception.
    """
    print("\n=== Example 4: TRY_CATCH Action Routing ===\n")

    workflow = Workflow(
        id="try_catch_example",
        name="TRY_CATCH Routing Example",
        version="1.0.0",
        format="graph",
        actions=[
            Action(id="start", type="CLICK", config={}, position=(0, 100)),
            Action(
                id="try_catch",
                type="TRY_CATCH",
                config={
                    "tryActions": ["risky_operation"],
                    "catchActions": ["log_error"],
                    "finallyActions": ["cleanup"],
                },
                position=(200, 100),
            ),
            Action(
                id="success_handler",
                type="SET_VARIABLE",
                config={"variableName": "success", "value": True},
                position=(400, 50),
            ),
            Action(
                id="error_handler", type="SCREENSHOT", config={}, position=(400, 150)
            ),
        ],
        connections=Connections(
            root={
                "start": {
                    "main": [[Connection(action="try_catch", type="main", index=0)]]
                },
                "try_catch": {
                    "main": [
                        [Connection(action="success_handler", type="main", index=0)]
                    ],
                    "error": [
                        [Connection(action="error_handler", type="error", index=0)]
                    ],
                },
            }
        ),
    )

    router = ConnectionRouter(workflow=workflow)
    try_catch_action = workflow.actions[1]

    # Success case
    print("Success: No exception")
    result_success = {"success": True}
    next_actions = router.route(try_catch_action, result_success, workflow.connections)
    print(
        f"  Output: {router.get_action_output_type(try_catch_action, result_success)}"
    )
    print(f"  Next: {[aid for aid, _, _ in next_actions]}")

    # Error case
    print("\nError: Exception occurred")
    result_error = {"success": False, "error": "ValueError: Invalid input"}
    next_actions = router.route(try_catch_action, result_error, workflow.connections)
    print(f"  Output: {router.get_action_output_type(try_catch_action, result_error)}")
    print(f"  Next: {[aid for aid, _, _ in next_actions]}")


# ============================================================================
# Example 5: Complex Workflow with Multiple Action Types
# ============================================================================


def example_complex_workflow():
    """
    Example: Complex workflow combining multiple action types.

    Flow: start -> if -> loop -> switch -> end
    """
    print("\n=== Example 5: Complex Workflow Routing ===\n")

    workflow = Workflow(
        id="complex_example",
        name="Complex Routing Example",
        version="1.0.0",
        format="graph",
        actions=[
            Action(id="start", type="CLICK", config={}, position=(0, 200)),
            Action(
                id="check_condition",
                type="IF",
                config={
                    "condition": {"type": "expression", "expression": "count > 0"},
                    "thenActions": ["process_loop"],
                    "elseActions": ["skip_to_end"],
                },
                position=(200, 200),
            ),
            Action(
                id="process_loop",
                type="LOOP",
                config={"loopType": "FOR", "iterations": 3, "actions": ["loop_body"]},
                position=(400, 100),
            ),
            Action(
                id="route_by_status",
                type="SWITCH",
                config={
                    "expression": "item_status",
                    "cases": [
                        {"value": "valid", "actions": ["process_valid"]},
                        {"value": "invalid", "actions": ["process_invalid"]},
                    ],
                    "defaultActions": ["process_unknown"],
                },
                position=(600, 100),
            ),
            Action(id="end", type="SCREENSHOT", config={}, position=(800, 200)),
        ],
        connections=Connections(
            root={
                "start": {
                    "main": [
                        [Connection(action="check_condition", type="main", index=0)]
                    ]
                },
                "check_condition": {
                    "true": [[Connection(action="process_loop", type="true", index=0)]],
                    "false": [[Connection(action="end", type="false", index=0)]],
                },
                "process_loop": {
                    "main": [
                        [Connection(action="route_by_status", type="main", index=0)]
                    ]
                },
                "route_by_status": {
                    "case_0": [[Connection(action="end", type="case_0", index=0)]],
                    "case_1": [[Connection(action="end", type="case_1", index=0)]],
                    "default": [[Connection(action="end", type="default", index=0)]],
                },
            }
        ),
    )

    context = RoutingContext()
    router = ConnectionRouter(workflow=workflow, context=context)

    # Simulate execution path
    print("Execution Path Simulation:\n")

    # 1. Start
    start_action = workflow.actions[0]
    next_actions = router.route(start_action, {"success": True}, workflow.connections)
    print(f"1. START -> {[aid for aid, _, _ in next_actions]}")

    # 2. IF (true)
    if_action = workflow.actions[1]
    next_actions = router.route(
        if_action, {"success": True, "condition_result": True}, workflow.connections
    )
    print(f"2. IF (true) -> {[aid for aid, _, _ in next_actions]}")

    # 3. LOOP (exit)
    loop_action = workflow.actions[2]
    next_actions = router.route(
        loop_action,
        {"success": True, "continue_loop": False, "iteration": 3},
        workflow.connections,
    )
    print(f"3. LOOP (exit) -> {[aid for aid, _, _ in next_actions]}")

    # 4. SWITCH (case 0)
    switch_action = workflow.actions[3]
    next_actions = router.route(
        switch_action, {"success": True, "case_index": 0}, workflow.connections
    )
    print(f"4. SWITCH (case_0) -> {[aid for aid, _, _ in next_actions]}")

    # Show execution statistics
    print("\nExecution Statistics:")
    stats = context.get_statistics()
    for key, value in stats.items():
        if not key.endswith("_time"):
            print(f"  {key}: {value}")

    print("\nVisual Path:")
    print(f"  {context.get_visual_path()}")


# ============================================================================
# Example 6: Routing Analysis
# ============================================================================


def example_routing_analysis():
    """
    Example: Analyzing routing options and workflow complexity.
    """
    print("\n=== Example 6: Routing Analysis ===\n")

    workflow = Workflow(
        id="analysis_example",
        name="Routing Analysis",
        version="1.0.0",
        format="graph",
        actions=[
            Action(id="start", type="CLICK", config={}, position=(0, 0)),
            Action(
                id="if1",
                type="IF",
                config={
                    "condition": {"type": "expression", "expression": "x > 0"},
                    "thenActions": ["branch_a"],
                    "elseActions": ["branch_b"],
                },
                position=(200, 0),
            ),
            Action(
                id="branch_a", type="TYPE", config={"text": "A"}, position=(400, -100)
            ),
            Action(
                id="branch_b", type="TYPE", config={"text": "B"}, position=(400, 100)
            ),
            Action(id="merge", type="SCREENSHOT", config={}, position=(600, 0)),
        ],
        connections=Connections(
            root={
                "start": {"main": [[Connection(action="if1", type="main", index=0)]]},
                "if1": {
                    "true": [[Connection(action="branch_a", type="true", index=0)]],
                    "false": [[Connection(action="branch_b", type="false", index=0)]],
                },
                "branch_a": {
                    "main": [[Connection(action="merge", type="main", index=0)]]
                },
                "branch_b": {
                    "main": [[Connection(action="merge", type="main", index=0)]]
                },
            }
        ),
    )

    router = ConnectionRouter(workflow=workflow)

    # Get routing options for IF action
    if_action = workflow.actions[1]
    options = router.get_routing_options(if_action, workflow.connections)
    print("Routing options for IF action:")
    for output_type, targets in options.items():
        print(f"  {output_type}: {targets}")

    # Analyze convergence points
    convergence = router.analyze_convergence_points(
        workflow.connections, [a.id for a in workflow.actions]
    )
    print("\nConvergence points (merge nodes):")
    for action_id, sources in convergence.items():
        print(f"  {action_id}: {len(sources)} incoming connections")
        for source, output_type in sources:
            print(f"    <- {source} ({output_type})")

    # Find execution paths
    paths = router.get_execution_paths(
        "start", "merge", workflow.connections, max_paths=10
    )
    print("\nAll execution paths from start to merge:")
    for i, path in enumerate(paths, 1):
        print(f"  Path {i}: {' -> '.join(path)}")

    # Estimate complexity
    complexity = router.estimate_execution_complexity(workflow)
    print("\nWorkflow complexity metrics:")
    for key, value in complexity.items():
        print(f"  {key}: {value}")


# ============================================================================
# Run All Examples
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CONNECTION ROUTING EXAMPLES")
    print("=" * 80)

    example_if_routing()
    example_loop_routing()
    example_switch_routing()
    example_try_catch_routing()
    example_complex_workflow()
    example_routing_analysis()

    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80 + "\n")
