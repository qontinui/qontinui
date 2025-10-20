"""
Example workflows demonstrating merge node support.

This module provides practical examples of:
1. Simple IF with merge
2. Parallel processing with merge
3. Complex branching with multiple merges
4. Error handling in merge scenarios
"""

from qontinui.config.schema import Action, Connection, Connections, Workflow
from qontinui.execution.merge_handler import MergeHandler

# ============================================================================
# Example 1: Simple IF with Merge
# ============================================================================


def create_if_merge_example() -> Workflow:
    """
    Create a workflow demonstrating IF statement with merge.

    Flow:
        START -> CHECK_CONDITION -> IF_ACTION
                                      ├─(true)─> TRUE_BRANCH ──┐
                                      │                         ├─> MERGE -> FINAL
                                      └─(false)> FALSE_BRANCH ─┘

    Use case: Check if a file exists, then either process it (true) or
    create it (false), then continue with common steps.
    """
    actions = [
        Action(
            id="start",
            type="SET_VARIABLE",
            name="Set initial state",
            config={"variableName": "file_exists", "value": True, "scope": "process"},
            position=(0, 100),
        ),
        Action(
            id="check_file",
            type="IF",
            name="Check if file exists",
            config={
                "condition": {
                    "type": "variable",
                    "variableName": "file_exists",
                    "operator": "==",
                    "expectedValue": True,
                },
                "thenActions": ["process_file"],
                "elseActions": ["create_file"],
            },
            position=(200, 100),
        ),
        Action(
            id="process_file",
            type="SET_VARIABLE",
            name="Process existing file",
            config={"variableName": "result", "value": "File processed", "scope": "process"},
            position=(400, 50),
        ),
        Action(
            id="create_file",
            type="SET_VARIABLE",
            name="Create new file",
            config={"variableName": "result", "value": "File created", "scope": "process"},
            position=(400, 150),
        ),
        Action(
            id="merge_point",
            type="SET_VARIABLE",
            name="Continue after file handling",
            config={"variableName": "status", "value": "Ready to proceed", "scope": "process"},
            position=(600, 100),
        ),
        Action(
            id="final_step",
            type="SET_VARIABLE",
            name="Complete workflow",
            config={"variableName": "complete", "value": True, "scope": "process"},
            position=(800, 100),
        ),
    ]

    connections = Connections(
        root={
            "start": {"main": [[Connection(action="check_file", type="main", index=0)]]},
            "check_file": {
                "true": [[Connection(action="process_file", type="true", index=0)]],
                "false": [[Connection(action="create_file", type="false", index=0)]],
            },
            "process_file": {"main": [[Connection(action="merge_point", type="main", index=0)]]},
            "create_file": {"main": [[Connection(action="merge_point", type="main", index=1)]]},
            "merge_point": {"main": [[Connection(action="final_step", type="main", index=0)]]},
        }
    )

    return Workflow(
        id="if_merge_example",
        name="IF with Merge Example",
        version="1.0.0",
        format="graph",
        actions=actions,
        connections=connections,
        metadata={
            "description": "Demonstrates IF statement with branches that reconverge",
            "author": "qontinui",
        },
    )


# ============================================================================
# Example 2: Parallel Processing with Merge
# ============================================================================


def create_parallel_processing_example() -> Workflow:
    """
    Create a workflow demonstrating parallel processing with merge.

    Flow:
        START -> SPLIT_DATA ──┬─> PROCESS_A ──┐
                              ├─> PROCESS_B ──┤
                              └─> PROCESS_C ──┴─> MERGE_RESULTS -> FINALIZE

    Use case: Process different data chunks in parallel, then merge results.
    """
    actions = [
        Action(
            id="init_data",
            type="SET_VARIABLE",
            name="Initialize data",
            config={
                "variableName": "data",
                "value": {"chunk_a": [1, 2, 3], "chunk_b": [4, 5, 6], "chunk_c": [7, 8, 9]},
                "scope": "process",
            },
            position=(0, 150),
        ),
        Action(
            id="process_chunk_a",
            type="SET_VARIABLE",
            name="Process chunk A",
            config={"variableName": "result_a", "value": "Processed A", "scope": "process"},
            position=(200, 50),
        ),
        Action(
            id="process_chunk_b",
            type="SET_VARIABLE",
            name="Process chunk B",
            config={"variableName": "result_b", "value": "Processed B", "scope": "process"},
            position=(200, 150),
        ),
        Action(
            id="process_chunk_c",
            type="SET_VARIABLE",
            name="Process chunk C",
            config={"variableName": "result_c", "value": "Processed C", "scope": "process"},
            position=(200, 250),
        ),
        Action(
            id="merge_results",
            type="SET_VARIABLE",
            name="Merge all results",
            config={
                "variableName": "all_results",
                "value": "All chunks processed",
                "scope": "process",
            },
            position=(400, 150),
        ),
        Action(
            id="finalize",
            type="SET_VARIABLE",
            name="Finalize processing",
            config={"variableName": "complete", "value": True, "scope": "process"},
            position=(600, 150),
        ),
    ]

    connections = Connections(
        root={
            "init_data": {
                "main": [
                    [Connection(action="process_chunk_a", type="main", index=0)],
                    [Connection(action="process_chunk_b", type="main", index=0)],
                    [Connection(action="process_chunk_c", type="main", index=0)],
                ]
            },
            "process_chunk_a": {
                "main": [[Connection(action="merge_results", type="main", index=0)]]
            },
            "process_chunk_b": {
                "main": [[Connection(action="merge_results", type="main", index=1)]]
            },
            "process_chunk_c": {
                "main": [[Connection(action="merge_results", type="main", index=2)]]
            },
            "merge_results": {"main": [[Connection(action="finalize", type="main", index=0)]]},
        }
    )

    return Workflow(
        id="parallel_merge_example",
        name="Parallel Processing with Merge",
        version="1.0.0",
        format="graph",
        actions=actions,
        connections=connections,
        metadata={
            "description": "Demonstrates parallel processing with result merging",
            "author": "qontinui",
        },
    )


# ============================================================================
# Example 3: Complex Branching with Multiple Merges
# ============================================================================


def create_complex_merge_example() -> Workflow:
    """
    Create a workflow with multiple merge points.

    Flow:
        START ──┬─> PATH_A1 ──┐
                │              ├─> MERGE_1 ──┐
                ├─> PATH_A2 ──┘              │
                │                            ├─> FINAL_MERGE -> END
                ├─> PATH_B1 ──┐              │
                │              ├─> MERGE_2 ──┘
                └─> PATH_B2 ──┘

    Use case: Complex workflow with multiple parallel branches that merge
    at different points before final merge.
    """
    actions = [
        Action(
            id="start",
            type="SET_VARIABLE",
            name="Start workflow",
            config={"variableName": "workflow_started", "value": True, "scope": "process"},
            position=(0, 150),
        ),
        Action(
            id="path_a1",
            type="SET_VARIABLE",
            name="Path A1",
            config={"variableName": "a1_result", "value": "A1 complete", "scope": "process"},
            position=(200, 50),
        ),
        Action(
            id="path_a2",
            type="SET_VARIABLE",
            name="Path A2",
            config={"variableName": "a2_result", "value": "A2 complete", "scope": "process"},
            position=(200, 100),
        ),
        Action(
            id="path_b1",
            type="SET_VARIABLE",
            name="Path B1",
            config={"variableName": "b1_result", "value": "B1 complete", "scope": "process"},
            position=(200, 200),
        ),
        Action(
            id="path_b2",
            type="SET_VARIABLE",
            name="Path B2",
            config={"variableName": "b2_result", "value": "B2 complete", "scope": "process"},
            position=(200, 250),
        ),
        Action(
            id="merge_1",
            type="SET_VARIABLE",
            name="Merge A paths",
            config={"variableName": "a_merged", "value": "A paths merged", "scope": "process"},
            position=(400, 75),
        ),
        Action(
            id="merge_2",
            type="SET_VARIABLE",
            name="Merge B paths",
            config={"variableName": "b_merged", "value": "B paths merged", "scope": "process"},
            position=(400, 225),
        ),
        Action(
            id="final_merge",
            type="SET_VARIABLE",
            name="Final merge",
            config={"variableName": "all_merged", "value": "All paths merged", "scope": "process"},
            position=(600, 150),
        ),
        Action(
            id="end",
            type="SET_VARIABLE",
            name="Complete",
            config={"variableName": "complete", "value": True, "scope": "process"},
            position=(800, 150),
        ),
    ]

    connections = Connections(
        root={
            "start": {
                "main": [
                    [Connection(action="path_a1", type="main", index=0)],
                    [Connection(action="path_a2", type="main", index=0)],
                    [Connection(action="path_b1", type="main", index=0)],
                    [Connection(action="path_b2", type="main", index=0)],
                ]
            },
            "path_a1": {"main": [[Connection(action="merge_1", type="main", index=0)]]},
            "path_a2": {"main": [[Connection(action="merge_1", type="main", index=1)]]},
            "path_b1": {"main": [[Connection(action="merge_2", type="main", index=0)]]},
            "path_b2": {"main": [[Connection(action="merge_2", type="main", index=1)]]},
            "merge_1": {"main": [[Connection(action="final_merge", type="main", index=0)]]},
            "merge_2": {"main": [[Connection(action="final_merge", type="main", index=1)]]},
            "final_merge": {"main": [[Connection(action="end", type="main", index=0)]]},
        }
    )

    return Workflow(
        id="complex_merge_example",
        name="Complex Multi-Merge Workflow",
        version="1.0.0",
        format="graph",
        actions=actions,
        connections=connections,
        metadata={
            "description": "Demonstrates complex workflow with multiple merge points",
            "author": "qontinui",
        },
    )


# ============================================================================
# Example 4: Error Handling in Merge Scenarios
# ============================================================================


def create_error_handling_example() -> Workflow:
    """
    Create a workflow demonstrating error handling with merges.

    Flow:
        START ──┬─> RISKY_OP_A ──┬─(success)─> SUCCESS_A ──┐
                │                 └─(error)───> ERROR_A ────┤
                │                                            ├─> MERGE -> CLEANUP
                └─> RISKY_OP_B ──┬─(success)─> SUCCESS_B ──┤
                                 └─(error)───> ERROR_B ────┘

    Use case: Parallel operations that might fail, with error handling
    paths that merge with success paths.
    """
    actions = [
        Action(
            id="start",
            type="SET_VARIABLE",
            name="Start",
            config={"variableName": "started", "value": True, "scope": "process"},
            position=(0, 150),
        ),
        Action(
            id="risky_op_a",
            type="TRY_CATCH",
            name="Risky Operation A",
            config={
                "tryActions": ["success_a"],
                "catchActions": ["error_a"],
                "errorVariable": "error_a_msg",
            },
            position=(200, 50),
        ),
        Action(
            id="risky_op_b",
            type="TRY_CATCH",
            name="Risky Operation B",
            config={
                "tryActions": ["success_b"],
                "catchActions": ["error_b"],
                "errorVariable": "error_b_msg",
            },
            position=(200, 250),
        ),
        Action(
            id="success_a",
            type="SET_VARIABLE",
            name="Success A",
            config={"variableName": "result_a", "value": "A succeeded", "scope": "process"},
            position=(400, 20),
        ),
        Action(
            id="error_a",
            type="SET_VARIABLE",
            name="Error A",
            config={"variableName": "result_a", "value": "A failed", "scope": "process"},
            position=(400, 80),
        ),
        Action(
            id="success_b",
            type="SET_VARIABLE",
            name="Success B",
            config={"variableName": "result_b", "value": "B succeeded", "scope": "process"},
            position=(400, 220),
        ),
        Action(
            id="error_b",
            type="SET_VARIABLE",
            name="Error B",
            config={"variableName": "result_b", "value": "B failed", "scope": "process"},
            position=(400, 280),
        ),
        Action(
            id="merge_point",
            type="SET_VARIABLE",
            name="Merge all paths",
            config={"variableName": "merged", "value": "All paths merged", "scope": "process"},
            position=(600, 150),
        ),
        Action(
            id="cleanup",
            type="SET_VARIABLE",
            name="Cleanup",
            config={"variableName": "cleaned_up", "value": True, "scope": "process"},
            position=(800, 150),
        ),
    ]

    connections = Connections(
        root={
            "start": {
                "main": [
                    [Connection(action="risky_op_a", type="main", index=0)],
                    [Connection(action="risky_op_b", type="main", index=0)],
                ]
            },
            "risky_op_a": {
                "success": [[Connection(action="success_a", type="success", index=0)]],
                "error": [[Connection(action="error_a", type="error", index=0)]],
            },
            "risky_op_b": {
                "success": [[Connection(action="success_b", type="success", index=0)]],
                "error": [[Connection(action="error_b", type="error", index=0)]],
            },
            "success_a": {"main": [[Connection(action="merge_point", type="main", index=0)]]},
            "error_a": {"main": [[Connection(action="merge_point", type="main", index=1)]]},
            "success_b": {"main": [[Connection(action="merge_point", type="main", index=2)]]},
            "error_b": {"main": [[Connection(action="merge_point", type="main", index=3)]]},
            "merge_point": {"main": [[Connection(action="cleanup", type="main", index=0)]]},
        }
    )

    return Workflow(
        id="error_handling_merge",
        name="Error Handling with Merge",
        version="1.0.0",
        format="graph",
        actions=actions,
        connections=connections,
        metadata={
            "description": "Demonstrates error handling paths merging with success paths",
            "author": "qontinui",
        },
    )


# ============================================================================
# Demo Functions
# ============================================================================


def demonstrate_merge_detection():
    """Demonstrate merge node detection."""
    print("=" * 80)
    print("Merge Node Detection Demo")
    print("=" * 80)

    workflows = [
        ("Simple IF Merge", create_if_merge_example()),
        ("Parallel Processing", create_parallel_processing_example()),
        ("Complex Multi-Merge", create_complex_merge_example()),
        ("Error Handling", create_error_handling_example()),
    ]

    for name, workflow in workflows:
        print(f"\n{name}:")
        print(f"  Total actions: {len(workflow.actions)}")

        action_map = {action.id: action for action in workflow.actions}
        handler = MergeHandler(workflow.connections, action_map)

        merge_points = handler.get_all_merge_points()
        print(f"  Merge points: {len(merge_points)}")

        for merge_id in merge_points:
            status = handler.get_merge_status(merge_id)
            print(f"    - {merge_id}: {status['total_paths']} incoming paths")


def demonstrate_merge_execution():
    """Demonstrate merge execution with WaitAll strategy."""
    print("\n" + "=" * 80)
    print("Merge Execution Demo - WaitAll Strategy")
    print("=" * 80)

    workflow = create_parallel_processing_example()
    action_map = {action.id: action for action in workflow.actions}
    handler = MergeHandler(workflow.connections, action_map)

    print("\nSimulating parallel execution...")

    # Simulate execution of parallel paths
    paths = ["process_chunk_a", "process_chunk_b", "process_chunk_c"]

    for i, path_id in enumerate(paths, 1):
        print(f"\nPath {i} completed: {path_id}")
        is_ready = handler.register_arrival(
            "merge_results",
            path_id,
            {"success": True, "context": {f"result_{path_id}": f"Data from {path_id}"}},
        )

        status = handler.get_merge_status("merge_results")
        print(f"  Merge status: {status['arrived_paths']}/{status['total_paths']} paths arrived")
        print(f"  Ready to execute: {is_ready}")

    # Get merged context
    print("\nMerged context:")
    merged = handler.get_merged_context("merge_results")
    for key, value in merged.items():
        print(f"  {key}: {value}")


def demonstrate_timeout_strategy():
    """Demonstrate timeout-based merge strategy."""
    print("\n" + "=" * 80)
    print("Timeout Strategy Demo")
    print("=" * 80)

    workflow = create_parallel_processing_example()
    action_map = {action.id: action for action in workflow.actions}
    handler = MergeHandler(workflow.connections, action_map)

    # Simulate timeout scenario
    print("\nScenario: 2 of 3 paths complete, third times out")
    print("(In real execution, timeout would trigger after specified duration)")

    handler.register_arrival(
        "merge_results", "process_chunk_a", {"success": True, "context": {"result_a": "Data A"}}
    )

    handler.register_arrival(
        "merge_results", "process_chunk_b", {"success": True, "context": {"result_b": "Data B"}}
    )

    print("  2 paths completed")
    print("  Waiting for timeout...")
    print("  After timeout: executing with available results")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_merge_detection()
    demonstrate_merge_execution()
    demonstrate_timeout_strategy()

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
