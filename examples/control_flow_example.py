"""Example demonstrating control flow actions in qontinui.

This script shows how to use LOOP, IF, BREAK, and CONTINUE actions
with the ControlFlowExecutor class.
"""

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qontinui.actions.control_flow import ControlFlowExecutor
from qontinui.config import Action


def main():
    """Run control flow examples."""
    print("=" * 70)
    print("Qontinui Control Flow Examples")
    print("=" * 70)

    # Create mock action executor that tracks what was executed
    executed_actions = []

    def mock_executor(action_id: str, variables: dict) -> dict:
        """Mock action executor that logs execution."""
        executed_actions.append({"action_id": action_id, "variables": variables.copy()})
        print(f"  Executed: {action_id} with variables: {variables}")
        return {"success": True}

    # Initialize executor
    executor = ControlFlowExecutor(action_executor=mock_executor)

    # ========================================================================
    # Example 1: Simple FOR loop
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 1: FOR Loop (3 iterations)")
    print("=" * 70)

    executed_actions.clear()
    loop_action = Action(
        id="example-1",
        type="LOOP",
        config={
            "loopType": "FOR",
            "iterations": 3,
            "iteratorVariable": "i",
            "actions": ["print-number", "process-data"],
        },
    )

    result = executor.execute_loop(loop_action)
    print(f"\nResult: {result['iterations_completed']} iterations completed")
    print(f"Success: {result['success']}")

    # ========================================================================
    # Example 2: WHILE loop with variable condition
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 2: WHILE Loop (count to 5)")
    print("=" * 70)

    executed_actions.clear()
    executor.variables = {"counter": 0}

    def incrementing_executor(action_id: str, variables: dict) -> dict:
        """Executor that increments counter."""
        variables["counter"] = variables.get("counter", 0) + 1
        print(f"  Executed: {action_id}, counter = {variables['counter']}")
        return {"success": True}

    executor.action_executor = incrementing_executor

    while_action = Action(
        id="example-2",
        type="LOOP",
        config={
            "loopType": "WHILE",
            "condition": {
                "type": "variable",
                "variableName": "counter",
                "operator": "<",
                "expectedValue": 5,
            },
            "actions": ["increment"],
        },
    )

    result = executor.execute_loop(while_action)
    print(f"\nResult: {result['iterations_completed']} iterations completed")
    print(f"Final counter value: {executor.variables['counter']}")
    print(f"Success: {result['success']}")

    # ========================================================================
    # Example 3: FOREACH loop
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 3: FOREACH Loop (iterate over list)")
    print("=" * 70)

    executed_actions.clear()
    executor.action_executor = mock_executor
    executor.variables = {"fruits": ["apple", "banana", "cherry"]}

    foreach_action = Action(
        id="example-3",
        type="LOOP",
        config={
            "loopType": "FOREACH",
            "collection": {"type": "variable", "variableName": "fruits"},
            "iteratorVariable": "fruit",
            "actions": ["process-fruit"],
        },
    )

    result = executor.execute_loop(foreach_action)
    print(f"\nResult: {result['iterations_completed']} iterations completed")
    print(f"Success: {result['success']}")

    # ========================================================================
    # Example 4: IF/ELSE conditional
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 4: IF/ELSE Conditional")
    print("=" * 70)

    executed_actions.clear()
    executor.variables = {"temperature": 25}

    print("\nTest 1: temperature = 25 (warm)")
    if_action = Action(
        id="example-4a",
        type="IF",
        config={
            "condition": {
                "type": "variable",
                "variableName": "temperature",
                "operator": ">",
                "expectedValue": 20,
            },
            "thenActions": ["turn-on-ac"],
            "elseActions": ["turn-on-heater"],
        },
    )

    result = executor.execute_if(if_action)
    print(f"Condition result: {result['condition_result']}")
    print(f"Branch taken: {result['branch_taken']}")

    print("\nTest 2: temperature = 15 (cold)")
    executed_actions.clear()
    executor.variables["temperature"] = 15

    result = executor.execute_if(if_action)
    print(f"Condition result: {result['condition_result']}")
    print(f"Branch taken: {result['branch_taken']}")

    # ========================================================================
    # Example 5: Expression-based condition
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 5: IF with Expression Condition")
    print("=" * 70)

    executed_actions.clear()
    executor.variables = {"x": 10, "y": 20}

    expr_if_action = Action(
        id="example-5",
        type="IF",
        config={
            "condition": {"type": "expression", "expression": "x + y > 25"},
            "thenActions": ["log-sum-large"],
            "elseActions": ["log-sum-small"],
        },
    )

    result = executor.execute_if(expr_if_action)
    print(f"Expression 'x + y > 25' evaluated to: {result['condition_result']}")
    print(f"Branch taken: {result['branch_taken']}")

    # ========================================================================
    # Example 6: FOREACH with range
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 6: FOREACH with Range")
    print("=" * 70)

    executed_actions.clear()

    range_foreach_action = Action(
        id="example-6",
        type="LOOP",
        config={
            "loopType": "FOREACH",
            "collection": {"type": "range", "start": 0, "end": 10, "step": 2},
            "iteratorVariable": "num",
            "actions": ["process-even-number"],
        },
    )

    result = executor.execute_loop(range_foreach_action)
    print(f"\nResult: {result['iterations_completed']} iterations completed")
    print(f"Success: {result['success']}")

    # ========================================================================
    # Example 7: Loop with max_iterations safety
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 7: FOR Loop with max_iterations Safety")
    print("=" * 70)

    executed_actions.clear()

    safe_loop_action = Action(
        id="example-7",
        type="LOOP",
        config={
            "loopType": "FOR",
            "iterations": 1000000,  # Very large number
            "maxIterations": 10,  # Safety limit
            "actions": ["process-item"],
        },
    )

    result = executor.execute_loop(safe_loop_action)
    print("\nRequested iterations: 1,000,000")
    print(
        f"Actual iterations: {result['iterations_completed']} (capped by maxIterations)"
    )
    print(f"Success: {result['success']}")

    # ========================================================================
    # Example 8: Nested loops simulation
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 8: Simulated Nested Loops")
    print("=" * 70)

    print("\nOuter loop (3 iterations):")
    for outer_i in range(3):
        print(f"\n  Outer iteration {outer_i}:")
        print("  Inner loop (2 iterations):")
        for inner_i in range(2):
            print(f"    Inner iteration {inner_i}")

    print(
        "\nNote: Actual nested loops would use action references to inner loop actions"
    )

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nAll examples completed successfully!")
    print("\nFeatures demonstrated:")
    print("  - FOR loops with fixed iterations")
    print("  - WHILE loops with conditions")
    print("  - FOREACH loops over collections and ranges")
    print("  - IF/ELSE conditionals")
    print("  - Expression-based conditions")
    print("  - Variable management")
    print("  - Safety limits (max_iterations)")
    print("\nAdditional features (not shown):")
    print("  - BREAK and CONTINUE statements")
    print("  - Conditional breaks/continues")
    print("  - Error handling with break_on_error")
    print("  - Multiple comparison operators")
    print("  - Match-based collections (with image finding)")


if __name__ == "__main__":
    main()
