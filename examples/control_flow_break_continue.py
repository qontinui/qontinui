"""Example demonstrating BREAK and CONTINUE in control flow.

This example shows how to use break and continue statements to control
loop execution dynamically.
"""

import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qontinui.actions.control_flow import BreakLoop, ContinueLoop, ControlFlowExecutor
from qontinui.config import Action


def main():
    """Run break/continue examples."""
    print("=" * 70)
    print("Control Flow: BREAK and CONTINUE Examples")
    print("=" * 70)

    # ========================================================================
    # Example 1: BREAK - Stop when target found
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 1: BREAK - Search until found")
    print("=" * 70)

    items_to_search = [
        {"id": 1, "name": "item-1", "target": False},
        {"id": 2, "name": "item-2", "target": False},
        {"id": 3, "name": "item-3", "target": True},  # Target is here
        {"id": 4, "name": "item-4", "target": False},
        {"id": 5, "name": "item-5", "target": False},
    ]

    def search_executor(action_id: str, variables: dict) -> dict:
        """Executor that searches for target and breaks when found."""
        current_item = variables.get("item")

        if action_id == "check-item":
            print(f"  Checking item {current_item['id']}: {current_item['name']}")
            if current_item.get("target"):
                print("  *** Found target! Breaking loop ***")
                raise BreakLoop("Target found")

        return {"success": True}

    executor = ControlFlowExecutor(
        action_executor=search_executor, variables={"items": items_to_search}
    )

    search_loop = Action(
        id="search-loop",
        type="LOOP",
        config={
            "loopType": "FOREACH",
            "collection": {"type": "variable", "variableName": "items"},
            "iteratorVariable": "item",
            "actions": ["check-item"],
        },
    )

    result = executor.execute_loop(search_loop)
    print(
        f"\nResult: Checked {result['iterations_completed']}/{len(items_to_search)} items"
    )
    print(f"Stopped early: {result['stopped_early']}")
    if result.get("break_message"):
        print(f"Break reason: {result['break_message']}")

    # ========================================================================
    # Example 2: CONTINUE - Skip invalid items
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 2: CONTINUE - Process only valid items")
    print("=" * 70)

    items = [
        {"id": 1, "name": "valid-1", "valid": True},
        {"id": 2, "name": "invalid-1", "valid": False},  # Skip this
        {"id": 3, "name": "valid-2", "valid": True},
        {"id": 4, "name": "invalid-2", "valid": False},  # Skip this
        {"id": 5, "name": "valid-3", "valid": True},
    ]

    processed_items = []

    def validation_executor(action_id: str, variables: dict) -> dict:
        """Executor that skips invalid items."""
        current_item = variables.get("item")

        if action_id == "validate-item":
            print(f"  Validating item {current_item['id']}: {current_item['name']}")
            if not current_item.get("valid"):
                print("    Invalid - skipping remaining actions")
                raise ContinueLoop("Item is invalid")

        elif action_id == "process-item":
            print(f"  Processing valid item {current_item['id']}")
            processed_items.append(current_item)

        return {"success": True}

    executor = ControlFlowExecutor(
        action_executor=validation_executor, variables={"items": items}
    )

    process_loop = Action(
        id="process-loop",
        type="LOOP",
        config={
            "loopType": "FOREACH",
            "collection": {"type": "variable", "variableName": "items"},
            "iteratorVariable": "item",
            "actions": [
                "validate-item",
                "process-item",
            ],  # process-item skipped if invalid
        },
    )

    result = executor.execute_loop(process_loop)
    print(f"\nResult: {result['iterations_completed']} total iterations")
    print(f"Processed items: {len(processed_items)}/{len(items)}")
    for item in processed_items:
        print(f"  - {item['name']}")

    # ========================================================================
    # Example 3: Conditional BREAK - Stop at threshold
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 3: Conditional BREAK - Stop at threshold")
    print("=" * 70)

    executor = ControlFlowExecutor(variables={"sum": 0, "threshold": 50})

    def accumulator_executor(action_id: str, variables: dict) -> dict:
        """Executor that accumulates values and breaks at threshold."""
        if action_id == "add-value":
            value = variables.get("value")
            variables["sum"] += value
            print(f"  Added {value}, sum = {variables['sum']}")

        elif action_id == "check-threshold":
            if variables["sum"] >= variables["threshold"]:
                print(
                    f"  *** Threshold reached ({variables['threshold']}) - breaking ***"
                )
                raise BreakLoop(f"Sum reached threshold: {variables['sum']}")

        return {"success": True}

    executor.action_executor = accumulator_executor

    accumulate_loop = Action(
        id="accumulate-loop",
        type="LOOP",
        config={
            "loopType": "FOREACH",
            "collection": {"type": "range", "start": 1, "end": 20},
            "iteratorVariable": "value",
            "actions": ["add-value", "check-threshold"],
        },
    )

    result = executor.execute_loop(accumulate_loop)
    print(f"\nResult: {result['iterations_completed']} iterations")
    print(f"Final sum: {executor.variables['sum']}")
    print(f"Stopped early: {result['stopped_early']}")

    # ========================================================================
    # Example 4: CONTINUE with condition - Skip even numbers
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 4: CONTINUE with Condition - Process odd numbers only")
    print("=" * 70)

    odd_numbers = []

    def odd_filter_executor(action_id: str, variables: dict) -> dict:
        """Executor that processes only odd numbers."""
        num = variables.get("num")

        if action_id == "check-even":
            if num % 2 == 0:
                print(f"  Skipping even number: {num}")
                raise ContinueLoop(f"Number {num} is even")

        elif action_id == "process-odd":
            print(f"  Processing odd number: {num}")
            odd_numbers.append(num)

        return {"success": True}

    executor = ControlFlowExecutor(action_executor=odd_filter_executor)

    odd_loop = Action(
        id="odd-loop",
        type="LOOP",
        config={
            "loopType": "FOREACH",
            "collection": {"type": "range", "start": 1, "end": 11},
            "iteratorVariable": "num",
            "actions": ["check-even", "process-odd"],
        },
    )

    result = executor.execute_loop(odd_loop)
    print(f"\nResult: {result['iterations_completed']} iterations")
    print(f"Odd numbers found: {odd_numbers}")

    # ========================================================================
    # Example 5: Complex workflow - Early exit with cleanup
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 5: Complex Workflow - Early exit with state")
    print("=" * 70)

    executor = ControlFlowExecutor(
        variables={
            "attempts": 0,
            "max_attempts": 3,
            "success": False,
            "retry_errors": [],
        }
    )

    def retry_executor(action_id: str, variables: dict) -> dict:
        """Executor that retries operations with early exit on success."""
        variables["attempts"] += 1
        attempt_num = variables["attempts"]

        if action_id == "try-operation":
            print(f"  Attempt {attempt_num}/{variables['max_attempts']}")

            # Simulate success on 3rd attempt
            if attempt_num >= 3:
                print("    Operation succeeded!")
                variables["success"] = True
            else:
                print("    Operation failed")
                variables["retry_errors"].append(f"Attempt {attempt_num} failed")

        elif action_id == "check-success":
            if variables["success"]:
                print("  *** Success! Breaking retry loop ***")
                raise BreakLoop("Operation successful")

        elif action_id == "check-max-attempts":
            if variables["attempts"] >= variables["max_attempts"]:
                print("  *** Max attempts reached, stopping ***")
                raise BreakLoop("Max attempts exceeded")

        return {"success": True}

    executor.action_executor = retry_executor

    retry_loop = Action(
        id="retry-loop",
        type="LOOP",
        config={
            "loopType": "WHILE",
            "condition": {
                "type": "expression",
                "expression": "True",  # Infinite loop, controlled by break
            },
            "actions": ["try-operation", "check-success", "check-max-attempts"],
            "maxIterations": 10,  # Safety limit
        },
    )

    result = executor.execute_loop(retry_loop)
    print("\nResult:")
    print(f"  Total attempts: {executor.variables['attempts']}")
    print(f"  Success: {executor.variables['success']}")
    print(f"  Errors: {len(executor.variables['retry_errors'])}")
    print(f"  Stopped early: {result['stopped_early']}")
    if result.get("break_message"):
        print(f"  Reason: {result['break_message']}")

    # ========================================================================
    # Example 6: Multiple conditions - BREAK or CONTINUE
    # ========================================================================
    print("\n" + "=" * 70)
    print("Example 6: Complex Logic - Multiple Break/Continue Conditions")
    print("=" * 70)

    data_points = [
        {"value": 5, "status": "ok"},
        {"value": 15, "status": "warning"},  # Skip this
        {"value": 25, "status": "ok"},
        {"value": 100, "status": "critical"},  # Break here
        {"value": 35, "status": "ok"},  # Never reached
    ]

    processed = []

    def complex_executor(action_id: str, variables: dict) -> dict:
        """Executor with multiple break/continue conditions."""
        point = variables.get("point")

        if action_id == "check-status":
            print(f"  Checking point: value={point['value']}, status={point['status']}")

            if point["status"] == "warning":
                print("    Warning status - skipping")
                raise ContinueLoop("Warning status")

            if point["status"] == "critical":
                print("    *** Critical status - aborting ***")
                raise BreakLoop("Critical status encountered")

        elif action_id == "process-point":
            print(f"    Processing point: {point['value']}")
            processed.append(point)

        return {"success": True}

    executor = ControlFlowExecutor(
        action_executor=complex_executor, variables={"points": data_points}
    )

    complex_loop = Action(
        id="complex-loop",
        type="LOOP",
        config={
            "loopType": "FOREACH",
            "collection": {"type": "variable", "variableName": "points"},
            "iteratorVariable": "point",
            "actions": ["check-status", "process-point"],
        },
    )

    result = executor.execute_loop(complex_loop)
    print("\nResult:")
    print(f"  Total iterations: {result['iterations_completed']}/{len(data_points)}")
    print(f"  Processed: {len(processed)}")
    print(f"  Stopped early: {result['stopped_early']}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nAll examples completed successfully!")
    print("\nKey takeaways:")
    print("  1. BREAK exits the loop immediately")
    print("  2. CONTINUE skips remaining actions in current iteration")
    print("  3. Both can be conditional or unconditional")
    print("  4. Useful for search, validation, retry logic")
    print("  5. Helps optimize loops by avoiding unnecessary work")
    print("  6. Can implement complex control flow patterns")


if __name__ == "__main__":
    main()
